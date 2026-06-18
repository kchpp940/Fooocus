from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ast

from .context import GenerationContext, SingleTaskContext, WorkerRuntime
from .progress import ProgressReporter, progressbar, yield_result
from .result_saver import save_images_with_metadata
from .tasks import (
    build_generation_context,
    parse_dimensions_from_aspect_ratio,
    normalize_performance_settings,
    apply_patch_settings_to_context,
    apply_overrides_to_context,
    expand_tasks_from_prompt,
    process_image_input,
    patch_samplers_from_context,
    apply_freeu_from_context,
    compute_total_steps,
    execute_diffusion_task,
)
from .stages import (
    apply_vary_to_context,
    apply_upscale_to_context,
    apply_inpaint_to_context,
    apply_control_nets_from_context,
)
from .enhance import run_enhance_pipeline


STATE_FIELDS: Set[str] = set()


@dataclass
class PipelineState:
    ctx: GenerationContext = None
    progress: int = 0
    inpaint_image: Any = None
    inpaint_mask: Any = None
    loras: List[Tuple[str, float]] = field(default_factory=list)
    tasks: List[SingleTaskContext] = field(default_factory=list)
    final_scheduler_name: str = ''
    all_steps: int = 0
    images: List[Any] = field(default_factory=list)
    processing_started: bool = False
    processing_start_time: float = 0.0
    preparation_steps: int = 0
    callback: Any = None
    progress_reporter: Optional[ProgressReporter] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# Now we can fill STATE_FIELDS
STATE_FIELDS.update(
    f.name for f in dataclass_fields(PipelineState)
)


@dataclass
class StageResult:
    state: PipelineState
    direct_return: bool = False


StageFn = Callable[[WorkerRuntime, Any, PipelineState], StageResult]


@dataclass
class StageSpec:
    name: str
    description: str
    required_state: Tuple[str, ...] = ()
    produced_state: Tuple[str, ...] = ()
    may_direct_return: bool = False
    on_failure: str = 'raise'  # 'raise' | 'skip' | 'abort'

    def validate_input(self, state: PipelineState) -> Tuple[bool, List[str]]:
        problems = []
        for fname in self.required_state:
            if fname not in STATE_FIELDS:
                problems.append(f'unknown state field: {fname}')
                continue
            value = getattr(state, fname)
            if fname == 'ctx' and value is None:
                problems.append(f'required field is None: {fname}')
            elif fname in ('progress_reporter', 'callback') and value is None:
                problems.append(f'required field is None: {fname}')
        return len(problems) == 0, problems

    def validate_output(self, state: PipelineState) -> Tuple[bool, List[str]]:
        problems = []
        for fname in self.produced_state:
            if fname not in STATE_FIELDS:
                problems.append(f'unknown state field: {fname}')
        return len(problems) == 0, problems


def _print_ctx_params(ctx: GenerationContext) -> None:
    print(f'[Parameters] Adaptive CFG = {ctx.adaptive_cfg}')
    print(f'[Parameters] CLIP Skip = {ctx.clip_skip}')
    print(f'[Parameters] Sharpness = {ctx.sharpness}')
    print(f'[Parameters] ControlNet Softness = {ctx.controlnet_softness}')
    print(
        f'[Parameters] ADM Scale = '
        f'{ctx.adm_scaler_positive} : '
        f'{ctx.adm_scaler_negative} : '
        f'{ctx.adm_scaler_end}'
    )
    print(f'[Parameters] Seed = {ctx.seed}')


spec_build_context = StageSpec(
    name='build_context',
    description='Build GenerationContext from AsyncTask, parse dimensions, normalize performance settings',
    required_state=(),
    produced_state=('ctx', 'progress'),
    on_failure='raise',
)


def stage_build_context(runtime, async_task, state):
    runtime.inpaint_worker.current_task = None

    ctx = build_generation_context(async_task)
    ctx.width, ctx.height = parse_dimensions_from_aspect_ratio(ctx.aspect_ratios_selection)

    if runtime.fooocus_expansion in ctx.style_selections:
        ctx.use_expansion = True
        ctx.style_selections.remove(runtime.fooocus_expansion)
    else:
        ctx.use_expansion = False

    ctx.use_style = len(ctx.style_selections) > 0

    state.ctx = ctx
    state.progress = normalize_performance_settings(ctx, runtime.pid, current_progress=0)
    _print_ctx_params(ctx)
    return StageResult(state=state)


spec_apply_patches = StageSpec(
    name='apply_patches',
    description='Apply pid-based patch settings, set default flags (denoising_strength, tiled, etc.)',
    required_state=('ctx', 'progress'),
    produced_state=('ctx', 'progress'),
    on_failure='raise',
)


def stage_apply_patches(runtime, async_task, state):
    apply_patch_settings_to_context(state.ctx, runtime.pid)
    print(f'[Parameters] CFG = {state.ctx.cfg_scale}')

    state.ctx.denoising_strength = 1.0
    state.ctx.tiled = False
    state.ctx.skip_prompt_processing = False
    state.ctx.inpaint_parameterized = state.ctx.inpaint_engine != 'None'
    state.ctx.use_synthetic_refiner = False
    state.progress = 1
    return StageResult(state=state)


spec_process_image_input = StageSpec(
    name='process_image_input',
    description='Process inpaint input image and extract mask if input_image_checkbox is set',
    required_state=('ctx', 'progress'),
    produced_state=('inpaint_image', 'inpaint_mask', 'progress'),
    on_failure='raise',
)


def stage_process_image_input(runtime, async_task, state):
    if state.ctx.input_image_checkbox:
        state.inpaint_image, state.inpaint_mask, state.progress = process_image_input(
            state.ctx, state.progress)
    return StageResult(state=state)


spec_load_control_models = StageSpec(
    name='load_control_models',
    description='Load ControlNet and IP-Adapter models, apply dimension/step overrides',
    required_state=('ctx', 'progress'),
    produced_state=('ctx', 'progress'),
    on_failure='raise',
)


def stage_load_control_models(runtime, async_task, state):
    import modules.config
    progressbar(async_task, state.progress, 'Loading control models ...')
    runtime.pipeline.refresh_controlnets(
        [state.ctx.controlnet_canny_path, state.ctx.controlnet_cpds_path])
    runtime.ip_adapter.load_ip_adapter(
        state.ctx.clip_vision_path, state.ctx.ip_negative_path, state.ctx.ip_adapter_path)
    runtime.ip_adapter.load_ip_adapter(
        state.ctx.clip_vision_path, state.ctx.ip_negative_path, state.ctx.ip_adapter_face_path)

    state.ctx.steps, state.ctx.switch, state.ctx.width, state.ctx.height = (
        apply_overrides_to_context(state.ctx))
    print(f'[Parameters] Sampler = {state.ctx.sampler_name} - {state.ctx.scheduler_name}')
    print(f'[Parameters] Steps = {state.ctx.steps} - {state.ctx.switch}')
    return StageResult(state=state)


spec_expand_prompts = StageSpec(
    name='expand_prompts',
    description='Refresh pipeline models, set CLIP skip, expand prompt into per-task contexts',
    required_state=('ctx', 'progress'),
    produced_state=('loras', 'tasks', 'progress'),
    on_failure='raise',
)


def stage_expand_prompts(runtime, async_task, state):
    progressbar(async_task, state.progress, 'Initializing ...')
    state.loras = state.ctx.loras
    state.tasks = []
    if not state.ctx.skip_prompt_processing:
        runtime.pipeline.refresh_everything(
            refiner_model_name=state.ctx.refiner_model_name,
            base_model_name=state.ctx.base_model_name,
            loras=state.loras,
            base_model_additional_loras=state.ctx.base_model_additional_loras,
            use_synthetic_refiner=state.ctx.use_synthetic_refiner,
            vae_name=state.ctx.vae_name,
        )
        runtime.pipeline.set_clip_skip(state.ctx.clip_skip)
        state.tasks, state.loras, state.progress = expand_tasks_from_prompt(
            state.ctx, state.ctx.base_model_additional_loras, state.progress, advance_progress=True)
    return StageResult(state=state)


spec_apply_goals_prefix = StageSpec(
    name='apply_goals_prefix',
    description='Bump progress and show "Image processing" message when goals exist',
    required_state=('ctx', 'progress'),
    produced_state=('progress',),
    on_failure='skip',
)


def stage_apply_goals_prefix(runtime, async_task, state):
    if len(state.ctx.goals) > 0:
        state.progress += 1
        progressbar(async_task, state.progress, 'Image processing ...')
    return StageResult(state=state)


spec_goal_vary = StageSpec(
    name='goal_vary',
    description='Apply vary (subtle/strong) context transformation',
    required_state=('ctx', 'progress'),
    produced_state=('ctx', 'progress'),
    on_failure='raise',
)


def stage_goal_vary(runtime, async_task, state):
    if 'vary' in state.ctx.goals:
        state.ctx, state.progress = apply_vary_to_context(
            state.ctx, state.ctx.switch, state.progress)
    return StageResult(state=state)


spec_goal_upscale = StageSpec(
    name='goal_upscale',
    description='Apply upscale context transformation; may short-circuit on Fast Upscale path',
    required_state=('ctx', 'progress'),
    produced_state=('ctx', 'progress'),
    may_direct_return=True,
    on_failure='raise',
)


def stage_goal_upscale(runtime, async_task, state):
    import modules.config
    if 'upscale' not in state.ctx.goals:
        return StageResult(state=state)

    state.ctx, direct_return, state.progress = apply_upscale_to_context(
        state.ctx, state.ctx.switch, state.progress, advance_progress=True)
    if not direct_return:
        return StageResult(state=state)

    d = [('Upscale (Fast)', 'upscale_fast', '2x')]
    if modules.config.default_black_out_nsfw or state.ctx.black_out_nsfw:
        progressbar(async_task, 100, 'Checking for NSFW content ...')
        state.ctx.uov_input_image = runtime.default_censor(state.ctx.uov_input_image)
    progressbar(async_task, 100, 'Saving image to system ...')
    uov_input_image_path = runtime.log(
        state.ctx.uov_input_image, d, output_format=state.ctx.output_format)
    yield_result(
        async_task, uov_input_image_path, 100, state.ctx.black_out_nsfw, False,
        do_not_show_finished_images=True)
    return StageResult(state=state, direct_return=True)


class _EarlyReturnMarker(BaseException):
    pass


spec_goal_inpaint = StageSpec(
    name='goal_inpaint',
    description='Apply inpaint context setup; may short-circuit on debug preprocessor mode',
    required_state=('ctx', 'inpaint_image', 'inpaint_mask', 'progress'),
    produced_state=('ctx', 'progress'),
    may_direct_return=True,
    on_failure='raise',
)


def stage_goal_inpaint(runtime, async_task, state):
    if 'inpaint' not in state.ctx.goals:
        return StageResult(state=state)

    try:
        state.ctx, state.progress = apply_inpaint_to_context(
            state.ctx, state.inpaint_image, state.inpaint_mask, state.ctx.switch, state.progress,
            advance_progress=True)
        if state.ctx.debugging_inpaint_preprocessor:
            yield_result(
                async_task, runtime.inpaint_worker.current_task.visualize_mask_processing(),
                100, state.ctx.black_out_nsfw, do_not_show_finished_images=True)
            return StageResult(state=state, direct_return=True)
    except _EarlyReturnMarker:
        return StageResult(state=state, direct_return=True)
    return StageResult(state=state)


spec_goal_controlnet = StageSpec(
    name='goal_controlnet',
    description='Apply ControlNet preprocessors; may short-circuit on CN debug mode',
    required_state=('ctx', 'progress'),
    produced_state=(),
    may_direct_return=True,
    on_failure='raise',
)


def stage_goal_controlnet(runtime, async_task, state):
    if 'cn' in state.ctx.goals:
        apply_control_nets_from_context(state.ctx, state.progress)
        if state.ctx.debugging_cn_preprocessor:
            return StageResult(state=state, direct_return=True)
    return StageResult(state=state)


spec_apply_freeu_and_overrides = StageSpec(
    name='apply_freeu_and_overrides',
    description='Apply FreeU settings and recompute step overrides after goal transformations',
    required_state=('ctx',),
    produced_state=('ctx',),
    on_failure='raise',
)


def stage_apply_freeu_and_overrides(runtime, async_task, state):
    if state.ctx.freeu_enabled:
        apply_freeu_from_context(state.ctx)
    state.ctx.steps, _, _, _ = apply_overrides_to_context(state.ctx)
    return StageResult(state=state)


spec_prepare_enhance_input = StageSpec(
    name='prepare_enhance_input',
    description='If enhance goal is active, seed images list with enhance input and adjust ctx dimensions',
    required_state=('ctx', 'progress'),
    produced_state=('images', 'progress'),
    on_failure='raise',
)


def stage_prepare_enhance_input(runtime, async_task, state):
    if 'enhance' in state.ctx.goals:
        state.ctx.image_number = 1
        state.images += [state.ctx.enhance_input_image]
        state.ctx.height, state.ctx.width, _ = state.ctx.enhance_input_image.shape
        state.ctx.steps = 0
        yield_result(
            async_task, state.ctx.enhance_input_image, state.progress,
            state.ctx.black_out_nsfw, False, state.ctx.disable_intermediate_results)
    return StageResult(state=state)


spec_compute_total_steps = StageSpec(
    name='compute_total_steps',
    description='Compute total diffusion steps and print denoising strength / latent shape',
    required_state=('ctx',),
    produced_state=('all_steps',),
    on_failure='raise',
)


def stage_compute_total_steps(runtime, async_task, state):
    state.all_steps = compute_total_steps(state.ctx)
    print(f'[Parameters] Denoising Strength = {state.ctx.denoising_strength}')
    if isinstance(state.ctx.initial_latent, dict) and 'samples' in state.ctx.initial_latent:
        log_shape = state.ctx.initial_latent['samples'].shape
    else:
        log_shape = f'Image Space {(state.ctx.width, state.ctx.height)}'
    print(f'[Parameters] Initial Latent shape: {log_shape}')
    return StageResult(state=state)


spec_finalize_preparation = StageSpec(
    name='finalize_preparation',
    description='Patch samplers, initialize ProgressReporter, mark processing_started',
    required_state=('ctx', 'progress', 'all_steps'),
    produced_state=(
        'final_scheduler_name', 'processing_started', 'processing_start_time',
        'preparation_steps', 'callback', 'progress_reporter',
    ),
    on_failure='raise',
)


def stage_finalize_preparation(runtime, async_task, state):
    import time
    state.final_scheduler_name = patch_samplers_from_context(state.ctx)
    print(f'Using {state.final_scheduler_name} scheduler.')
    async_task.yields.append(
        ['preview', (state.progress, 'Moving model to GPU ...', None)])
    state.processing_start_time = time.perf_counter()
    state.preparation_steps = state.progress
    total_count = state.ctx.image_number
    state.progress_reporter = ProgressReporter(
        async_task, state.all_steps, state.preparation_steps, total_count)
    state.callback = state.progress_reporter.get_diffusion_callback(state.progress)
    state.processing_started = True
    return StageResult(state=state)


spec_run_diffusion = StageSpec(
    name='run_diffusion',
    description='Run diffusion loop over tasks, accumulate generated images into state.images',
    required_state=(
        'ctx', 'progress', 'loras', 'tasks', 'final_scheduler_name',
        'all_steps', 'preparation_steps', 'callback', 'progress_reporter',
    ),
    produced_state=('images', 'progress'),
    on_failure='raise',
)


def stage_run_diffusion(runtime, async_task, state):
    import time
    ctx = state.ctx
    show_intermediate_results = len(state.tasks) > 1 or ctx.should_enhance
    persist_image = not ctx.should_enhance or not ctx.save_final_enhanced_image_only
    total_count = ctx.image_number

    for current_task_id, task_ctx in enumerate(state.tasks):
        progressbar(
            async_task, state.progress,
            f'Preparing task {current_task_id + 1}/{ctx.image_number} ...')
        execution_start_time = time.perf_counter()
        try:
            imgs, img_paths, state.progress = execute_diffusion_task(
                ctx, task_ctx, state.loras, state.final_scheduler_name, state.callback,
                current_task_id, total_count, state.progress, state.preparation_steps,
                state.all_steps, show_intermediate_results, persist_image,
                state.progress_reporter, runtime.pid)
            state.progress = int(
                state.preparation_steps
                + (100 - state.preparation_steps) / float(state.all_steps)
                * ctx.steps * (current_task_id + 1))
            state.images += imgs
        except runtime.ldm_model_management.InterruptProcessingException:
            if async_task.last_stop == 'skip':
                print('User skipped')
                async_task.last_stop = False
                continue
            else:
                print('User stopped')
                break
        del task_ctx.c, task_ctx.uc
        execution_time = time.perf_counter() - execution_start_time
        print(f'Generating and saving time: {execution_time:.2f} seconds')
    return StageResult(state=state)


spec_run_enhance = StageSpec(
    name='run_enhance',
    description='Run full enhance pipeline if should_enhance is set',
    required_state=(
        'ctx', 'all_steps', 'callback', 'final_scheduler_name',
        'images', 'progress', 'preparation_steps',
    ),
    produced_state=(),
    may_direct_return=True,
    on_failure='raise',
)


def stage_run_enhance(runtime, async_task, state):
    if not state.ctx.should_enhance:
        print(f'[Enhance] Skipping, preconditions aren\'t met')
        return StageResult(state=state, direct_return=True)

    persist_image = (
        not state.ctx.should_enhance or not state.ctx.save_final_enhanced_image_only)
    run_enhance_pipeline(
        runtime, async_task, state.ctx, state.all_steps, state.callback,
        state.final_scheduler_name, state.images, state.progress,
        state.preparation_steps, persist_image)
    return StageResult(state=state)


PIPELINE_STAGES: Tuple[Tuple[StageSpec, StageFn], ...] = (
    (spec_build_context, stage_build_context),
    (spec_apply_patches, stage_apply_patches),
    (spec_process_image_input, stage_process_image_input),
    (spec_load_control_models, stage_load_control_models),
    (spec_expand_prompts, stage_expand_prompts),
    (spec_apply_goals_prefix, stage_apply_goals_prefix),
    (spec_goal_vary, stage_goal_vary),
    (spec_goal_upscale, stage_goal_upscale),
    (spec_goal_inpaint, stage_goal_inpaint),
    (spec_goal_controlnet, stage_goal_controlnet),
    (spec_apply_freeu_and_overrides, stage_apply_freeu_and_overrides),
    (spec_prepare_enhance_input, stage_prepare_enhance_input),
    (spec_compute_total_steps, stage_compute_total_steps),
    (spec_finalize_preparation, stage_finalize_preparation),
    (spec_run_diffusion, stage_run_diffusion),
    (spec_run_enhance, stage_run_enhance),
)


# Backwards-compatible alias — a list of just the functions, without specs
DEFAULT_PIPELINE: Tuple[StageFn, ...] = tuple(fn for _, fn in PIPELINE_STAGES)


class PipelineValidationError(Exception):
    """Raised when pipeline stage validation fails."""
    pass


def _format_missing_fields(fields: List[str]) -> str:
    return ', '.join(fields) if fields else '(none)'


def execute_pipeline(
    runtime: WorkerRuntime,
    async_task,
    pipeline: Tuple[Tuple[StageSpec, StageFn], ...] = PIPELINE_STAGES,
    audit_log: Optional[List[Dict[str, Any]]] = None,
    strict: bool = True,
) -> PipelineState:
    """
    Execute a pipeline of stages with input/output validation and auditing.

    Args:
        runtime: The WorkerRuntime containing all shared runtime state
        async_task: The AsyncTask being processed
        pipeline: Tuple of (StageSpec, stage_fn) pairs
        audit_log: Optional list to append audit events to
        strict: If True, raise on validation failures; if False, log and continue

    Returns:
        Final PipelineState after execution completes or short-circuits

    Raises:
        PipelineValidationError: If a stage fails input/output validation and strict=True
    """
    state = PipelineState()

    for i, (spec, stage_fn) in enumerate(pipeline):
        audit = {
            'index': i,
            'stage': spec.name,
            'description': spec.description,
            'required': list(spec.required_state),
            'produced': list(spec.produced_state),
            'may_direct_return': spec.may_direct_return,
            'on_failure': spec.on_failure,
            'status': 'pending',
        }

        ok, problems = spec.validate_input(state)
        if not ok:
            msg = (f'Stage "{spec.name}" failed input validation: '
                   f'missing {_format_missing_fields(problems)}')
            audit['status'] = 'input_validation_failed'
            audit['errors'] = problems
            if audit_log is not None:
                audit_log.append(audit)
            if spec.on_failure == 'raise' and strict:
                raise PipelineValidationError(msg)
            elif spec.on_failure == 'skip':
                print(f'[Pipeline] Skipping stage {spec.name}: {msg}')
                continue
            else:  # abort
                print(f'[Pipeline] Aborting at stage {spec.name}: {msg}')
                return state

        audit['input_valid'] = True
        print(f'[Pipeline] Stage {i:>2}/{len(pipeline):<2} {spec.name:<30} '
              f'| required: {", ".join(spec.required_state) or "(none)"}')

        try:
            result = stage_fn(runtime, async_task, state)
        except Exception as e:
            audit['status'] = 'exception'
            audit['exception'] = f'{type(e).__name__}: {e}'
            if audit_log is not None:
                audit_log.append(audit)
            if spec.on_failure == 'raise' and strict:
                raise
            elif spec.on_failure == 'skip':
                print(f'[Pipeline] Stage {spec.name} raised {type(e).__name__}: {e}; skipping')
                continue
            else:
                print(f'[Pipeline] Stage {spec.name} raised {type(e).__name__}: {e}; aborting')
                return state

        state = result.state

        ok_out, problems_out = spec.validate_output(state)
        if not ok_out:
            msg = (f'Stage "{spec.name}" failed output validation: '
                   f'unknown fields {_format_missing_fields(problems_out)}')
            audit['status'] = 'output_validation_failed'
            audit['output_errors'] = problems_out
            if audit_log is not None:
                audit_log.append(audit)
            if strict:
                raise PipelineValidationError(msg)
            print(f'[Pipeline] Warning: {msg}')

        audit['status'] = 'completed' if not result.direct_return else 'completed_direct_return'
        audit['direct_return'] = result.direct_return
        if audit_log is not None:
            audit_log.append(audit)

        if result.direct_return:
            if not spec.may_direct_return:
                print(f'[Pipeline] Stage {spec.name} requested direct return (allowed)')
            else:
                msg = f'Stage "{spec.name}" returned direct_return but may_direct_return=False'
                if strict:
                    raise PipelineValidationError(msg)
                print(f'[Pipeline] Warning: {msg}')
            break

    print(f'[Pipeline] Completed {i + 1} stages executed')
    return state


def validate_pipeline(
    pipeline: Tuple[Tuple[StageSpec, StageFn], ...] = PIPELINE_STAGES,
) -> Tuple[bool, List[str]]:
    """
    Validate a pipeline's stage specs for consistency.

    Checks:
    1. All required_state fields exist
    2. produced_state fields exist
    3. No gaps in the state flow (earlier stages produce fields that later stages require)
    4. No duplicate stage names

    Returns:
        (is_valid, list of problems)
    """
    problems: List[str] = []
    produced_so_far: Set[str] = set()
    seen_names: Set[str] = set()

    for i, (spec, fn) in enumerate(pipeline):
        if spec.name in seen_names:
            problems.append(f'Duplicate stage name: {spec.name} at index {i}')
        seen_names.add(spec.name)

        for fname in spec.required_state:
            if fname not in STATE_FIELDS:
                problems.append(
                    f'Stage {spec.name}: required_state "{fname}" not a valid PipelineState field')
            elif fname not in produced_so_far and fname != 'ctx' and spec.required_state != ():
                # ctx is always allowed to be None initially; first stage produces it
                if i > 0 and fname not in produced_so_far:
                    problems.append(
                        f'Stage {spec.name}: required "{fname}" not produced by any prior stage')

        for fname in spec.produced_state:
            if fname not in STATE_FIELDS:
                problems.append(
                    f'Stage {spec.name}: produced_state "{fname}" not a valid PipelineState field')
            produced_so_far.add(fname)

    return len(problems) == 0, problems
