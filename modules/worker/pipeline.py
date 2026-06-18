from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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


@dataclass
class StageResult:
    state: PipelineState
    direct_return: bool = False


StageFn = Callable[[WorkerRuntime, Any, PipelineState], StageResult]


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


def stage_process_image_input(runtime, async_task, state):
    if state.ctx.input_image_checkbox:
        state.inpaint_image, state.inpaint_mask, state.progress = process_image_input(
            state.ctx, state.progress)
    return StageResult(state=state)


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


def stage_apply_goals_prefix(runtime, async_task, state):
    if len(state.ctx.goals) > 0:
        state.progress += 1
        progressbar(async_task, state.progress, 'Image processing ...')
    return StageResult(state=state)


def stage_goal_vary(runtime, async_task, state):
    if 'vary' in state.ctx.goals:
        state.ctx, state.progress = apply_vary_to_context(
            state.ctx, state.ctx.switch, state.progress)
    return StageResult(state=state)


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


def stage_goal_controlnet(runtime, async_task, state):
    if 'cn' in state.ctx.goals:
        apply_control_nets_from_context(state.ctx, state.progress)
        if state.ctx.debugging_cn_preprocessor:
            return StageResult(state=state, direct_return=True)
    return StageResult(state=state)


def stage_apply_freeu_and_overrides(runtime, async_task, state):
    if state.ctx.freeu_enabled:
        apply_freeu_from_context(state.ctx)
    state.ctx.steps, _, _, _ = apply_overrides_to_context(state.ctx)
    return StageResult(state=state)


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


def stage_compute_total_steps(runtime, async_task, state):
    state.all_steps = compute_total_steps(state.ctx)
    print(f'[Parameters] Denoising Strength = {state.ctx.denoising_strength}')
    if isinstance(state.ctx.initial_latent, dict) and 'samples' in state.ctx.initial_latent:
        log_shape = state.ctx.initial_latent['samples'].shape
    else:
        log_shape = f'Image Space {(state.ctx.width, state.ctx.height)}'
    print(f'[Parameters] Initial Latent shape: {log_shape}')
    return StageResult(state=state)


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


DEFAULT_PIPELINE: Tuple[StageFn, ...] = (
    stage_build_context,
    stage_apply_patches,
    stage_process_image_input,
    stage_load_control_models,
    stage_expand_prompts,
    stage_apply_goals_prefix,
    stage_goal_vary,
    stage_goal_upscale,
    stage_goal_inpaint,
    stage_goal_controlnet,
    stage_apply_freeu_and_overrides,
    stage_prepare_enhance_input,
    stage_compute_total_steps,
    stage_finalize_preparation,
    stage_run_diffusion,
    stage_run_enhance,
)


def execute_pipeline(
    runtime: WorkerRuntime,
    async_task,
    pipeline: Tuple[StageFn, ...] = DEFAULT_PIPELINE,
) -> PipelineState:
    state = PipelineState()
    for stage_fn in pipeline:
        result = stage_fn(runtime, async_task, state)
        state = result.state
        if result.direct_return:
            break
    return state
