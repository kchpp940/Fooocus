import threading
import time
import os
import traceback

from modules.patch import patch_all
import modules.config

from modules.worker.context import GenerationContext, SingleTaskContext, WorkerRuntime
from modules.worker.progress import ProgressReporter, progressbar, yield_result
from modules.worker.result_saver import save_images_with_metadata, build_image_wall_from_results
from modules.worker.tasks import (
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
from modules.worker.stages import (
    apply_vary_to_context,
    apply_upscale_to_context,
    apply_inpaint_to_context,
    apply_control_nets_from_context,
    generate_enhance_mask,
    prepare_enhance_prompt,
)
from modules.worker.enhance import run_enhance_pipeline

patch_all()


class AsyncTask:
    def __init__(self, args):
        from modules.flags import Performance, MetadataScheme, ip_list, disabled
        from modules.util import get_enabled_loras
        from modules.config import default_max_lora_number
        import args_manager

        self.args = args.copy()
        self.yields = []
        self.results = []
        self.last_stop = False
        self.processing = False

        self.performance_loras = []

        if len(args) == 0:
            return

        args.reverse()
        self.generate_image_grid = args.pop()
        self.prompt = args.pop()
        self.negative_prompt = args.pop()
        self.style_selections = args.pop()

        self.performance_selection = Performance(args.pop())
        self.steps = self.performance_selection.steps()
        self.original_steps = self.steps

        self.aspect_ratios_selection = args.pop()
        self.image_number = args.pop()
        self.output_format = args.pop()
        self.seed = int(args.pop())
        self.read_wildcards_in_order = args.pop()
        self.sharpness = args.pop()
        self.cfg_scale = args.pop()
        self.base_model_name = args.pop()
        self.refiner_model_name = args.pop()
        self.refiner_switch = args.pop()
        self.loras = get_enabled_loras([(bool(args.pop()), str(args.pop()), float(args.pop())) for _ in
                                        range(default_max_lora_number)])
        self.input_image_checkbox = args.pop()
        self.current_tab = args.pop()
        self.uov_method = args.pop()
        self.uov_input_image = args.pop()
        self.outpaint_selections = args.pop()
        self.inpaint_input_image = args.pop()
        self.inpaint_additional_prompt = args.pop()
        self.inpaint_mask_image_upload = args.pop()

        self.disable_preview = args.pop()
        self.disable_intermediate_results = args.pop()
        self.disable_seed_increment = args.pop()
        self.black_out_nsfw = args.pop()
        self.adm_scaler_positive = args.pop()
        self.adm_scaler_negative = args.pop()
        self.adm_scaler_end = args.pop()
        self.adaptive_cfg = args.pop()
        self.clip_skip = args.pop()
        self.sampler_name = args.pop()
        self.scheduler_name = args.pop()
        self.vae_name = args.pop()
        self.overwrite_step = args.pop()
        self.overwrite_switch = args.pop()
        self.overwrite_width = args.pop()
        self.overwrite_height = args.pop()
        self.overwrite_vary_strength = args.pop()
        self.overwrite_upscale_strength = args.pop()
        self.mixing_image_prompt_and_vary_upscale = args.pop()
        self.mixing_image_prompt_and_inpaint = args.pop()
        self.debugging_cn_preprocessor = args.pop()
        self.skipping_cn_preprocessor = args.pop()
        self.canny_low_threshold = args.pop()
        self.canny_high_threshold = args.pop()
        self.refiner_swap_method = args.pop()
        self.controlnet_softness = args.pop()
        self.freeu_enabled = args.pop()
        self.freeu_b1 = args.pop()
        self.freeu_b2 = args.pop()
        self.freeu_s1 = args.pop()
        self.freeu_s2 = args.pop()
        self.debugging_inpaint_preprocessor = args.pop()
        self.inpaint_disable_initial_latent = args.pop()
        self.inpaint_engine = args.pop()
        self.inpaint_strength = args.pop()
        self.inpaint_respective_field = args.pop()
        self.inpaint_advanced_masking_checkbox = args.pop()
        self.invert_mask_checkbox = args.pop()
        self.inpaint_erode_or_dilate = args.pop()
        self.save_final_enhanced_image_only = args.pop() if not args_manager.args.disable_image_log else False
        self.save_metadata_to_images = args.pop() if not args_manager.args.disable_metadata else False
        self.metadata_scheme = MetadataScheme(
            args.pop()) if not args_manager.args.disable_metadata else MetadataScheme.FOOOCUS

        self.cn_tasks = {x: [] for x in ip_list}
        for _ in range(modules.config.default_controlnet_image_count):
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                self.cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        self.debugging_dino = args.pop()
        self.dino_erode_or_dilate = args.pop()
        self.debugging_enhance_masks_checkbox = args.pop()

        self.enhance_input_image = args.pop()
        self.enhance_checkbox = args.pop()
        self.enhance_uov_method = args.pop()
        self.enhance_uov_processing_order = args.pop()
        self.enhance_uov_prompt_type = args.pop()
        self.enhance_ctrls = []
        for _ in range(modules.config.default_enhance_tabs):
            enhance_enabled = args.pop()
            enhance_mask_dino_prompt_text = args.pop()
            enhance_prompt = args.pop()
            enhance_negative_prompt = args.pop()
            enhance_mask_model = args.pop()
            enhance_mask_cloth_category = args.pop()
            enhance_mask_sam_model = args.pop()
            enhance_mask_text_threshold = args.pop()
            enhance_mask_box_threshold = args.pop()
            enhance_mask_sam_max_detections = args.pop()
            enhance_inpaint_disable_initial_latent = args.pop()
            enhance_inpaint_engine = args.pop()
            enhance_inpaint_strength = args.pop()
            enhance_inpaint_respective_field = args.pop()
            enhance_inpaint_erode_or_dilate = args.pop()
            enhance_mask_invert = args.pop()
            if enhance_enabled:
                self.enhance_ctrls.append([
                    enhance_mask_dino_prompt_text,
                    enhance_prompt,
                    enhance_negative_prompt,
                    enhance_mask_model,
                    enhance_mask_cloth_category,
                    enhance_mask_sam_model,
                    enhance_mask_text_threshold,
                    enhance_mask_box_threshold,
                    enhance_mask_sam_max_detections,
                    enhance_inpaint_disable_initial_latent,
                    enhance_inpaint_engine,
                    enhance_inpaint_strength,
                    enhance_inpaint_respective_field,
                    enhance_inpaint_erode_or_dilate,
                    enhance_mask_invert
                ])
        self.should_enhance = self.enhance_checkbox and (self.enhance_uov_method != disabled.casefold() or len(self.enhance_ctrls) > 0)
        self.images_to_enhance_count = 0
        self.enhance_stats = {}


async_tasks = []


class EarlyReturnException(BaseException):
    pass


def stop_processing(async_task, processing_start_time, time_module):
    async_task.processing = False
    processing_time = time_module.perf_counter() - processing_start_time
    print(f'Processing time (total): {processing_time:.2f} seconds')


def build_worker_runtime() -> WorkerRuntime:
    import torch
    import modules.default_pipeline as pipeline
    import modules.inpaint_worker as inpaint_worker
    import modules.flags as flags
    import ldm_patched.modules.model_management as ldm_model_management
    import extras.ip_adapter as ip_adapter
    from extras.censor import default_censor
    from modules.sdxl_styles import fooocus_expansion
    from modules.private_logger import log

    pid = os.getpid()
    print(f'Started worker with PID {pid}')

    return WorkerRuntime(
        pid=pid,
        pipeline=pipeline,
        inpaint_worker=inpaint_worker,
        flags=flags,
        ldm_model_management=ldm_model_management,
        ip_adapter=ip_adapter,
        default_censor=default_censor,
        fooocus_expansion=fooocus_expansion,
        log=log,
        time_module=time,
    )


def print_welcome():
    try:
        import shared
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)


import torch


@torch.no_grad()
@torch.inference_mode()
def execute_handler(runtime: WorkerRuntime, async_task: AsyncTask):
    preparation_start_time = time.perf_counter()
    async_task.processing = True

    runtime.inpaint_worker.current_task = None

    ctx = build_generation_context(async_task)

    ctx.width, ctx.height = parse_dimensions_from_aspect_ratio(ctx.aspect_ratios_selection)

    if runtime.fooocus_expansion in ctx.style_selections:
        ctx.use_expansion = True
        ctx.style_selections.remove(runtime.fooocus_expansion)
    else:
        ctx.use_expansion = False

    ctx.use_style = len(ctx.style_selections) > 0

    current_progress = normalize_performance_settings(ctx, runtime.pid, current_progress=0)

    print(f'[Parameters] Adaptive CFG = {ctx.adaptive_cfg}')
    print(f'[Parameters] CLIP Skip = {ctx.clip_skip}')
    print(f'[Parameters] Sharpness = {ctx.sharpness}')
    print(f'[Parameters] ControlNet Softness = {ctx.controlnet_softness}')
    print(f'[Parameters] ADM Scale = '
          f'{ctx.adm_scaler_positive} : '
          f'{ctx.adm_scaler_negative} : '
          f'{ctx.adm_scaler_end}')
    print(f'[Parameters] Seed = {ctx.seed}')

    apply_patch_settings_to_context(ctx, runtime.pid)

    print(f'[Parameters] CFG = {ctx.cfg_scale}')

    ctx.denoising_strength = 1.0
    ctx.tiled = False
    ctx.skip_prompt_processing = False
    ctx.inpaint_parameterized = ctx.inpaint_engine != 'None'
    ctx.use_synthetic_refiner = False

    inpaint_image = None
    inpaint_mask = None

    current_progress = 1

    if ctx.input_image_checkbox:
        inpaint_image, inpaint_mask, current_progress = process_image_input(ctx, current_progress)

    progressbar(async_task, current_progress, 'Loading control models ...')
    runtime.pipeline.refresh_controlnets([ctx.controlnet_canny_path, ctx.controlnet_cpds_path])
    runtime.ip_adapter.load_ip_adapter(ctx.clip_vision_path, ctx.ip_negative_path, ctx.ip_adapter_path)
    runtime.ip_adapter.load_ip_adapter(ctx.clip_vision_path, ctx.ip_negative_path, ctx.ip_adapter_face_path)

    ctx.steps, ctx.switch, ctx.width, ctx.height = apply_overrides_to_context(ctx)

    print(f'[Parameters] Sampler = {ctx.sampler_name} - {ctx.scheduler_name}')
    print(f'[Parameters] Steps = {ctx.steps} - {ctx.switch}')

    progressbar(async_task, current_progress, 'Initializing ...')

    loras = ctx.loras
    tasks = []
    if not ctx.skip_prompt_processing:
        runtime.pipeline.refresh_everything(
            refiner_model_name=ctx.refiner_model_name,
            base_model_name=ctx.base_model_name,
            loras=loras, base_model_additional_loras=ctx.base_model_additional_loras,
            use_synthetic_refiner=ctx.use_synthetic_refiner, vae_name=ctx.vae_name,
        )
        runtime.pipeline.set_clip_skip(ctx.clip_skip)
        tasks, loras, current_progress = expand_tasks_from_prompt(
            ctx, ctx.base_model_additional_loras, current_progress, advance_progress=True)

    if len(ctx.goals) > 0:
        current_progress += 1
        progressbar(async_task, current_progress, 'Image processing ...')

    if 'vary' in ctx.goals:
        ctx, current_progress = apply_vary_to_context(ctx, ctx.switch, current_progress)

    if 'upscale' in ctx.goals:
        ctx, direct_return, current_progress = apply_upscale_to_context(
            ctx, ctx.switch, current_progress, advance_progress=True)
        if direct_return:
            d = [('Upscale (Fast)', 'upscale_fast', '2x')]
            if modules.config.default_black_out_nsfw or ctx.black_out_nsfw:
                progressbar(async_task, 100, 'Checking for NSFW content ...')
                ctx.uov_input_image = runtime.default_censor(ctx.uov_input_image)
            progressbar(async_task, 100, 'Saving image to system ...')
            uov_input_image_path = runtime.log(ctx.uov_input_image, d, output_format=ctx.output_format)
            yield_result(async_task, uov_input_image_path, 100, ctx.black_out_nsfw, False,
                         do_not_show_finished_images=True)
            return

    if 'inpaint' in ctx.goals:
        try:
            ctx, current_progress = apply_inpaint_to_context(
                ctx, inpaint_image, inpaint_mask, ctx.switch, current_progress,
                advance_progress=True)
            if ctx.debugging_inpaint_preprocessor:
                yield_result(async_task, runtime.inpaint_worker.current_task.visualize_mask_processing(), 100,
                             ctx.black_out_nsfw, do_not_show_finished_images=True)
                raise EarlyReturnException
        except EarlyReturnException:
            return

    if 'cn' in ctx.goals:
        apply_control_nets_from_context(ctx, current_progress)
        if ctx.debugging_cn_preprocessor:
            return

    if ctx.freeu_enabled:
        apply_freeu_from_context(ctx)

    ctx.steps, _, _, _ = apply_overrides_to_context(ctx)

    images_to_enhance = []
    if 'enhance' in ctx.goals:
        ctx.image_number = 1
        images_to_enhance += [ctx.enhance_input_image]
        ctx.height, ctx.width, _ = ctx.enhance_input_image.shape
        ctx.steps = 0
        yield_result(async_task, ctx.enhance_input_image, current_progress, ctx.black_out_nsfw, False,
                     ctx.disable_intermediate_results)

    all_steps = compute_total_steps(ctx)

    print(f'[Parameters] Denoising Strength = {ctx.denoising_strength}')

    if isinstance(ctx.initial_latent, dict) and 'samples' in ctx.initial_latent:
        log_shape = ctx.initial_latent['samples'].shape
    else:
        log_shape = f'Image Space {(ctx.width, ctx.height)}'

    print(f'[Parameters] Initial Latent shape: {log_shape}')

    preparation_time = time.perf_counter() - preparation_start_time
    print(f'Preparation time: {preparation_time:.2f} seconds')

    final_scheduler_name = patch_samplers_from_context(ctx)
    print(f'Using {final_scheduler_name} scheduler.')

    async_task.yields.append(['preview', (current_progress, 'Moving model to GPU ...', None)])

    processing_start_time = time.perf_counter()

    preparation_steps = current_progress
    total_count = ctx.image_number

    progress_reporter = ProgressReporter(async_task, all_steps, preparation_steps, total_count)

    show_intermediate_results = len(tasks) > 1 or ctx.should_enhance
    persist_image = not ctx.should_enhance or not ctx.save_final_enhanced_image_only

    callback = progress_reporter.get_diffusion_callback(current_progress)

    for current_task_id, task_ctx in enumerate(tasks):
        progressbar(
            async_task, current_progress,
            f'Preparing task {current_task_id + 1}/{ctx.image_number} ...',
        )
        execution_start_time = time.perf_counter()

        try:
            imgs, img_paths, current_progress = execute_diffusion_task(
                ctx, task_ctx, loras, final_scheduler_name, callback,
                current_task_id, total_count, current_progress, preparation_steps,
                all_steps, show_intermediate_results, persist_image,
                progress_reporter, runtime.pid)

            current_progress = int(
                preparation_steps + (100 - preparation_steps) / float(all_steps) * ctx.steps * (current_task_id + 1)
            )
            images_to_enhance += imgs

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

    if not ctx.should_enhance:
        print(f'[Enhance] Skipping, preconditions aren\'t met')
        stop_processing(async_task, processing_start_time, time)
        return

    run_enhance_pipeline(
        runtime, async_task, ctx, all_steps, callback, final_scheduler_name,
        images_to_enhance, current_progress, preparation_steps, persist_image,
    )

    stop_processing(async_task, processing_start_time, time)


def worker():
    global async_tasks

    import modules.patch

    runtime = build_worker_runtime()
    print_welcome()

    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)

            try:
                execute_handler(runtime, task)
                if task.generate_image_grid:
                    wall = build_image_wall_from_results(task.results)
                    if wall is not None:
                        task.results = task.results + [wall]
                task.yields.append(['finish', task.results])
                runtime.pipeline.prepare_text_encoder(async_call=True)
            except:
                traceback.print_exc()
                task.yields.append(['finish', task.results])
            finally:
                if runtime.pid in modules.patch.patch_settings:
                    del modules.patch.patch_settings[runtime.pid]


threading.Thread(target=worker, daemon=True).start()
