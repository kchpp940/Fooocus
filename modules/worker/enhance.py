from typing import Tuple, Any, Optional, List

from .context import GenerationContext, WorkerRuntime
from .progress import ProgressReporter, progressbar, yield_result
from .stages import (
    prepare_enhance_prompt, apply_inpaint_to_context, apply_upscale_to_context,
    generate_enhance_mask,
)
from .tasks import (
    expand_tasks_from_prompt,
    apply_overrides_to_context,
    execute_diffusion_task,
    patch_samplers_from_context,
    apply_freeu_from_context,
    compute_total_steps,
)


def process_enhance(
    runtime: WorkerRuntime,
    async_task,
    ctx: GenerationContext,
    callback,
    current_progress: int,
    current_task_id: int,
    inpaint_disable_initial_latent: bool,
    inpaint_engine: str,
    inpaint_respective_field: float,
    inpaint_strength: float,
    prompt: str,
    negative_prompt: str,
    final_scheduler_name: str,
    img,
    mask,
    preparation_steps: int,
    enhance_steps: int,
    total_count: int,
    show_intermediate_results: bool = True,
    persist_image: bool = True,
):
    import modules.config
    import modules.core as core

    runtime.inpaint_worker.current_task = None

    base_model_additional_loras = []
    inpaint_head_model_path = None
    inpaint_parameterized = inpaint_engine != 'None'

    prompt = prepare_enhance_prompt(prompt, async_task.prompt)
    negative_prompt = prepare_enhance_prompt(negative_prompt, async_task.negative_prompt)

    enhance_ctx = ctx.clone()
    enhance_ctx.denoising_strength = ctx.denoising_strength
    enhance_ctx.initial_latent = None

    if inpaint_parameterized:
        progressbar(async_task, current_progress, 'Downloading inpainter ...')
        inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
            inpaint_engine)
        if inpaint_patch_model_path not in base_model_additional_loras:
            base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]

    progressbar(async_task, current_progress, 'Preparing enhance prompts ...')

    runtime.pipeline.refresh_everything(
        refiner_model_name=enhance_ctx.refiner_model_name,
        base_model_name=enhance_ctx.base_model_name,
        loras=enhance_ctx.loras,
        base_model_additional_loras=base_model_additional_loras,
        use_synthetic_refiner=enhance_ctx.use_synthetic_refiner,
        vae_name=enhance_ctx.vae_name,
    )
    runtime.pipeline.set_clip_skip(enhance_ctx.clip_skip)

    enhance_ctx.prompt = prompt
    enhance_ctx.negative_prompt = negative_prompt
    enhance_ctx.image_number = 1
    enhance_ctx.base_model_additional_loras = base_model_additional_loras

    tasks_enhance, loras, current_progress = expand_tasks_from_prompt(
        enhance_ctx, base_model_additional_loras, current_progress, advance_progress=False)
    task_enhance = tasks_enhance[0]

    if enhance_ctx.freeu_enabled:
        apply_freeu_from_context(enhance_ctx)

    final_scheduler_name_local = patch_samplers_from_context(enhance_ctx)

    if inpaint_parameterized:
        enhance_ctx.inpaint_head_model_path = inpaint_head_model_path
        enhance_ctx.inpaint_strength = inpaint_strength
        enhance_ctx.inpaint_respective_field = inpaint_respective_field
        enhance_ctx.inpaint_disable_initial_latent = inpaint_disable_initial_latent
        enhance_ctx.inpaint_parameterized = inpaint_parameterized
        enhance_ctx, current_progress = apply_inpaint_to_context(
            enhance_ctx, img, mask, enhance_ctx.switch, current_progress,
            skip_apply_outpaint=True, advance_progress=False)

    all_steps = compute_total_steps(enhance_ctx)

    progress_reporter = ProgressReporter(async_task, all_steps, preparation_steps, total_count)

    imgs, img_paths, current_progress = execute_diffusion_task(
        enhance_ctx, task_enhance, loras, final_scheduler_name_local, callback,
        current_task_id, total_count, current_progress, preparation_steps,
        all_steps, show_intermediate_results, persist_image,
        progress_reporter, runtime.pid)

    del task_enhance.c, task_enhance.uc
    return current_progress, imgs[0], prompt, negative_prompt


def enhance_upscale(
    runtime: WorkerRuntime,
    async_task,
    ctx: GenerationContext,
    all_steps: int,
    base_progress: int,
    callback,
    enhance_steps: int,
    current_task_id: int,
    done_steps_inpainting: int,
    done_steps_upscaling: int,
    img,
    preparation_steps: int,
    total_count: int,
    persist_image: bool = True,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
):
    import modules.config
    from modules.util import HWC3

    runtime.inpaint_worker.current_task = None

    if prompt is None:
        prompt = async_task.prompt
    if negative_prompt is None:
        negative_prompt = async_task.negative_prompt

    current_progress = int(
        base_progress + (100 - preparation_steps) / float(all_steps) * (done_steps_upscaling + done_steps_inpainting)
    )

    enhance_ctx = ctx.clone()
    enhance_ctx.uov_input_image = img
    enhance_ctx.uov_method = async_task.enhance_uov_method
    enhance_ctx.steps = enhance_steps

    goals_enhance = []
    uov_input_image = HWC3(img)
    if 'vary' in enhance_ctx.uov_method:
        goals_enhance.append('vary')
    elif 'upscale' in enhance_ctx.uov_method:
        goals_enhance.append('upscale')
        if 'fast' in enhance_ctx.uov_method:
            enhance_ctx.steps = 0
        else:
            enhance_ctx.steps = enhance_ctx.performance_selection.steps_uov()
        modules.config.downloading_upscale_model()

    enhance_ctx.goals = goals_enhance
    enhance_ctx.steps, enhance_ctx.switch, enhance_ctx.width, enhance_ctx.height = apply_overrides_to_context(
        enhance_ctx)

    exception_result = ''
    if len(goals_enhance) > 0:
        try:
            current_progress, img, _, _ = process_enhance(
                runtime,
                async_task, enhance_ctx, callback, current_progress, current_task_id,
                False, 'None', 0.0, 0.0, prompt, negative_prompt,
                patch_samplers_from_context(enhance_ctx),
                img, None, preparation_steps, enhance_ctx.steps, total_count,
                persist_image=persist_image,
            )

        except runtime.ldm_model_management.InterruptProcessingException:
            if async_task.last_stop == 'skip':
                print('User skipped')
                async_task.last_stop = False
                if async_task.enhance_uov_processing_order == runtime.flags.enhancement_uov_before:
                    done_steps_inpainting += len(async_task.enhance_ctrls) * enhance_steps
                exception_result = 'continue'
            else:
                print('User stopped')
                exception_result = 'break'
        finally:
            done_steps_upscaling += enhance_ctx.steps
    return current_task_id, done_steps_inpainting, done_steps_upscaling, img, exception_result


def run_enhance_pipeline(
    runtime: WorkerRuntime,
    async_task,
    ctx: GenerationContext,
    all_steps: int,
    callback,
    final_scheduler_name: str,
    images_to_enhance: List[Any],
    current_progress: int,
    preparation_steps: int,
    persist_image_base: bool,
) -> None:
    progressbar(async_task, current_progress, 'Processing enhance ...')

    active_enhance_tabs = len(ctx.enhance_ctrls)
    should_process_enhance_uov = ctx.enhance_uov_method != runtime.flags.disabled.casefold()
    enhance_uov_before = False
    enhance_uov_after = False
    if should_process_enhance_uov:
        active_enhance_tabs += 1
        enhance_uov_before = ctx.enhance_uov_processing_order == runtime.flags.enhancement_uov_before
        enhance_uov_after = ctx.enhance_uov_processing_order == runtime.flags.enhancement_uov_after
    total_count = len(images_to_enhance) * active_enhance_tabs
    async_task.images_to_enhance_count = len(images_to_enhance)

    base_progress = current_progress
    current_task_id = -1
    done_steps_upscaling = 0
    done_steps_inpainting = 0
    temp_ctx = ctx.clone()
    temp_ctx.steps = ctx.original_steps
    enhance_steps, _, _, _ = apply_overrides_to_context(temp_ctx)
    exception_result = None

    for index, img in enumerate(images_to_enhance):
        async_task.enhance_stats[index] = 0
        enhancement_image_start_time = runtime.time_module.perf_counter()

        last_enhance_prompt = ctx.prompt
        last_enhance_negative_prompt = ctx.negative_prompt

        if enhance_uov_before:
            current_task_id, done_steps_inpainting, done_steps_upscaling, img, exception_result = enhance_upscale(
                runtime,
                async_task, ctx, all_steps, base_progress, callback, enhance_steps,
                current_task_id, done_steps_inpainting, done_steps_upscaling,
                img, preparation_steps, total_count,
                persist_image=not ctx.save_final_enhanced_image_only or active_enhance_tabs == 0,
            )
            async_task.enhance_stats[index] += 1

            if exception_result == 'continue':
                continue
            elif exception_result == 'break':
                break

        for (
            enhance_mask_dino_prompt_text, enhance_prompt, enhance_negative_prompt,
            enhance_mask_model, enhance_mask_cloth_category, enhance_mask_sam_model,
            enhance_mask_text_threshold, enhance_mask_box_threshold,
            enhance_mask_sam_max_detections, enhance_inpaint_disable_initial_latent,
            enhance_inpaint_engine, enhance_inpaint_strength,
            enhance_inpaint_respective_field, enhance_inpaint_erode_or_dilate,
            enhance_mask_invert,
        ) in ctx.enhance_ctrls:
            current_task_id += 1
            current_progress = int(
                base_progress + (100 - preparation_steps) / float(all_steps) * (done_steps_upscaling + done_steps_inpainting)
            )
            progressbar(
                async_task, current_progress,
                f'Preparing enhancement {current_task_id + 1}/{total_count} ...',
            )
            enhancement_task_start_time = runtime.time_module.perf_counter()
            is_last_enhance_for_image = (
                (current_task_id + 1) % active_enhance_tabs == 0 and not enhance_uov_after
            )
            persist_image = not ctx.save_final_enhanced_image_only or is_last_enhance_for_image

            mask, dino_detection_count, sam_detection_count, sam_detection_on_mask_count = generate_enhance_mask(
                img, mask_model=enhance_mask_model, dino_prompt=enhance_mask_dino_prompt_text,
                box_threshold=enhance_mask_box_threshold, text_threshold=enhance_mask_text_threshold,
                sam_model=enhance_mask_sam_model, max_detections=enhance_mask_sam_max_detections,
                dino_erode_or_dilate=ctx.dino_erode_or_dilate, dino_debug=ctx.debugging_dino,
                cloth_category=enhance_mask_cloth_category, mask_invert=enhance_mask_invert,
                inpaint_erode_or_dilate=enhance_inpaint_erode_or_dilate,
            )

            if ctx.debugging_enhance_masks_checkbox:
                async_task.yields.append(['preview', (current_progress, 'Loading ...', mask)])
                yield_result(async_task, mask, current_progress, ctx.black_out_nsfw, False,
                             ctx.disable_intermediate_results)
                async_task.enhance_stats[index] += 1

            print(f'[Enhance] {dino_detection_count} boxes detected')
            print(f'[Enhance] {sam_detection_count} segments detected in boxes')
            print(f'[Enhance] {sam_detection_on_mask_count} segments applied to mask')

            if enhance_mask_model == 'sam' and (
                dino_detection_count == 0 or not ctx.debugging_dino and sam_detection_on_mask_count == 0
            ):
                print(f'[Enhance] No "{enhance_mask_dino_prompt_text}" detected, skipping')
                continue

            try:
                current_progress, img, enhance_prompt_processed, enhance_negative_prompt_processed = process_enhance(
                    runtime,
                    async_task, ctx, callback, current_progress, current_task_id,
                    enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                    enhance_inpaint_respective_field, enhance_inpaint_strength,
                    enhance_prompt, enhance_negative_prompt, final_scheduler_name,
                    img, mask, preparation_steps, enhance_steps, total_count,
                    persist_image=persist_image,
                )

                async_task.enhance_stats[index] += 1

                if (
                    should_process_enhance_uov
                    and ctx.enhance_uov_processing_order == runtime.flags.enhancement_uov_after
                    and ctx.enhance_uov_prompt_type == runtime.flags.enhancement_uov_prompt_type_last_filled
                ):
                    if enhance_prompt_processed != '':
                        last_enhance_prompt = enhance_prompt_processed
                    if enhance_negative_prompt_processed != '':
                        last_enhance_negative_prompt = enhance_negative_prompt_processed

            except runtime.ldm_model_management.InterruptProcessingException:
                if async_task.last_stop == 'skip':
                    print('User skipped')
                    async_task.last_stop = False
                    continue
                else:
                    print('User stopped')
                    exception_result = 'break'
                    break
            finally:
                done_steps_inpainting += enhance_steps

            enhancement_task_time = runtime.time_module.perf_counter() - enhancement_task_start_time
            print(f'Enhancement time: {enhancement_task_time:.2f} seconds')

        if exception_result == 'break':
            break

        if enhance_uov_after:
            current_task_id, done_steps_inpainting, done_steps_upscaling, img, exception_result = enhance_upscale(
                runtime,
                async_task, ctx, all_steps, base_progress, callback, enhance_steps,
                current_task_id, done_steps_inpainting, done_steps_upscaling,
                img, preparation_steps, total_count, persist_image=True,
                prompt=last_enhance_prompt, negative_prompt=last_enhance_negative_prompt,
            )
            async_task.enhance_stats[index] += 1

            if exception_result == 'continue':
                continue
            elif exception_result == 'break':
                break

        enhancement_image_time = runtime.time_module.perf_counter() - enhancement_image_start_time
        print(f'Enhancement image time: {enhancement_image_time:.2f} seconds')
