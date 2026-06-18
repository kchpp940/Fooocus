from __future__ import annotations

from typing import List, Tuple, Any
import copy
import random

import args_manager
import modules.config
import modules.util
import modules.constants as constants
import modules.flags as flags
from modules.flags import Performance, MetadataScheme, disabled
from modules.patch import PatchSettings, patch_settings
from modules.sdxl_styles import apply_style, get_random_style, fooocus_expansion, apply_arrays, random_style_name
from modules.util import (remove_empty_str, parse_lora_references_from_prompt, apply_wildcards, HWC3, resample_image, erode_or_dilate)
from extras.expansion import safe_str
from extras.inpaint_mask import SAMOptions

from .context import GenerationContext, SingleTaskContext
from .progress import ProgressReporter
from .result_saver import save_images_with_metadata


def build_generation_context(async_task) -> GenerationContext:
    ctx = GenerationContext()

    ctx.performance_selection = async_task.performance_selection
    ctx.steps = async_task.steps
    ctx.original_steps = async_task.original_steps
    ctx.aspect_ratios_selection = async_task.aspect_ratios_selection if hasattr(async_task, 'aspect_ratios_selection') else '1024×1024'
    ctx.image_number = async_task.image_number
    ctx.output_format = async_task.output_format
    ctx.seed = async_task.seed
    ctx.read_wildcards_in_order = async_task.read_wildcards_in_order
    ctx.sharpness = async_task.sharpness
    ctx.cfg_scale = async_task.cfg_scale
    ctx.base_model_name = async_task.base_model_name
    ctx.refiner_model_name = async_task.refiner_model_name
    ctx.refiner_switch = async_task.refiner_switch
    ctx.loras = list(async_task.loras)
    ctx.input_image_checkbox = async_task.input_image_checkbox
    ctx.current_tab = async_task.current_tab
    ctx.uov_method = async_task.uov_method.casefold()
    ctx.uov_input_image = async_task.uov_input_image
    ctx.outpaint_selections = [o.lower() for o in async_task.outpaint_selections]
    ctx.inpaint_input_image = async_task.inpaint_input_image
    ctx.inpaint_additional_prompt = async_task.inpaint_additional_prompt
    ctx.inpaint_mask_image_upload = async_task.inpaint_mask_image_upload

    ctx.disable_preview = async_task.disable_preview
    ctx.disable_intermediate_results = async_task.disable_intermediate_results
    ctx.disable_seed_increment = async_task.disable_seed_increment
    ctx.black_out_nsfw = async_task.black_out_nsfw
    ctx.adm_scaler_positive = async_task.adm_scaler_positive
    ctx.adm_scaler_negative = async_task.adm_scaler_negative
    ctx.adm_scaler_end = async_task.adm_scaler_end
    ctx.adaptive_cfg = async_task.adaptive_cfg
    ctx.clip_skip = async_task.clip_skip
    ctx.sampler_name = async_task.sampler_name
    ctx.scheduler_name = async_task.scheduler_name
    ctx.vae_name = async_task.vae_name
    ctx.overwrite_step = async_task.overwrite_step
    ctx.overwrite_switch = async_task.overwrite_switch
    ctx.overwrite_width = async_task.overwrite_width
    ctx.overwrite_height = async_task.overwrite_height
    ctx.overwrite_vary_strength = async_task.overwrite_vary_strength
    ctx.overwrite_upscale_strength = async_task.overwrite_upscale_strength
    ctx.mixing_image_prompt_and_vary_upscale = async_task.mixing_image_prompt_and_vary_upscale
    ctx.mixing_image_prompt_and_inpaint = async_task.mixing_image_prompt_and_inpaint
    ctx.debugging_cn_preprocessor = async_task.debugging_cn_preprocessor
    ctx.skipping_cn_preprocessor = async_task.skipping_cn_preprocessor
    ctx.canny_low_threshold = async_task.canny_low_threshold
    ctx.canny_high_threshold = async_task.canny_high_threshold
    ctx.refiner_swap_method = async_task.refiner_swap_method
    ctx.controlnet_softness = async_task.controlnet_softness
    ctx.freeu_enabled = async_task.freeu_enabled
    ctx.freeu_b1 = async_task.freeu_b1
    ctx.freeu_b2 = async_task.freeu_b2
    ctx.freeu_s1 = async_task.freeu_s1
    ctx.freeu_s2 = async_task.freeu_s2
    ctx.debugging_inpaint_preprocessor = async_task.debugging_inpaint_preprocessor
    ctx.inpaint_disable_initial_latent = async_task.inpaint_disable_initial_latent
    ctx.inpaint_engine = async_task.inpaint_engine
    ctx.inpaint_strength = async_task.inpaint_strength
    ctx.inpaint_respective_field = async_task.inpaint_respective_field
    ctx.inpaint_advanced_masking_checkbox = async_task.inpaint_advanced_masking_checkbox
    ctx.invert_mask_checkbox = async_task.invert_mask_checkbox
    ctx.inpaint_erode_or_dilate = async_task.inpaint_erode_or_dilate
    ctx.save_final_enhanced_image_only = async_task.save_final_enhanced_image_only if not args_manager.args.disable_image_log else False
    ctx.save_metadata_to_images = async_task.save_metadata_to_images if not args_manager.args.disable_metadata else False
    ctx.metadata_scheme = async_task.metadata_scheme if not args_manager.args.disable_metadata else MetadataScheme.FOOOCUS

    ctx.cn_tasks = {k: [list(item) for item in v] for k, v in async_task.cn_tasks.items()}

    ctx.debugging_dino = async_task.debugging_dino
    ctx.dino_erode_or_dilate = async_task.dino_erode_or_dilate
    ctx.debugging_enhance_masks_checkbox = async_task.debugging_enhance_masks_checkbox

    ctx.enhance_input_image = async_task.enhance_input_image
    ctx.enhance_checkbox = async_task.enhance_checkbox
    ctx.enhance_uov_method = async_task.enhance_uov_method.casefold()
    ctx.enhance_uov_processing_order = async_task.enhance_uov_processing_order
    ctx.enhance_uov_prompt_type = async_task.enhance_uov_prompt_type
    ctx.enhance_ctrls = [list(ctrl) for ctrl in async_task.enhance_ctrls]
    ctx.should_enhance = async_task.should_enhance
    ctx.prompt = async_task.prompt
    ctx.negative_prompt = async_task.negative_prompt
    ctx.style_selections = list(async_task.style_selections)
    ctx.performance_loras = list(async_task.performance_loras)

    return ctx


def parse_dimensions_from_aspect_ratio(aspect_ratios_selection: str) -> Tuple[int, int]:
    width, height = aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
    return int(width), int(height)


def _set_hyper_sd_defaults(ctx: GenerationContext, current_progress: int, advance_progress=False):
    print('Enter Hyper-SD mode.')
    if advance_progress:
        current_progress += 1
    ctx.performance_loras += [(modules.config.downloading_sdxl_hyper_sd_lora(), 0.8)]
    if ctx.refiner_model_name != 'None':
        print(f'Refiner disabled in Hyper-SD mode.')
    ctx.refiner_model_name = 'None'
    ctx.sampler_name = 'dpmpp_sde_gpu'
    ctx.scheduler_name = 'karras'
    ctx.sharpness = 0.0
    ctx.cfg_scale = 1.0
    ctx.adaptive_cfg = 1.0
    ctx.refiner_switch = 1.0
    ctx.adm_scaler_positive = 1.0
    ctx.adm_scaler_negative = 1.0
    ctx.adm_scaler_end = 0.0
    return current_progress


def _set_lightning_defaults(ctx: GenerationContext, current_progress: int, advance_progress=False):
    print('Enter Lightning mode.')
    if advance_progress:
        current_progress += 1
    ctx.performance_loras += [(modules.config.downloading_sdxl_lightning_lora(), 1.0)]
    if ctx.refiner_model_name != 'None':
        print(f'Refiner disabled in Lightning mode.')
    ctx.refiner_model_name = 'None'
    ctx.sampler_name = 'euler'
    ctx.scheduler_name = 'sgm_uniform'
    ctx.sharpness = 0.0
    ctx.cfg_scale = 1.0
    ctx.adaptive_cfg = 1.0
    ctx.refiner_switch = 1.0
    ctx.adm_scaler_positive = 1.0
    ctx.adm_scaler_negative = 1.0
    ctx.adm_scaler_end = 0.0
    return current_progress


def _set_lcm_defaults(ctx: GenerationContext, current_progress: int, advance_progress=False):
    print('Enter LCM mode.')
    if advance_progress:
        current_progress += 1
    ctx.performance_loras += [(modules.config.downloading_sdxl_lcm_lora(), 1.0)]
    if ctx.refiner_model_name != 'None':
        print(f'Refiner disabled in LCM mode.')
    ctx.refiner_model_name = 'None'
    ctx.sampler_name = 'lcm'
    ctx.scheduler_name = 'lcm'
    ctx.sharpness = 0.0
    ctx.cfg_scale = 1.0
    ctx.adaptive_cfg = 1.0
    ctx.refiner_switch = 1.0
    ctx.adm_scaler_positive = 1.0
    ctx.adm_scaler_negative = 1.0
    ctx.adm_scaler_end = 0.0
    return current_progress


def normalize_performance_settings(ctx: GenerationContext, pid: int, current_progress: int = 0) -> int:
    if ctx.performance_selection == Performance.EXTREME_SPEED:
        current_progress = _set_lcm_defaults(ctx, current_progress, advance_progress=True)
    elif ctx.performance_selection == Performance.LIGHTNING:
        current_progress = _set_lightning_defaults(ctx, current_progress, advance_progress=True)
    elif ctx.performance_selection == Performance.HYPER_SD:
        current_progress = _set_hyper_sd_defaults(ctx, current_progress, advance_progress=True)

    if ctx.base_model_name == ctx.refiner_model_name:
        print(f'Refiner disabled because base model and refiner are same.')
        ctx.refiner_model_name = 'None'

    return current_progress


def apply_patch_settings_to_context(ctx: GenerationContext, pid: int):
    patch_settings[pid] = PatchSettings(
        ctx.sharpness,
        ctx.adm_scaler_end,
        ctx.adm_scaler_positive,
        ctx.adm_scaler_negative,
        ctx.controlnet_softness,
        ctx.adaptive_cfg
    )


def apply_overrides_to_context(ctx: GenerationContext) -> Tuple[int, int, int, int]:
    steps = ctx.steps
    width = ctx.width
    height = ctx.height

    if ctx.overwrite_step > 0:
        steps = ctx.overwrite_step
    switch = int(round(ctx.steps * ctx.refiner_switch))
    if ctx.overwrite_switch > 0:
        switch = ctx.overwrite_switch
    if ctx.overwrite_width > 0:
        width = ctx.overwrite_width
    if ctx.overwrite_height > 0:
        height = ctx.overwrite_height

    return steps, switch, width, height


def expand_tasks_from_prompt(ctx: GenerationContext, base_model_additional_loras: List[Tuple[str, float]],
                              current_progress: int = 0, advance_progress: bool = False):
    import modules.default_pipeline as pipeline

    prompts = remove_empty_str([safe_str(p) for p in ctx.prompt.splitlines()], default='')
    negative_prompts = remove_empty_str([safe_str(p) for p in ctx.negative_prompt.splitlines()], default='')
    prompt = prompts[0]
    negative_prompt = negative_prompts[0]

    if prompt == '':
        ctx.use_expansion = False

    extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
    extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

    if advance_progress:
        current_progress += 1

    lora_filenames = modules.util.remove_performance_lora(modules.config.lora_filenames,
                                                          ctx.performance_selection)
    loras, prompt = parse_lora_references_from_prompt(prompt, ctx.loras,
                                                      modules.config.default_max_lora_number,
                                                      lora_filenames=lora_filenames)
    loras += ctx.performance_loras

    tasks = []
    for i in range(ctx.image_number):
        if ctx.disable_seed_increment:
            task_seed = ctx.seed % (constants.MAX_SEED + 1)
        else:
            task_seed = (ctx.seed + i) % (constants.MAX_SEED + 1)

        task_rng = random.Random(task_seed)
        task_prompt = apply_wildcards(prompt, task_rng, i, ctx.read_wildcards_in_order)
        task_prompt = apply_arrays(task_prompt, i)
        task_negative_prompt = apply_wildcards(negative_prompt, task_rng, i, ctx.read_wildcards_in_order)
        task_extra_positive_prompts = [apply_wildcards(pmt, task_rng, i, ctx.read_wildcards_in_order) for pmt
                                       in extra_positive_prompts]
        task_extra_negative_prompts = [apply_wildcards(pmt, task_rng, i, ctx.read_wildcards_in_order) for pmt
                                       in extra_negative_prompts]

        positive_basic_workloads = []
        negative_basic_workloads = []

        task_styles = ctx.style_selections.copy()
        if ctx.use_style:
            placeholder_replaced = False

            for j, s in enumerate(task_styles):
                if s == random_style_name:
                    s = get_random_style(task_rng)
                    task_styles[j] = s
                p, n, style_has_placeholder = apply_style(s, positive=task_prompt)
                if style_has_placeholder:
                    placeholder_replaced = True
                positive_basic_workloads = positive_basic_workloads + p
                negative_basic_workloads = negative_basic_workloads + n

            if not placeholder_replaced:
                positive_basic_workloads = [task_prompt] + positive_basic_workloads
        else:
            positive_basic_workloads.append(task_prompt)

        negative_basic_workloads.append(task_negative_prompt)

        positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
        negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

        positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
        negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

        task_ctx = SingleTaskContext(
            task_seed=task_seed,
            task_prompt=task_prompt,
            task_negative_prompt=task_negative_prompt,
            positive=positive_basic_workloads,
            negative=negative_basic_workloads,
            expansion='',
            positive_top_k=len(positive_basic_workloads),
            negative_top_k=len(negative_basic_workloads),
            log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
            log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
            styles=task_styles
        )
        tasks.append(task_ctx)

    if ctx.use_expansion:
        if advance_progress:
            current_progress += 1
        for i, t in enumerate(tasks):
            expansion = pipeline.final_expansion(t.task_prompt, t.task_seed)
            print(f'[Prompt Expansion] {expansion}')
            t.expansion = expansion
            t.positive = copy.deepcopy(t.positive) + [expansion]

    if advance_progress:
        current_progress += 1
    for i, t in enumerate(tasks):
        t.c = pipeline.clip_encode(texts=t.positive, pool_top_k=t.positive_top_k)

    if advance_progress:
        current_progress += 1
    for i, t in enumerate(tasks):
        if abs(float(ctx.cfg_scale) - 1.0) < 1e-4:
            t.uc = pipeline.clone_cond(t.c)
        else:
            t.uc = pipeline.clip_encode(texts=t.negative, pool_top_k=t.negative_top_k)

    return tasks, loras, current_progress


def process_image_input(ctx: GenerationContext, current_progress: int = 0):
    import numpy as np

    inpaint_image = None
    inpaint_mask = None

    if (ctx.current_tab == 'uov' or (
            ctx.current_tab == 'ip' and ctx.mixing_image_prompt_and_vary_upscale)) \
            and ctx.uov_method != disabled.casefold() and ctx.uov_input_image is not None:
        ctx.uov_input_image, ctx.skip_prompt_processing, ctx.steps = _prepare_upscale(
            ctx, ctx.uov_input_image, ctx.uov_method, ctx.steps, current_progress)

    if (ctx.current_tab == 'inpaint' or (
            ctx.current_tab == 'ip' and ctx.mixing_image_prompt_and_inpaint)) \
            and isinstance(ctx.inpaint_input_image, dict):
        inpaint_image = ctx.inpaint_input_image['image']
        inpaint_mask = ctx.inpaint_input_image['mask'][:, :, 0]

        if ctx.inpaint_advanced_masking_checkbox:
            if isinstance(ctx.inpaint_mask_image_upload, dict):
                if (isinstance(ctx.inpaint_mask_image_upload['image'], np.ndarray)
                        and isinstance(ctx.inpaint_mask_image_upload['mask'], np.ndarray)
                        and ctx.inpaint_mask_image_upload['image'].ndim == 3):
                    ctx.inpaint_mask_image_upload = np.maximum(
                        ctx.inpaint_mask_image_upload['image'],
                        ctx.inpaint_mask_image_upload['mask'])
            if isinstance(ctx.inpaint_mask_image_upload,
                          np.ndarray) and ctx.inpaint_mask_image_upload.ndim == 3:
                H, W, C = inpaint_image.shape
                ctx.inpaint_mask_image_upload = resample_image(ctx.inpaint_mask_image_upload,
                                                               width=W, height=H)
                ctx.inpaint_mask_image_upload = np.mean(ctx.inpaint_mask_image_upload, axis=2)
                ctx.inpaint_mask_image_upload = (ctx.inpaint_mask_image_upload > 127).astype(
                    np.uint8) * 255
                inpaint_mask = np.maximum(inpaint_mask, ctx.inpaint_mask_image_upload)

        if int(ctx.inpaint_erode_or_dilate) != 0:
            inpaint_mask = erode_or_dilate(inpaint_mask, ctx.inpaint_erode_or_dilate)

        if ctx.invert_mask_checkbox:
            inpaint_mask = 255 - inpaint_mask

        inpaint_image = HWC3(inpaint_image)
        if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                and (np.any(inpaint_mask > 127) or len(ctx.outpaint_selections) > 0):
            modules.config.downloading_upscale_model()
            if ctx.inpaint_parameterized:
                ctx.inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                    ctx.inpaint_engine)
                ctx.base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                if ctx.refiner_model_name == 'None':
                    ctx.use_synthetic_refiner = True
                    ctx.refiner_switch = 0.8
            else:
                ctx.inpaint_head_model_path = None
                print(f'[Inpaint] Parameterized inpaint is disabled.')
            if ctx.inpaint_additional_prompt != '':
                if ctx.prompt == '':
                    ctx.prompt = ctx.inpaint_additional_prompt
                else:
                    ctx.prompt = ctx.inpaint_additional_prompt + '\n' + ctx.prompt
            ctx.goals.append('inpaint')

    if ctx.current_tab == 'ip' or \
            ctx.mixing_image_prompt_and_vary_upscale or \
            ctx.mixing_image_prompt_and_inpaint:
        ctx.goals.append('cn')
        if len(ctx.cn_tasks[flags.cn_canny]) > 0:
            ctx.controlnet_canny_path = modules.config.downloading_controlnet_canny()
        if len(ctx.cn_tasks[flags.cn_cpds]) > 0:
            ctx.controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
        if len(ctx.cn_tasks[flags.cn_ip]) > 0:
            ctx.clip_vision_path, ctx.ip_negative_path, ctx.ip_adapter_path = modules.config.downloading_ip_adapters('ip')
        if len(ctx.cn_tasks[flags.cn_ip_face]) > 0:
            ctx.clip_vision_path, ctx.ip_negative_path, ctx.ip_adapter_face_path = modules.config.downloading_ip_adapters(
                'face')

    if ctx.current_tab == 'enhance' and ctx.enhance_input_image is not None:
        ctx.goals.append('enhance')
        ctx.skip_prompt_processing = True
        ctx.enhance_input_image = HWC3(ctx.enhance_input_image)

    return inpaint_image, inpaint_mask, current_progress


def _prepare_upscale(ctx: GenerationContext, uov_input_image, uov_method, steps,
                     current_progress, advance_progress=False, skip_prompt_processing=False):
    from modules.upscaler import perform_upscale

    uov_input_image = HWC3(uov_input_image)
    if 'vary' in uov_method:
        ctx.goals.append('vary')
    elif 'upscale' in uov_method:
        ctx.goals.append('upscale')
        if 'fast' in uov_method:
            skip_prompt_processing = True
            steps = 0
        else:
            steps = ctx.performance_selection.steps_uov()

        if advance_progress:
            current_progress += 1
        modules.config.downloading_upscale_model()
    return uov_input_image, skip_prompt_processing, steps


def patch_samplers_from_context(ctx: GenerationContext):
    import modules.default_pipeline as pipeline
    import modules.core as core

    final_scheduler_name = ctx.scheduler_name

    if ctx.scheduler_name in ['lcm', 'tcd']:
        final_scheduler_name = 'sgm_uniform'
        if pipeline.final_unet is not None:
            pipeline.final_unet = core.opModelSamplingDiscrete.patch(pipeline.final_unet, ctx.scheduler_name, False)[0]
        if pipeline.final_refiner_unet is not None:
            pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(pipeline.final_refiner_unet, ctx.scheduler_name, False)[0]

    elif ctx.scheduler_name == 'edm_playground_v2.5':
        final_scheduler_name = 'karras'
        if pipeline.final_unet is not None:
            pipeline.final_unet = core.opModelSamplingContinuousEDM.patch(pipeline.final_unet, ctx.scheduler_name, 120.0, 0.002)[0]
        if pipeline.final_refiner_unet is not None:
            pipeline.final_refiner_unet = core.opModelSamplingContinuousEDM.patch(pipeline.final_refiner_unet, ctx.scheduler_name, 120.0, 0.002)[0]

    return final_scheduler_name


def apply_freeu_from_context(ctx: GenerationContext):
    import modules.default_pipeline as pipeline
    import modules.core as core

    print(f'FreeU is enabled!')
    pipeline.final_unet = core.apply_freeu(
        pipeline.final_unet,
        ctx.freeu_b1,
        ctx.freeu_b2,
        ctx.freeu_s1,
        ctx.freeu_s2
    )


def compute_total_steps(ctx: GenerationContext) -> int:
    all_steps = ctx.steps * ctx.image_number

    if ctx.enhance_checkbox and ctx.enhance_uov_method != flags.disabled.casefold():
        enhance_upscale_steps = ctx.performance_selection.steps()
        if 'upscale' in ctx.enhance_uov_method:
            if 'fast' in ctx.enhance_uov_method:
                enhance_upscale_steps = 0
            else:
                enhance_upscale_steps = ctx.performance_selection.steps_uov()
        temp_ctx = ctx.clone()
        temp_ctx.steps = enhance_upscale_steps
        enhance_upscale_steps, _, _, _ = apply_overrides_to_context(temp_ctx)
        all_steps += ctx.image_number * enhance_upscale_steps

    if ctx.enhance_checkbox and len(ctx.enhance_ctrls) != 0:
        temp_ctx = ctx.clone()
        temp_ctx.steps = ctx.original_steps
        enhance_steps, _, _, _ = apply_overrides_to_context(temp_ctx)
        all_steps += ctx.image_number * len(ctx.enhance_ctrls) * enhance_steps

    all_steps = max(all_steps, 1)
    return all_steps


def execute_diffusion_task(ctx: GenerationContext, task_ctx: SingleTaskContext, loras: List[Tuple[str, float]],
                           final_scheduler_name: str, callback, current_task_id: int, total_count: int,
                           base_progress: int, preparation_steps: int, all_steps: int,
                           show_intermediate_results: bool, persist_image: bool,
                           progress_reporter: ProgressReporter, pid: int) -> Tuple[List, List[str], int]:
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.inpaint_worker as inpaint_worker

    progress_reporter.set_current_task(current_task_id)

    if 'cn' in ctx.goals:
        for cn_flag, cn_path in [
            (flags.cn_canny, ctx.controlnet_canny_path),
            (flags.cn_cpds, ctx.controlnet_cpds_path)
        ]:
            for cn_img, cn_stop, cn_weight in ctx.cn_tasks[cn_flag]:
                task_ctx.c, task_ctx.uc = core.apply_controlnet(
                    task_ctx.c, task_ctx.uc,
                    pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)

    imgs = pipeline.process_diffusion(
        positive_cond=task_ctx.c,
        negative_cond=task_ctx.uc,
        steps=ctx.steps,
        switch=ctx.switch,
        width=ctx.width,
        height=ctx.height,
        image_seed=task_ctx.task_seed,
        callback=callback,
        sampler_name=ctx.sampler_name,
        scheduler_name=final_scheduler_name,
        latent=ctx.initial_latent,
        denoise=ctx.denoising_strength,
        tiled=ctx.tiled,
        cfg_scale=ctx.cfg_scale,
        refiner_swap_method=ctx.refiner_swap_method,
        disable_preview=ctx.disable_preview
    )

    del task_ctx.c, task_ctx.uc

    if inpaint_worker.current_task is not None:
        imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

    current_progress = int(base_progress + (100 - preparation_steps) / float(all_steps) * ctx.steps)

    img_paths = save_images_with_metadata(ctx, task_ctx, imgs, loras, pid, ctx.width, ctx.height, persist_image)

    progress_reporter.yield_result(img_paths, current_progress, censor=False,
                                   do_not_show_finished_images=not show_intermediate_results or ctx.disable_intermediate_results)

    return imgs, img_paths, current_progress
