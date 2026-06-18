from __future__ import annotations

from typing import Tuple, Any
import numpy as np

import modules.config
import modules.flags as flags
import modules.util
from modules.util import (HWC3, resize_image, set_image_shape_ceil, get_image_shape_ceil,
                          get_shape_ceil, resample_image, erode_or_dilate)
from modules.upscaler import perform_upscale
from extras.inpaint_mask import generate_mask_from_image, SAMOptions
from extras.expansion import safe_str

from .context import GenerationContext


def prepare_enhance_prompt(prompt: str, fallback_prompt: str) -> str:
    from modules.util import remove_empty_str

    if safe_str(prompt) == '' or len(remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')) == 0:
        prompt = fallback_prompt

    return prompt


def generate_enhance_mask(img, mask_model: str, dino_prompt: str, box_threshold: float,
                          text_threshold: float, sam_model: str, max_detections: int,
                          dino_erode_or_dilate: int, dino_debug: bool,
                          cloth_category: str = None, mask_invert: bool = False,
                          inpaint_erode_or_dilate: int = 0):
    extras = {}
    if mask_model == 'sam':
        print(f'[Enhance] Searching for "{dino_prompt}"')
    elif mask_model == 'u2net_cloth_seg':
        extras['cloth_category'] = cloth_category

    mask, dino_detection_count, sam_detection_count, sam_detection_on_mask_count = generate_mask_from_image(
        img, mask_model=mask_model, extras=extras, sam_options=SAMOptions(
            dino_prompt=dino_prompt,
            dino_box_threshold=box_threshold,
            dino_text_threshold=text_threshold,
            dino_erode_or_dilate=dino_erode_or_dilate,
            dino_debug=dino_debug,
            max_detections=max_detections,
            model_type=sam_model,
        ))

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    if int(inpaint_erode_or_dilate) != 0:
        mask = erode_or_dilate(mask, inpaint_erode_or_dilate)

    if mask_invert:
        mask = 255 - mask

    return mask, dino_detection_count, sam_detection_count, sam_detection_on_mask_count


def apply_vary_to_context(ctx: GenerationContext, switch: int, current_progress: int,
                          advance_progress: bool = False) -> Tuple[GenerationContext, int]:
    import modules.default_pipeline as pipeline
    import modules.core as core

    new_ctx = ctx.clone()

    denoising_strength = new_ctx.denoising_strength
    if 'subtle' in new_ctx.uov_method:
        denoising_strength = 0.5
    if 'strong' in new_ctx.uov_method:
        denoising_strength = 0.85
    if new_ctx.overwrite_vary_strength > 0:
        denoising_strength = new_ctx.overwrite_vary_strength

    shape_ceil = get_image_shape_ceil(new_ctx.uov_input_image)
    if shape_ceil < 1024:
        print(f'[Vary] Image is resized because it is too small.')
        shape_ceil = 1024
    elif shape_ceil > 2048:
        print(f'[Vary] Image is resized because it is too big.')
        shape_ceil = 2048

    uov_input_image = set_image_shape_ceil(new_ctx.uov_input_image, shape_ceil)
    initial_pixels = core.numpy_to_pytorch(uov_input_image)

    if advance_progress:
        current_progress += 1

    candidate_vae, _ = pipeline.get_candidate_vae(
        steps=new_ctx.steps,
        switch=switch,
        denoise=denoising_strength,
        refiner_swap_method=new_ctx.refiner_swap_method
    )
    initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
    B, C, H, W = initial_latent['samples'].shape
    width = W * 8
    height = H * 8
    print(f'Final resolution is {str((width, height))}.')

    new_ctx.denoising_strength = denoising_strength
    new_ctx.initial_latent = initial_latent
    new_ctx.width = width
    new_ctx.height = height

    return new_ctx, current_progress


def apply_upscale_to_context(ctx: GenerationContext, switch: int, current_progress: int,
                             advance_progress: bool = False) -> Tuple[GenerationContext, bool, int]:
    import modules.default_pipeline as pipeline
    import modules.core as core

    new_ctx = ctx.clone()

    H, W, C = new_ctx.uov_input_image.shape

    if advance_progress:
        current_progress += 1

    uov_input_image = perform_upscale(new_ctx.uov_input_image)
    print(f'Image upscaled.')

    if '1.5x' in new_ctx.uov_method:
        f = 1.5
    elif '2x' in new_ctx.uov_method:
        f = 2.0
    else:
        f = 1.0

    shape_ceil = get_shape_ceil(H * f, W * f)
    if shape_ceil < 1024:
        print(f'[Upscale] Image is resized because it is too small.')
        uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
        shape_ceil = 1024
    else:
        uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)

    image_is_super_large = shape_ceil > 2800
    if 'fast' in new_ctx.uov_method:
        direct_return = True
    elif image_is_super_large:
        print('Image is too large. Directly returned the SR image. '
              'Usually directly return SR image at 4K resolution '
              'yields better results than SDXL diffusion.')
        direct_return = True
    else:
        direct_return = False

    if direct_return:
        new_ctx.uov_input_image = uov_input_image
        return new_ctx, direct_return, current_progress

    tiled = True
    denoising_strength = 0.382
    if new_ctx.overwrite_upscale_strength > 0:
        denoising_strength = new_ctx.overwrite_upscale_strength

    initial_pixels = core.numpy_to_pytorch(uov_input_image)

    if advance_progress:
        current_progress += 1

    candidate_vae, _ = pipeline.get_candidate_vae(
        steps=new_ctx.steps,
        switch=switch,
        denoise=denoising_strength,
        refiner_swap_method=new_ctx.refiner_swap_method
    )
    initial_latent = core.encode_vae(
        vae=candidate_vae,
        pixels=initial_pixels, tiled=True)
    B, C, H, W = initial_latent['samples'].shape
    width = W * 8
    height = H * 8
    print(f'Final resolution is {str((width, height))}.')

    new_ctx.uov_input_image = uov_input_image
    new_ctx.denoising_strength = denoising_strength
    new_ctx.initial_latent = initial_latent
    new_ctx.tiled = tiled
    new_ctx.width = width
    new_ctx.height = height

    return new_ctx, direct_return, current_progress


def _apply_outpaint(ctx: GenerationContext, inpaint_image, inpaint_mask):
    if len(ctx.outpaint_selections) > 0:
        H, W, C = inpaint_image.shape
        if 'top' in ctx.outpaint_selections:
            inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
            inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                  constant_values=255)
        if 'bottom' in ctx.outpaint_selections:
            inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
            inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                  constant_values=255)

        H, W, C = inpaint_image.shape
        if 'left' in ctx.outpaint_selections:
            inpaint_image = np.pad(inpaint_image, [[0, 0], [int(W * 0.3), 0], [0, 0]], mode='edge')
            inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(W * 0.3), 0]], mode='constant',
                                  constant_values=255)
        if 'right' in ctx.outpaint_selections:
            inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(W * 0.3)], [0, 0]], mode='edge')
            inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(W * 0.3)]], mode='constant',
                                  constant_values=255)

        inpaint_image = np.ascontiguousarray(inpaint_image.copy())
        inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())

    return inpaint_image, inpaint_mask


def apply_inpaint_to_context(ctx: GenerationContext, inpaint_image, inpaint_mask, switch: int,
                             current_progress: int, skip_apply_outpaint: bool = False,
                             advance_progress: bool = False) -> Tuple[GenerationContext, int]:
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.inpaint_worker as inpaint_worker

    new_ctx = ctx.clone()

    if not skip_apply_outpaint:
        inpaint_image, inpaint_mask = _apply_outpaint(new_ctx, inpaint_image, inpaint_mask)

    inpaint_worker.current_task = inpaint_worker.InpaintWorker(
        image=inpaint_image,
        mask=inpaint_mask,
        use_fill=new_ctx.inpaint_strength > 0.99,
        k=new_ctx.inpaint_respective_field
    )

    if advance_progress:
        current_progress += 1

    inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
    inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
    inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

    candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
        steps=new_ctx.steps,
        switch=switch,
        denoise=new_ctx.inpaint_strength,
        refiner_swap_method=new_ctx.refiner_swap_method
    )
    latent_inpaint, latent_mask = core.encode_vae_inpaint(
        mask=inpaint_pixel_mask,
        vae=candidate_vae,
        pixels=inpaint_pixel_image)

    latent_swap = None
    if candidate_vae_swap is not None:
        if advance_progress:
            current_progress += 1
        latent_swap = core.encode_vae(
            vae=candidate_vae_swap,
            pixels=inpaint_pixel_fill)['samples']

    if advance_progress:
        current_progress += 1

    latent_fill = core.encode_vae(
        vae=candidate_vae,
        pixels=inpaint_pixel_fill)['samples']

    inpaint_worker.current_task.load_latent(
        latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

    if new_ctx.inpaint_parameterized:
        pipeline.final_unet = inpaint_worker.current_task.patch(
            inpaint_head_model_path=new_ctx.inpaint_head_model_path,
            inpaint_latent=latent_inpaint,
            inpaint_latent_mask=latent_mask,
            model=pipeline.final_unet
        )

    if not new_ctx.inpaint_disable_initial_latent:
        new_ctx.initial_latent = {'samples': latent_fill}

    B, C, H, W = latent_fill.shape
    height, width = H * 8, W * 8
    final_height, final_width = inpaint_worker.current_task.image.shape[:2]
    print(f'Final resolution is {str((final_width, final_height))}, latent is {str((width, height))}.')

    new_ctx.denoising_strength = new_ctx.inpaint_strength
    new_ctx.width = width
    new_ctx.height = height

    return new_ctx, current_progress


def apply_control_nets_from_context(ctx: GenerationContext, current_progress: int):
    import modules.default_pipeline as pipeline
    import modules.core as core
    import extras.preprocessors as preprocessors
    import extras.ip_adapter as ip_adapter
    import extras.face_crop

    for task in ctx.cn_tasks[flags.cn_canny]:
        cn_img, cn_stop, cn_weight = task
        cn_img = resize_image(HWC3(cn_img), width=ctx.width, height=ctx.height)

        if not ctx.skipping_cn_preprocessor:
            cn_img = preprocessors.canny_pyramid(cn_img, ctx.canny_low_threshold,
                                                 ctx.canny_high_threshold)

        cn_img = HWC3(cn_img)
        task[0] = core.numpy_to_pytorch(cn_img)

    for task in ctx.cn_tasks[flags.cn_cpds]:
        cn_img, cn_stop, cn_weight = task
        cn_img = resize_image(HWC3(cn_img), width=ctx.width, height=ctx.height)

        if not ctx.skipping_cn_preprocessor:
            cn_img = preprocessors.cpds(cn_img)

        cn_img = HWC3(cn_img)
        task[0] = core.numpy_to_pytorch(cn_img)

    for task in ctx.cn_tasks[flags.cn_ip]:
        cn_img, cn_stop, cn_weight = task
        cn_img = HWC3(cn_img)
        cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)
        task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ctx.ip_adapter_path)

    for task in ctx.cn_tasks[flags.cn_ip_face]:
        cn_img, cn_stop, cn_weight = task
        cn_img = HWC3(cn_img)

        if not ctx.skipping_cn_preprocessor:
            cn_img = extras.face_crop.crop_image(cn_img)

        cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)
        task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ctx.ip_adapter_face_path)

    all_ip_tasks = ctx.cn_tasks[flags.cn_ip] + ctx.cn_tasks[flags.cn_ip_face]
    if len(all_ip_tasks) > 0:
        pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)
