from __future__ import annotations

from typing import List, Tuple
import math
import os
import numpy as np
import cv2

import modules.patch
import modules.config
import modules.flags as flags
import fooocus_version
import modules.meta_parser
from modules.private_logger import log

from .context import GenerationContext, SingleTaskContext


def build_metadata_list(ctx: GenerationContext, task_ctx: SingleTaskContext, loras: List[Tuple[str, float]],
                        pid: int, width: int, height: int) -> List[Tuple[str, str, str]]:
    d = [('Prompt', 'prompt', task_ctx.log_positive_prompt),
         ('Negative Prompt', 'negative_prompt', task_ctx.log_negative_prompt),
         ('Fooocus V2 Expansion', 'prompt_expansion', task_ctx.expansion),
         ('Styles', 'styles',
          str(task_ctx.styles if not ctx.use_expansion else ['fooocus_expansion'] + task_ctx.styles)),
         ('Performance', 'performance', ctx.performance_selection.value),
         ('Steps', 'steps', ctx.steps),
         ('Resolution', 'resolution', str((width, height))),
         ('Guidance Scale', 'guidance_scale', ctx.cfg_scale),
         ('Sharpness', 'sharpness', ctx.sharpness),
         ('ADM Guidance', 'adm_guidance', str((
             modules.patch.patch_settings[pid].positive_adm_scale,
             modules.patch.patch_settings[pid].negative_adm_scale,
             modules.patch.patch_settings[pid].adm_scaler_end))),
         ('Base Model', 'base_model', ctx.base_model_name),
         ('Refiner Model', 'refiner_model', ctx.refiner_model_name),
         ('Refiner Switch', 'refiner_switch', ctx.refiner_switch)]

    if ctx.refiner_model_name != 'None':
        if ctx.overwrite_switch > 0:
            d.append(('Overwrite Switch', 'overwrite_switch', ctx.overwrite_switch))
        if ctx.refiner_swap_method != flags.refiner_swap_method:
            d.append(('Refiner Swap Method', 'refiner_swap_method', ctx.refiner_swap_method))
    if modules.patch.patch_settings[pid].adaptive_cfg != modules.config.default_cfg_tsnr:
        d.append(
            ('CFG Mimicking from TSNR', 'adaptive_cfg', modules.patch.patch_settings[pid].adaptive_cfg))

    if ctx.clip_skip > 1:
        d.append(('CLIP Skip', 'clip_skip', ctx.clip_skip))
    d.append(('Sampler', 'sampler', ctx.sampler_name))
    d.append(('Scheduler', 'scheduler', ctx.scheduler_name))
    d.append(('VAE', 'vae', ctx.vae_name))
    d.append(('Seed', 'seed', str(task_ctx.task_seed)))

    if ctx.freeu_enabled:
        d.append(('FreeU', 'freeu',
                  str((ctx.freeu_b1, ctx.freeu_b2, ctx.freeu_s1, ctx.freeu_s2))))

    for li, (n, w) in enumerate(loras):
        if n != 'None':
            d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))

    d.append(('Metadata Scheme', 'metadata_scheme',
              ctx.metadata_scheme.value if ctx.save_metadata_to_images else ctx.save_metadata_to_images))
    d.append(('Version', 'version', 'Fooocus v' + fooocus_version.version))

    return d


def setup_metadata_parser(ctx: GenerationContext, task_ctx: SingleTaskContext, loras: List[Tuple[str, float]]):
    if not ctx.save_metadata_to_images:
        return None

    metadata_parser = modules.meta_parser.get_metadata_parser(ctx.metadata_scheme)
    metadata_parser.set_data(task_ctx.log_positive_prompt, task_ctx.positive,
                             task_ctx.log_negative_prompt, task_ctx.negative,
                             ctx.steps, ctx.base_model_name, ctx.refiner_model_name,
                             loras, ctx.vae_name)
    return metadata_parser


def save_images_with_metadata(ctx: GenerationContext, task_ctx: SingleTaskContext, imgs: List,
                              loras: List[Tuple[str, float]], pid: int, width: int, height: int,
                              persist_image: bool = True) -> List[str]:
    img_paths = []
    metadata_list = build_metadata_list(ctx, task_ctx, loras, pid, width, height)
    metadata_parser = setup_metadata_parser(ctx, task_ctx, loras)

    task_dict = {
        'positive': task_ctx.positive,
        'negative': task_ctx.negative,
        'seed': task_ctx.task_seed,
        'prompt': task_ctx.task_prompt
    }

    for x in imgs:
        img_paths.append(log(x, metadata_list, metadata_parser, ctx.output_format, task_dict, persist_image))

    return img_paths


def build_image_wall_from_results(results: List):
    if len(results) < 2:
        return None

    processed = []
    for img in results:
        if isinstance(img, str) and os.path.exists(img):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not isinstance(img, np.ndarray):
            return None
        if img.ndim != 3:
            return None
        processed.append(img)

    if len(processed) < 2:
        return None

    H, W, C = processed[0].shape

    for img in processed:
        Hn, Wn, Cn = img.shape
        if H != Hn or W != Wn or C != Cn:
            return None

    cols = float(len(processed)) ** 0.5
    cols = int(math.ceil(cols))
    rows = float(len(processed)) / float(cols)
    rows = int(math.ceil(rows))

    wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

    for y in range(rows):
        for x in range(cols):
            if y * cols + x < len(processed):
                img = processed[y * cols + x]
                wall[y * H:y * H + H, x * W:x * W + W, :] = img

    return wall
