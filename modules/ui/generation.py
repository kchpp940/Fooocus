import os
import time
import random
import json
import gradio as gr

import shared
import modules.config
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import args_manager


def get_task(*args):
    args = list(args)
    args.pop(0)
    return worker.AsyncTask(args=args)


def generate_clicked(task: worker.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=False)

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                if len(task.yields) > 0:
                    if task.yields[0][0] == 'preview':
                        continue

                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False)
            if flag == 'finish':
                if not args_manager.args.disable_enhance_output_sorting:
                    product = sort_enhance_images(product, task)

                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product)
                finished = True

                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


def sort_enhance_images(images, task):
    if not task.should_enhance or len(images) <= task.images_to_enhance_count:
        return images

    sorted_images = []
    walk_index = task.images_to_enhance_count

    for index, enhanced_img in enumerate(images[:task.images_to_enhance_count]):
        sorted_images.append(enhanced_img)
        if index not in task.enhance_stats:
            continue
        target_index = walk_index + task.enhance_stats[index]
        if walk_index < len(images) and target_index <= len(images):
            sorted_images += images[walk_index:target_index]
        walk_index += task.enhance_stats[index]

    return sorted_images


def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5, 0.0
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0, 0.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 1.0, 0.618
    ]


def stop_clicked(currentTask):
    import ldm_patched.modules.model_management as model_management
    currentTask.last_stop = 'stop'
    if currentTask.processing:
        model_management.interrupt_current_processing()
    return currentTask


def skip_clicked(currentTask):
    import ldm_patched.modules.model_management as model_management
    currentTask.last_stop = 'skip'
    if currentTask.processing:
        model_management.interrupt_current_processing()
    return currentTask


def refresh_seed(r, seed_string):
    if r:
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
    else:
        try:
            seed_value = int(seed_string)
            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                return seed_value
        except ValueError:
            pass
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)


def parse_meta(raw_prompt_txt, is_generating):
    from modules.util import is_json
    loaded_json = None
    if is_json(raw_prompt_txt):
        loaded_json = json.loads(raw_prompt_txt)

    if loaded_json is None:
        if is_generating:
            return gr.update(), gr.update(), gr.update()
        else:
            return gr.update(), gr.update(visible=True), gr.update(visible=False)

    return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)


def random_checked(r):
    return gr.update(visible=not r)


def trigger_describe(modes, img, apply_styles):
    describe_prompts = []
    styles = set()

    if flags.describe_type_photo in modes:
        from extras.interrogate import default_interrogator as default_interrogator_photo
        describe_prompts.append(default_interrogator_photo(img))
        styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])

    if flags.describe_type_anime in modes:
        from extras.wd14tagger import default_interrogator as default_interrogator_anime
        describe_prompts.append(default_interrogator_anime(img))
        styles.update(["Fooocus V2", "Fooocus Masterpiece"])

    if len(styles) == 0 or not apply_styles:
        styles = gr.update()
    else:
        styles = list(styles)

    if len(describe_prompts) == 0:
        describe_prompt = gr.update()
    else:
        describe_prompt = ', '.join(describe_prompts)

    return describe_prompt, styles


def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
    from extras.inpaint_mask import generate_mask_from_image
    from extras.inpaint_mask import SAMOptions

    extras = {}
    sam_options = None
    if mask_model == 'u2net_cloth_seg':
        extras['cloth_category'] = cloth_category
    elif mask_model == 'sam':
        sam_options = SAMOptions(
            dino_prompt=dino_prompt_text,
            dino_box_threshold=box_threshold,
            dino_text_threshold=text_threshold,
            dino_erode_or_dilate=dino_erode_or_dilate,
            dino_debug=dino_debug,
            max_detections=sam_max_detections,
            model_type=sam_model
        )

    mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)

    return mask


def build_preset_data_from_ui(*args):
    preset_data = {}

    arg_idx = 0
    default_model = args[arg_idx]; arg_idx += 1
    default_refiner = args[arg_idx]; arg_idx += 1
    default_refiner_switch = args[arg_idx]; arg_idx += 1
    default_cfg_scale = args[arg_idx]; arg_idx += 1
    default_sample_sharpness = args[arg_idx]; arg_idx += 1
    default_cfg_tsnr = args[arg_idx]; arg_idx += 1
    default_clip_skip = args[arg_idx]; arg_idx += 1
    default_sampler = args[arg_idx]; arg_idx += 1
    default_scheduler = args[arg_idx]; arg_idx += 1
    default_vae = args[arg_idx]; arg_idx += 1
    default_performance = args[arg_idx]; arg_idx += 1
    default_aspect_ratio_label = args[arg_idx]; arg_idx += 1
    default_styles = args[arg_idx]; arg_idx += 1
    default_overwrite_step = args[arg_idx]; arg_idx += 1
    default_inpaint_engine_version = args[arg_idx]; arg_idx += 1

    lora_count = modules.config.default_max_lora_number
    default_loras = []
    for _ in range(lora_count):
        enabled = args[arg_idx]; arg_idx += 1
        model = args[arg_idx]; arg_idx += 1
        weight = args[arg_idx]; arg_idx += 1
        default_loras.append([enabled, model if model else 'None', float(weight)])

    if '×' in str(default_aspect_ratio_label):
        ratio_part = str(default_aspect_ratio_label).split(' ')[0]
        default_aspect_ratio = ratio_part.replace('×', '*')
    else:
        default_aspect_ratio = str(default_aspect_ratio_label).replace('×', '*')

    preset_data = {
        'default_model': default_model if default_model else 'model.safetensors',
        'default_refiner': default_refiner if default_refiner else 'None',
        'default_refiner_switch': float(default_refiner_switch),
        'default_loras': default_loras,
        'default_cfg_scale': float(default_cfg_scale),
        'default_sample_sharpness': float(default_sample_sharpness),
        'default_cfg_tsnr': float(default_cfg_tsnr),
        'default_clip_skip': int(default_clip_skip),
        'default_sampler': default_sampler,
        'default_scheduler': default_scheduler,
        'default_vae': default_vae if default_vae != modules.flags.default_vae else 'Default (model)',
        'default_performance': default_performance,
        'default_aspect_ratio': default_aspect_ratio,
        'default_styles': list(default_styles) if default_styles else [],
        'default_overwrite_step': int(default_overwrite_step),
        'default_inpaint_engine_version': default_inpaint_engine_version,
        'checkpoint_downloads': {},
        'embeddings_downloads': {},
        'lora_downloads': {},
        'vae_downloads': {},
    }
    return preset_data


def format_preset_details_html(preset_name):
    details = modules.config.get_preset_details(preset_name)
    if details is None:
        return '<div style="color: #888;">No preset selected</div>'

    name = details['name']
    ptype = details['type']
    type_labels = {'initial': '⚪ Initial', 'builtin': '🔷 Built-in', 'user': '⭐ User'}
    type_label = type_labels.get(ptype, ptype)

    html_parts = [
        f'<div style="margin-bottom: 10px; padding: 8px; background: #f0f0f0; border-radius: 6px;">',
        f'<div style="font-weight: bold; font-size: 14px;">{name}</div>',
        f'<div style="color: #666; font-size: 12px;">{type_label}</div>',
        '</div>',
        '<div style="max-height: 300px; overflow-y: auto; padding: 4px;">'
    ]

    if not details['details']:
        html_parts.append('<div style="color: #888; font-style: italic;">(uses current default settings)</div>')
    else:
        for key, value in details['details'].items():
            html_parts.append(f'<div style="margin: 4px 0; padding: 4px 0; border-bottom: 1px solid #eee;">')
            html_parts.append(f'<span style="font-weight: 600; color: #444;">{key}:</span> ')
            if isinstance(value, list):
                html_parts.append('<ul style="margin: 4px 0 0 20px; padding: 0;">')
                for item in value:
                    html_parts.append(f'<li style="font-size: 12px; color: #555;">{item}</li>')
                html_parts.append('</ul>')
            else:
                html_parts.append(f'<span style="color: #2563eb;">{value}</span>')
            html_parts.append('</div>')

    html_parts.append('</div>')
    return ''.join(html_parts)
