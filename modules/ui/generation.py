import os
import json
import random
import time

import gradio as gr

import modules.config
import modules.flags as flags
import modules.constants as constants
import modules.async_worker as worker
import modules.html
import modules.util
import modules.style_sorter as style_sorter
import args_manager
import launch
from extras.inpaint_mask import SAMOptions, generate_mask_from_image
from modules.metadata_service import get_metadata_service, MetadataSource
from modules.private_logger import get_current_html_path
from modules.util import is_json


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


def random_checked(r):
    return gr.update(visible=not r)


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


def update_history_link(output_format):
    if args_manager.args.disable_image_log:
        return gr.update(value='')
    return gr.update(value=f'<a href="file={get_current_html_path(output_format)}" target="_blank">\U0001F4DA History Log</a>')


def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
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


def trigger_show_image_properties(image):
    value = modules.util.get_image_size_info(image, modules.flags.sdxl_aspect_ratios)
    return gr.update(value=value, visible=True)


def ip_advance_checked(x, ip_ad_cols, ip_types, ip_stops, ip_weights):
    return [gr.update(visible=x)] * len(ip_ad_cols) + \
        [flags.default_ip] * len(ip_types) + \
        [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
        [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)


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


def trigger_auto_describe(mode, img, prompt, apply_styles):
    if prompt == '':
        return trigger_describe(mode, img, apply_styles)
    return gr.update(), gr.update()


def trigger_metadata_preview(filepath):
    service = get_metadata_service()
    if filepath is None:
        return {}
    try:
        parsed = service.parse_from_image_with_log(filepath)
    except Exception as e:
        return {'error': f'Failed to parse metadata: {str(e)}'}

    results = {}
    if parsed.raw is not None:
        results['parameters'] = parsed.raw

        parsed_with_source = {}
        for key, field in parsed.fields.items():
            entry = {'value': field.value}
            if field.source != MetadataSource.UNKNOWN:
                entry['source'] = field.source
            if not field.valid:
                entry['error'] = field.error
            parsed_with_source[field.label] = entry
        results['parsed'] = parsed_with_source

        results['errors'] = parsed.errors() if parsed.has_errors() else {}

    if parsed.scheme is not None:
        results['metadata_scheme'] = parsed.scheme.value

    results['source'] = parsed.source

    if MetadataSource.PRIVATE_LOG in str(parsed.source):
        results['note'] = ('Fields were merged: private_log entries filled only when embedded metadata '
                           'is missing, empty, or failed to parse (default values are preserved).')

    log_diag = parsed.get_log_diagnostics()
    if log_diag is not None:
        results['log_match_diagnostics'] = log_diag.to_display_dict()

    return results


def trigger_metadata_import(filepath, state_is_generating, inpaint_mode):
    service = get_metadata_service()
    if filepath is None:
        print('No image provided for metadata import!')
        return service.load_parameters({}, state_is_generating, inpaint_mode)
    try:
        parsed = service.parse_from_image_with_log(filepath)
    except Exception as e:
        print(f'Could not parse metadata from image: {e}')
        parsed_parameters = {}
    else:
        if parsed.raw is None:
            print('Could not find metadata in the image!')
            parsed_parameters = {}
        else:
            parsed_parameters = parsed.to_dict()

    return service.load_parameters(parsed_parameters, state_is_generating, inpaint_mode)


def trigger_compare(left_path, right_path):
    service = get_metadata_service()
    if not left_path or not right_path:
        return 'Please upload both images.', {}
    try:
        base_parsed = service.parse_from_image_with_log(left_path)
        target_parsed = service.parse_from_image_with_log(right_path)
    except Exception as e:
        return f'Failed to parse metadata: {str(e)}', {}

    diff = service.diff(base_parsed, target_parsed)

    source_color_map = {
        MetadataSource.EMBEDDED: '#3b82f6',
        MetadataSource.PRIVATE_LOG: '#f59e0b',
        f'{MetadataSource.EMBEDDED}+{MetadataSource.PRIVATE_LOG}': '#8b5cf6',
        f'{MetadataSource.PRIVATE_LOG}+{MetadataSource.EMBEDDED}': '#8b5cf6',
        MetadataSource.PRESET: '#10b981',
    }

    def source_badge(src):
        color = source_color_map.get(src, '#666')
        display = str(src).replace('_', ' ')
        return f'<span style="font-size:10px;color:white;background:{color};padding:1px 6px;border-radius:3px;">{display}</span>'

    def render_diag_box(label, parsed):
        diag = parsed.get_log_diagnostics()
        if diag is None:
            return f'<div style="background:#f0f0f0;padding:6px 10px;border-radius:4px;"><b>{label}</b>: (no diagnostics)</div>'
        status_color = '#d4edda' if diag.available else ('#fff3cd' if diag.log_found else '#f8d7da')
        status_text = 'MATCHED' if diag.available else ('UNAVAILABLE' if diag.log_found else 'NO LOG')
        parts = [
            f'<div style="background:{status_color};padding:6px 10px;border-radius:4px;border:1px solid #ccc;">',
            f'<b>{label}</b> — <span style="font-family:monospace;">{status_text}</span>'
        ]
        if diag.log_found:
            parts.append(f' | log.html: found, entries scanned: <b>{diag.candidate_count}</b>')
        else:
            parts.append(f' | log.html: <b>not found</b> next to image')
        if diag.matched_by and diag.matched_by != 'none':
            parts.append(f' | matched_by: <code>{diag.matched_by}</code>')
        if diag.matched_entry_id:
            parts.append(f' | entry: <code>{diag.matched_entry_id}</code>')
        if diag.filled_field_count:
            parts.append(f' | filled: <b>{diag.filled_field_count}</b> fields')
        if diag.note:
            parts.append(f'<br/><small style="color:#555;">ℹ {diag.note}</small>')
        if diag.rejected_reasons:
            parts.append(f'<br/><details><summary>Rejected candidates ({len(diag.rejected_reasons)})</summary><ul style="margin:4px 0;">')
            for r in diag.rejected_reasons[:10]:
                parts.append(f'<li style="font-size:11px;color:#666;">{r}</li>')
            if len(diag.rejected_reasons) > 10:
                parts.append(f'<li style="font-size:11px;color:#666;">… and {len(diag.rejected_reasons) - 10} more</li>')
            parts.append('</ul></details>')
        parts.append('</div>')
        return ''.join(parts)

    summary_lines = [
        f'<h3 style="color:#444;">Difference Summary</h3>',
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">',
        render_diag_box('BASE (left)', base_parsed),
        render_diag_box('TARGET (right)', target_parsed),
        '</div>',
        f'<p>Total compared fields: <b>{diff.same_count + diff.diff_count}</b> | '
        f'Same: <b style="color:green;">{diff.same_count}</b> | '
        f'Different: <b style="color:red;">{diff.diff_count}</b></p>',
        f'<p style="font-size:12px;color:#666;">'
        f'Legend: {source_badge(MetadataSource.EMBEDDED)} = Embedded in image | '
        f'{source_badge(MetadataSource.PRIVATE_LOG)} = Private log (log.html) | '
        f'{source_badge(f"{MetadataSource.EMBEDDED}+{MetadataSource.PRIVATE_LOG}")} = Merged</p>',
        '<hr/>'
    ]

    table_rows = ''
    for item in diff.items:
        bg = '#e8ffe8' if item.same else '#ffe8e8'
        status = '<span style="color:green;">SAME</span>' if item.same else '<span style="color:red;">DIFF</span>'
        left_val = str(item.left_value).replace('<', '&lt;').replace('>', '&gt;')
        right_val = str(item.right_value).replace('<', '&lt;').replace('>', '&gt;')
        left_src = source_badge(item.left_source)
        right_src = source_badge(item.right_source)
        table_rows += (
            f'<tr style="background:{bg};">'
            f'<td><b>{item.label}</b><br/><small style="color:#888;">{item.key}</small></td>'
            f'<td><small>{left_val[:80]}</small><br/>{left_src}</td>'
            f'<td><small>{right_val[:80]}</small><br/>{right_src}</td>'
            f'<td>{status}</td>'
            f'</tr>'
        )

    if table_rows:
        summary_lines.append(
            '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
            '<thead><tr style="background:#ddd;">'
            '<th style="width:22%;">Field</th>'
            '<th style="width:29%;">Base (with source)</th>'
            '<th style="width:29%;">Target (with source)</th>'
            '<th style="width:20%;">Status</th>'
            '</tr></thead>'
            f'<tbody>{table_rows}</tbody></table>'
        )
    else:
        summary_lines.append('<p>No fields to compare.</p>')

    summary_html = '\n'.join(summary_lines)

    diff_display = {
        'summary': {
            'total_fields': diff.same_count + diff.diff_count,
            'same': diff.same_count,
            'different': diff.diff_count
        },
        'base_log_diagnostics': base_parsed.get_log_diagnostics().to_display_dict() if base_parsed.get_log_diagnostics() else None,
        'target_log_diagnostics': target_parsed.get_log_diagnostics().to_display_dict() if target_parsed.get_log_diagnostics() else None,
        'differences': [
            {
                'field': item.label,
                'key': item.key,
                'base_value': item.left_value,
                'base_source': item.left_source,
                'target_value': item.right_value,
                'target_source': item.right_source,
                'same': item.same
            }
            for item in diff.items
        ]
    }

    return summary_html, diff_display


def trigger_compare_fill(left_path, right_path, is_generating, fill_mode, inpaint_mode):
    service = get_metadata_service()
    if not left_path or not right_path:
        return service.load_parameters({}, is_generating, inpaint_mode)

    diff, fill_params = service.diff_and_get_fill_parameters(
        base=left_path,
        target=right_path,
        is_generating=is_generating,
        inpaint_mode=inpaint_mode,
        fill_mode=fill_mode
    )
    return fill_params


def parse_meta(raw_prompt_txt, is_generating):
    loaded_json = None
    if is_json(raw_prompt_txt):
        loaded_json = json.loads(raw_prompt_txt)

    if loaded_json is None:
        if is_generating:
            return gr.update(), gr.update(), gr.update()
        else:
            return gr.update(), gr.update(visible=True), gr.update(visible=False)

    return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)


def load_parameters_from_prompt(raw_metadata, state_is_generating, inpaint_mode):
    service = get_metadata_service()
    return service.load_parameters(raw_metadata, state_is_generating, inpaint_mode)


def preset_selection_change(preset, is_generating, inpaint_mode):
    preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
    service = get_metadata_service()
    preset_prepared = service.parse_from_preset(preset_content)

    default_model = preset_prepared.get('base_model')
    previous_default_models = preset_prepared.get('previous_default_models', [])
    checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
    embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
    lora_downloads = preset_prepared.get('lora_downloads', {})
    vae_downloads = preset_prepared.get('vae_downloads', {})

    preset_prepared['base_model'], preset_prepared['checkpoint_downloads'] = launch.download_models(
        default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads,
        vae_downloads)

    if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
        del preset_prepared['prompt']

    return service.load_parameters(json.dumps(preset_prepared), is_generating, inpaint_mode)


def inpaint_engine_state_change(inpaint_engine_version, *args):
    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    result = []
    for inpaint_mode in args:
        if inpaint_mode != modules.flags.inpaint_option_detail:
            result.append(gr.update(value=inpaint_engine_version))
        else:
            result.append(gr.update())

    return result


def preset_selection_update_details(preset_name):
    return format_preset_details_html(preset_name)


def dev_mode_checked(r):
    return gr.update(visible=r)


def refresh_files_clicked():
    modules.config.update_files()
    results = [gr.update(choices=modules.config.model_filenames)]
    results += [gr.update(choices=['None'] + modules.config.model_filenames)]
    results += [gr.update(choices=[flags.default_vae] + modules.config.vae_filenames)]
    if not args_manager.args.disable_preset_selection:
        results += [gr.update(choices=modules.config.available_presets)]
    for i in range(modules.config.default_max_lora_number):
        results += [gr.update(interactive=True),
                    gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
    return results


def save_current_as_preset(new_name, *ui_args):
    if not new_name or new_name.strip() == '':
        return gr.update(), gr.update(), '⚠️  Error: Preset name cannot be empty'
    try:
        preset_data = build_preset_data_from_ui(*ui_args)
        success, result = modules.config.save_user_preset(new_name.strip(), preset_data)
        if success:
            modules.config.update_files()
            return (
                gr.update(choices=modules.config.available_presets, value=result),
                format_preset_details_html(result),
                f'✅  Preset saved successfully: {result}'
            )
        else:
            return gr.update(), gr.update(), f'⚠️  Error: {result}'
    except Exception as e:
        return gr.update(), gr.update(), f'⚠️  Error saving preset: {str(e)}'


def duplicate_current_preset(current_preset, new_name):
    if not new_name or new_name.strip() == '':
        return gr.update(), gr.update(), '⚠️  Error: New preset name is required for duplication'
    if current_preset == 'initial':
        return gr.update(), gr.update(), '⚠️  Error: Cannot duplicate "initial", use "Save Current as Preset" instead'
    success, result = modules.config.duplicate_user_preset(current_preset, new_name.strip())
    if success:
        modules.config.update_files()
        return (
            gr.update(choices=modules.config.available_presets, value=result),
            format_preset_details_html(result),
            f'✅  Preset duplicated: {current_preset} → {result}'
        )
    else:
        return gr.update(), gr.update(), f'⚠️  Error: {result}'


def rename_current_preset(current_preset, new_name):
    if not modules.config.is_user_preset(current_preset):
        return gr.update(), gr.update(), '⚠️  Error: Can only rename user presets (marked with [User] prefix)'
    if not new_name or new_name.strip() == '':
        return gr.update(), gr.update(), '⚠️  Error: New preset name cannot be empty'
    success, result = modules.config.rename_user_preset(current_preset, new_name.strip())
    if success:
        modules.config.update_files()
        return (
            gr.update(choices=modules.config.available_presets, value=result),
            format_preset_details_html(result),
            f'✅  Preset renamed: {current_preset} → {result}'
        )
    else:
        return gr.update(), gr.update(), f'⚠️  Error: {result}'


def delete_current_preset(current_preset):
    if not modules.config.is_user_preset(current_preset):
        return gr.update(), gr.update(), '⚠️  Error: Can only delete user presets (marked with [User] prefix)'
    success, result = modules.config.delete_user_preset(current_preset)
    if success:
        modules.config.update_files()
        return (
            gr.update(choices=modules.config.available_presets, value='initial'),
            format_preset_details_html('initial'),
            f'✅  Preset deleted: {current_preset}'
        )
    else:
        return gr.update(), gr.update(), f'⚠️  Error: {result}'
