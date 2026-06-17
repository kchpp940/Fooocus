import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
from modules.metadata_service import get_metadata_service, MetadataSource
import args_manager
import copy
import launch
from extras.inpaint_mask import SAMOptions

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json

from modules.ui.generation import (
    get_task, generate_clicked, refresh_seed, update_history_link,
    generate_mask, trigger_describe, trigger_auto_describe,
    parse_meta, load_parameters_from_prompt
)
from modules.ui.prompt_area import (
    create_preview_components, create_prompt_and_buttons,
    create_top_checkboxes, bind_prompt_control_events
)
from modules.ui.image_input import (
    create_image_input_tabs, create_enhance_panel, bind_image_input_events
)
from modules.ui.metadata_compare import (
    create_metadata_tab, create_compare_tab,
    bind_metadata_events, bind_compare_events
)
from modules.ui.advanced_settings import (
    create_all_advanced_tabs, bind_advanced_column_events
)


reload_javascript()

title = f'Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title).queue()

with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')

    with gr.Row():
        with gr.Column(scale=2):
            preview_comps = create_preview_components(shared.gradio_root)
            progress_window = preview_comps['progress_window']
            progress_gallery = preview_comps['progress_gallery']
            progress_html = preview_comps['progress_html']
            gallery = preview_comps['gallery']

            prompt_comps = create_prompt_and_buttons(shared.gradio_root)
            prompt = prompt_comps['prompt']
            generate_button = prompt_comps['generate_button']
            reset_button = prompt_comps['reset_button']
            load_parameter_button = prompt_comps['load_parameter_button']
            skip_button = prompt_comps['skip_button']
            stop_button = prompt_comps['stop_button']

            bind_prompt_control_events(prompt_comps, currentTask)

            top_checkboxes = create_top_checkboxes()
            input_image_checkbox = top_checkboxes['input_image_checkbox']
            enhance_checkbox = top_checkboxes['enhance_checkbox']
            advanced_checkbox = top_checkboxes['advanced_checkbox']

            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                image_input_comps = create_image_input_tabs(shared.gradio_root)

                uov_tab = image_input_comps['uov_tab']
                uov_input_image = image_input_comps['uov_input_image']
                uov_method = image_input_comps['uov_method']

                ip_tab = image_input_comps['ip_tab']
                ip_ctrls = image_input_comps['ip_ctrls']
                ip_images = image_input_comps['ip_images']
                ip_types = image_input_comps['ip_types']
                ip_stops = image_input_comps['ip_stops']
                ip_weights = image_input_comps['ip_weights']
                ip_ad_cols = image_input_comps['ip_ad_cols']
                ip_advanced = image_input_comps['ip_advanced']

                inpaint_tab = image_input_comps['inpaint_tab']
                inpaint_input_image = image_input_comps['inpaint_input_image']
                inpaint_advanced_masking_checkbox = image_input_comps['inpaint_advanced_masking_checkbox']
                inpaint_mode = image_input_comps['inpaint_mode']
                inpaint_additional_prompt = image_input_comps['inpaint_additional_prompt']
                outpaint_selections = image_input_comps['outpaint_selections']
                example_inpaint_prompts = image_input_comps['example_inpaint_prompts']
                inpaint_mask_generation_col = image_input_comps['inpaint_mask_generation_col']
                inpaint_mask_image = image_input_comps['inpaint_mask_image']
                invert_mask_checkbox = image_input_comps['invert_mask_checkbox']
                inpaint_mask_model = image_input_comps['inpaint_mask_model']
                inpaint_mask_cloth_category = image_input_comps['inpaint_mask_cloth_category']
                inpaint_mask_dino_prompt_text = image_input_comps['inpaint_mask_dino_prompt_text']
                example_inpaint_mask_dino_prompt_text = image_input_comps['example_inpaint_mask_dino_prompt_text']
                inpaint_mask_advanced_options = image_input_comps['inpaint_mask_advanced_options']
                inpaint_mask_sam_model = image_input_comps['inpaint_mask_sam_model']
                inpaint_mask_box_threshold = image_input_comps['inpaint_mask_box_threshold']
                inpaint_mask_text_threshold = image_input_comps['inpaint_mask_text_threshold']
                inpaint_mask_sam_max_detections = image_input_comps['inpaint_mask_sam_max_detections']
                generate_mask_button = image_input_comps['generate_mask_button']

                describe_tab = image_input_comps['describe_tab']
                describe_input_image = image_input_comps['describe_input_image']
                describe_methods = image_input_comps['describe_methods']
                describe_apply_styles = image_input_comps['describe_apply_styles']
                describe_btn = image_input_comps['describe_btn']
                describe_image_size = image_input_comps['describe_image_size']

                enhance_tab = image_input_comps['enhance_tab']
                enhance_input_image = image_input_comps['enhance_input_image']

                metadata_comps = create_metadata_tab()
                metadata_tab = metadata_comps['metadata_tab']
                metadata_input_image = metadata_comps['metadata_input_image']
                metadata_json = metadata_comps['metadata_json']
                metadata_import_button = metadata_comps['metadata_import_button']

                compare_comps = create_compare_tab()
                compare_tab = compare_comps['compare_tab']
                compare_image_left = compare_comps['compare_image_left']
                compare_image_right = compare_comps['compare_image_right']
                compare_run_button = compare_comps['compare_run_button']
                compare_fill_diff_button = compare_comps['compare_fill_diff_button']
                compare_fill_all_button = compare_comps['compare_fill_all_button']
                compare_summary_html = compare_comps['compare_summary_html']
                compare_diff_json = compare_comps['compare_diff_json']

            with gr.Row(visible=modules.config.default_enhance_checkbox) as enhance_input_panel:
                enhance_panel_comps = create_enhance_panel(shared.gradio_root, inpaint_engine_state)
                enhance_ctrls = enhance_panel_comps['enhance_ctrls']
                enhance_inpaint_mode_ctrls = enhance_panel_comps['enhance_inpaint_mode_ctrls']
                enhance_inpaint_engine_ctrls = enhance_panel_comps['enhance_inpaint_engine_ctrls']
                enhance_inpaint_update_ctrls = enhance_panel_comps['enhance_inpaint_update_ctrls']
                enhance_uov_method = enhance_panel_comps['enhance_uov_method']
                enhance_uov_processing_order = enhance_panel_comps['enhance_uov_processing_order']
                enhance_uov_prompt_type = enhance_panel_comps['enhance_uov_prompt_type']
                debugging_dino = enhance_panel_comps['debugging_dino']
                dino_erode_or_dilate = enhance_panel_comps['dino_erode_or_dilate']
                debugging_enhance_masks_checkbox = enhance_panel_comps['debugging_enhance_masks_checkbox']

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox) as advanced_column:
            output_format_ref = [None]
            advanced_comps = create_all_advanced_tabs(shared.gradio_root, output_format_ref)

            if not args_manager.args.disable_preset_selection:
                preset_selection = advanced_comps['preset_selection']
                preset_details_html = advanced_comps['preset_details_html']
                new_preset_name_input = advanced_comps['new_preset_name_input']
                save_preset_btn = advanced_comps['save_preset_btn']
                duplicate_preset_btn = advanced_comps['duplicate_preset_btn']
                rename_preset_btn = advanced_comps['rename_preset_btn']
                delete_preset_btn = advanced_comps['delete_preset_btn']
                preset_operation_msg = advanced_comps['preset_operation_msg']

            performance_selection = advanced_comps['performance_selection']
            aspect_ratios_selection = advanced_comps['aspect_ratios_selection']
            image_number = advanced_comps['image_number']
            output_format = advanced_comps['output_format']
            negative_prompt = advanced_comps['negative_prompt']
            seed_random = advanced_comps['seed_random']
            image_seed = advanced_comps['image_seed']
            history_link = advanced_comps['history_link']

            style_selections = advanced_comps['style_selections']
            style_search_bar = advanced_comps['style_search_bar']
            gradio_receiver_style_selections = advanced_comps['gradio_receiver_style_selections']

            base_model = advanced_comps['base_model']
            refiner_model = advanced_comps['refiner_model']
            refiner_switch = advanced_comps['refiner_switch']
            lora_ctrls = advanced_comps['lora_ctrls']
            refresh_files = advanced_comps['refresh_files']

            guidance_scale = advanced_comps['guidance_scale']
            sharpness = advanced_comps['sharpness']
            dev_mode = advanced_comps['dev_mode']
            dev_tools = advanced_comps['dev_tools']

            adm_scaler_positive = advanced_comps['adm_scaler_positive']
            adm_scaler_negative = advanced_comps['adm_scaler_negative']
            adm_scaler_end = advanced_comps['adm_scaler_end']
            refiner_swap_method = advanced_comps['refiner_swap_method']
            adaptive_cfg = advanced_comps['adaptive_cfg']
            clip_skip = advanced_comps['clip_skip']
            sampler_name = advanced_comps['sampler_name']
            scheduler_name = advanced_comps['scheduler_name']
            vae_name = advanced_comps['vae_name']
            generate_image_grid = advanced_comps['generate_image_grid']
            overwrite_step = advanced_comps['overwrite_step']
            overwrite_switch = advanced_comps['overwrite_switch']
            overwrite_width = advanced_comps['overwrite_width']
            overwrite_height = advanced_comps['overwrite_height']
            overwrite_vary_strength = advanced_comps['overwrite_vary_strength']
            overwrite_upscale_strength = advanced_comps['overwrite_upscale_strength']
            disable_preview = advanced_comps['disable_preview']
            disable_intermediate_results = advanced_comps['disable_intermediate_results']
            disable_seed_increment = advanced_comps['disable_seed_increment']
            read_wildcards_in_order = advanced_comps['read_wildcards_in_order']
            black_out_nsfw = advanced_comps['black_out_nsfw']

            if not args_manager.args.disable_image_log:
                save_final_enhanced_image_only = advanced_comps['save_final_enhanced_image_only']

            if not args_manager.args.disable_metadata:
                save_metadata_to_images = advanced_comps['save_metadata_to_images']
                metadata_scheme = advanced_comps['metadata_scheme']

            debugging_cn_preprocessor = advanced_comps['debugging_cn_preprocessor']
            skipping_cn_preprocessor = advanced_comps['skipping_cn_preprocessor']
            mixing_image_prompt_and_vary_upscale = advanced_comps['mixing_image_prompt_and_vary_upscale']
            mixing_image_prompt_and_inpaint = advanced_comps['mixing_image_prompt_and_inpaint']
            controlnet_softness = advanced_comps['controlnet_softness']
            canny_low_threshold = advanced_comps['canny_low_threshold']
            canny_high_threshold = advanced_comps['canny_high_threshold']

            debugging_inpaint_preprocessor = advanced_comps['debugging_inpaint_preprocessor']
            debugging_enhance_masks_checkbox = advanced_comps['debugging_enhance_masks_checkbox']
            debugging_dino = advanced_comps['debugging_dino']
            inpaint_disable_initial_latent = advanced_comps['inpaint_disable_initial_latent']
            inpaint_engine = advanced_comps['inpaint_engine']
            inpaint_strength = advanced_comps['inpaint_strength']
            inpaint_respective_field = advanced_comps['inpaint_respective_field']
            inpaint_erode_or_dilate = advanced_comps['inpaint_erode_or_dilate']
            dino_erode_or_dilate = advanced_comps['dino_erode_or_dilate']
            inpaint_mask_color = advanced_comps['inpaint_mask_color']

            freeu_enabled = advanced_comps['freeu_enabled']
            freeu_b1 = advanced_comps['freeu_b1']
            freeu_b2 = advanced_comps['freeu_b2']
            freeu_s1 = advanced_comps['freeu_s1']
            freeu_s2 = advanced_comps['freeu_s2']
            freeu_ctrls = advanced_comps['freeu_ctrls']

            inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                             inpaint_strength, inpaint_respective_field,
                             inpaint_advanced_masking_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

    state_is_generating = gr.State(False)

    load_data_outputs = [advanced_checkbox, image_number, prompt, negative_prompt, style_selections,
                         performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                         overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                         adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, clip_skip,
                         base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, vae_name,
                         seed_random, image_seed, inpaint_engine, inpaint_engine_state,
                         inpaint_mode] + enhance_inpaint_mode_ctrls + [generate_button,
                         load_parameter_button] + freeu_ctrls + lora_ctrls

    current_tab = gr.Textbox(value='uov', visible=False)
    down_js = '(x)=>{setTimeout(()=>{window.scrollTo(0,document.body.scrollHeight);},200);return x;}'
    switch_js = '(x)=>{setTimeout(()=>{window.scrollTo(0,document.body.scrollHeight);},200);return x;}'

    uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    describe_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    enhance_tab.select(lambda: 'enhance', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    enhance_checkbox.change(lambda x: gr.update(visible=x), inputs=enhance_checkbox,
                            outputs=enhance_input_panel, queue=False, show_progress=False, _js=switch_js)

    bind_image_input_events(
        image_input_comps, inpaint_engine_state,
        inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts
    )

    bind_metadata_events(
        metadata_comps, state_is_generating, inpaint_mode,
        load_data_outputs, style_selections
    )

    bind_compare_events(
        compare_comps, state_is_generating, inpaint_mode,
        load_data_outputs, style_selections
    )

    bind_advanced_column_events(
        advanced_comps, shared.gradio_root, inpaint_engine_state,
        inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
        inpaint_input_image, inpaint_mask_image, inpaint_mask_generation_col,
        inpaint_advanced_masking_checkbox, invert_mask_checkbox,
        enhance_inpaint_mode_ctrls, enhance_inpaint_update_ctrls, enhance_inpaint_engine_ctrls,
        state_is_generating, inpaint_mode, load_data_outputs,
        style_selections, output_format_ref
    )

    advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                             queue=False, show_progress=False) \
        .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)

    performance_selection.change(lambda x: [gr.update(interactive=not flags.Performance.has_restricted_features(x))] * 11 +
                                           [gr.update(visible=not flags.Performance.has_restricted_features(x))] * 1 +
                                           [gr.update(value=flags.Performance.has_restricted_features(x))] * 1,
                                 inputs=performance_selection,
                                 outputs=[
                                     guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                     adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                     scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt, disable_intermediate_results
                                 ], queue=False, show_progress=False)

    generate_mask_button.click(fn=generate_mask,
                               inputs=[inpaint_input_image, inpaint_mask_model, inpaint_mask_cloth_category,
                                       inpaint_mask_dino_prompt_text, inpaint_mask_sam_model,
                                       inpaint_mask_box_threshold, inpaint_mask_text_threshold,
                                       inpaint_mask_sam_max_detections, dino_erode_or_dilate, debugging_dino],
                               outputs=inpaint_mask_image, show_progress=True, queue=True)

    ctrls = [currentTask, generate_image_grid]
    ctrls += [
        prompt, negative_prompt, style_selections,
        performance_selection, aspect_ratios_selection, image_number, output_format, image_seed,
        read_wildcards_in_order, sharpness, guidance_scale
    ]

    ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
    ctrls += [input_image_checkbox, current_tab]
    ctrls += [uov_method, uov_input_image]
    ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
    ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment, black_out_nsfw]
    ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip]
    ctrls += [sampler_name, scheduler_name, vae_name]
    ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
    ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
    ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
    ctrls += [refiner_swap_method, controlnet_softness]
    ctrls += freeu_ctrls
    ctrls += inpaint_ctrls

    if not args_manager.args.disable_image_log:
        ctrls += [save_final_enhanced_image_only]

    if not args_manager.args.disable_metadata:
        ctrls += [save_metadata_to_images, metadata_scheme]

    ctrls += ip_ctrls
    ctrls += [debugging_dino, dino_erode_or_dilate, debugging_enhance_masks_checkbox,
              enhance_input_image, enhance_checkbox, enhance_uov_method, enhance_uov_processing_order,
              enhance_uov_prompt_type]
    ctrls += enhance_ctrls

    prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

    load_parameter_button.click(load_parameters_from_prompt, inputs=[prompt, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=False)

    generate_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True),
                          outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating]) \
        .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
        .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
        .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
        .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
              outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
        .then(fn=update_history_link, outputs=history_link) \
        .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')

    reset_button.click(lambda: [worker.AsyncTask(args=[]), False, gr.update(visible=True, interactive=True)] +
                               [gr.update(visible=False)] * 6 +
                               [gr.update(visible=True, value=[])],
                       outputs=[currentTask, state_is_generating, generate_button,
                                reset_button, stop_button, skip_button,
                                progress_html, progress_window, progress_gallery, gallery],
                       queue=False)

    for notification_file in ['notification.ogg', 'notification.mp3']:
        if os.path.exists(notification_file):
            gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
            break

    describe_btn.click(trigger_describe, inputs=[describe_methods, describe_input_image, describe_apply_styles],
                       outputs=[prompt, style_selections], show_progress=True, queue=True) \
        .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
        .then(lambda: None, _js='()=>{refresh_style_localization();}')

    if args_manager.args.enable_auto_describe_image:
        uov_input_image.upload(trigger_auto_describe, inputs=[describe_methods, uov_input_image, prompt, describe_apply_styles],
                               outputs=[prompt, style_selections], show_progress=True, queue=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, _js='()=>{refresh_style_localization();}')

        enhance_input_image.upload(lambda: gr.update(value=True), outputs=enhance_checkbox, queue=False, show_progress=False) \
            .then(trigger_auto_describe, inputs=[describe_methods, enhance_input_image, prompt, describe_apply_styles],
                  outputs=[prompt, style_selections], show_progress=True, queue=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, _js='()=>{refresh_style_localization();}')


if auth_enabled():
    shared.gradio_root.queue().launch(
        server_name=args_manager.args.host,
        server_port=args_manager.args.port,
        auth=check_auth,
        inbrowser=args_manager.args.inbrowser,
        share=args_manager.args.share
    )
else:
    shared.gradio_root.queue().launch(
        server_name=args_manager.args.host,
        server_port=args_manager.args.port,
        inbrowser=args_manager.args.inbrowser,
        share=args_manager.args.share
    )
