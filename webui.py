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
from modules.ui.protocol import build_ctrls, build_load_data_outputs


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
            progress_window = preview_comps.progress_window
            progress_gallery = preview_comps.progress_gallery
            progress_html = preview_comps.progress_html
            gallery = preview_comps.gallery

            prompt_buttons = create_prompt_and_buttons(shared.gradio_root)
            prompt = prompt_buttons.prompt
            generate_button = prompt_buttons.generate_button
            reset_button = prompt_buttons.reset_button
            load_parameter_button = prompt_buttons.load_parameter_button
            skip_button = prompt_buttons.skip_button
            stop_button = prompt_buttons.stop_button

            bind_prompt_control_events(prompt_buttons, currentTask)

            top_checkboxes = create_top_checkboxes()
            input_image_checkbox = top_checkboxes.input_image_checkbox
            enhance_checkbox = top_checkboxes.enhance_checkbox
            advanced_checkbox = top_checkboxes.advanced_checkbox

            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                image_input = create_image_input_tabs(shared.gradio_root)

                uov_tab = image_input.uov_tab
                uov_input_image = image_input.uov_input_image
                uov_method = image_input.uov_method

                ip_tab = image_input.ip_tab

                inpaint_tab = image_input.inpaint_tab
                inpaint = image_input.inpaint
                inpaint_input_image = inpaint.inpaint_input_image
                inpaint_advanced_masking_checkbox = inpaint.inpaint_advanced_masking_checkbox
                inpaint_mode = inpaint.inpaint_mode
                inpaint_additional_prompt = inpaint.inpaint_additional_prompt
                outpaint_selections = inpaint.outpaint_selections
                example_inpaint_prompts = inpaint.example_inpaint_prompts
                inpaint_mask_generation_col = inpaint.inpaint_mask_generation_col
                inpaint_mask_image = inpaint.inpaint_mask_image
                invert_mask_checkbox = inpaint.invert_mask_checkbox
                inpaint_mask_model = inpaint.inpaint_mask_model
                generate_mask_button = inpaint.generate_mask_button
                inpaint_mask_cloth_category = inpaint.inpaint_mask_cloth_category
                inpaint_mask_dino_prompt_text = inpaint.inpaint_mask_dino_prompt_text
                example_inpaint_mask_dino_prompt_text = inpaint.example_inpaint_mask_dino_prompt_text
                inpaint_mask_sam_model = inpaint.inpaint_mask_sam_model
                inpaint_mask_box_threshold = inpaint.inpaint_mask_box_threshold
                inpaint_mask_text_threshold = inpaint.inpaint_mask_text_threshold
                inpaint_mask_sam_max_detections = inpaint.inpaint_mask_sam_max_detections

                describe_tab = image_input.describe_tab
                describe = image_input.describe
                describe_input_image = describe.describe_input_image
                describe_methods = describe.describe_methods
                describe_apply_styles = describe.describe_apply_styles
                describe_btn = describe.describe_btn
                describe_image_size = describe.describe_image_size

                enhance_tab = image_input.enhance_tab
                enhance_input_image = image_input.enhance_input_image

                metadata_tab = create_metadata_tab()
                compare_tab = create_compare_tab()

            with gr.Row(visible=modules.config.default_enhance_checkbox) as enhance_input_panel:
                enhance_panel = create_enhance_panel(shared.gradio_root, inpaint_engine_state)
                enhance_ctrls = enhance_panel.enhance_ctrls
                enhance_inpaint_mode_ctrls = enhance_panel.enhance_inpaint_mode_ctrls
                enhance_inpaint_engine_ctrls = enhance_panel.enhance_inpaint_engine_ctrls
                enhance_inpaint_update_ctrls = enhance_panel.enhance_inpaint_update_ctrls
                enhance_uov_method = enhance_panel.enhance_uov_method
                enhance_uov_processing_order = enhance_panel.enhance_uov_processing_order
                enhance_uov_prompt_type = enhance_panel.enhance_uov_prompt_type

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox) as advanced_column:
            output_format_ref = [None]
            advanced = create_all_advanced_tabs(shared.gradio_root, output_format_ref)

            if not args_manager.args.disable_preset_selection:
                preset_selection = advanced.settings.preset_selection
                preset_details_html = advanced.settings.preset_details_html
                new_preset_name_input = advanced.settings.new_preset_name_input
                save_preset_btn = advanced.settings.save_preset_btn
                duplicate_preset_btn = advanced.settings.duplicate_preset_btn
                rename_preset_btn = advanced.settings.rename_preset_btn
                delete_preset_btn = advanced.settings.delete_preset_btn
                preset_operation_msg = advanced.settings.preset_operation_msg

            performance_selection = advanced.settings.performance_selection
            aspect_ratios_selection = advanced.settings.aspect_ratios_selection
            image_number = advanced.settings.image_number
            output_format = advanced.settings.output_format
            negative_prompt = advanced.settings.negative_prompt
            seed_random = advanced.settings.seed_random
            image_seed = advanced.settings.image_seed
            history_link = advanced.settings.history_link

            style_selections = advanced.styles.style_selections
            style_search_bar = advanced.styles.style_search_bar
            gradio_receiver_style_selections = advanced.styles.gradio_receiver_style_selections

            base_model = advanced.models.base_model
            refiner_model = advanced.models.refiner_model
            refiner_switch = advanced.models.refiner_switch
            lora_ctrls = advanced.models.lora_ctrls
            refresh_files = advanced.models.refresh_files

            guidance_scale = advanced.advanced.guidance_scale
            sharpness = advanced.advanced.sharpness
            dev_mode = advanced.advanced.dev_mode
            dev_tools = advanced.advanced.dev_tools

            debug_tools = advanced.advanced.debug_tools
            adm_scaler_positive = debug_tools.adm_scaler_positive
            adm_scaler_negative = debug_tools.adm_scaler_negative
            adm_scaler_end = debug_tools.adm_scaler_end
            refiner_swap_method = debug_tools.refiner_swap_method
            adaptive_cfg = debug_tools.adaptive_cfg
            clip_skip = debug_tools.clip_skip
            sampler_name = debug_tools.sampler_name
            scheduler_name = debug_tools.scheduler_name
            vae_name = debug_tools.vae_name
            generate_image_grid = debug_tools.generate_image_grid
            overwrite_step = debug_tools.overwrite_step
            overwrite_switch = debug_tools.overwrite_switch
            overwrite_width = debug_tools.overwrite_width
            overwrite_height = debug_tools.overwrite_height
            overwrite_vary_strength = debug_tools.overwrite_vary_strength
            overwrite_upscale_strength = debug_tools.overwrite_upscale_strength
            disable_preview = debug_tools.disable_preview
            disable_intermediate_results = debug_tools.disable_intermediate_results
            disable_seed_increment = debug_tools.disable_seed_increment
            read_wildcards_in_order = debug_tools.read_wildcards_in_order
            black_out_nsfw = debug_tools.black_out_nsfw

            if not args_manager.args.disable_image_log:
                save_final_enhanced_image_only = debug_tools.save_final_enhanced_image_only

            if not args_manager.args.disable_metadata:
                save_metadata_to_images = debug_tools.save_metadata_to_images
                metadata_scheme = debug_tools.metadata_scheme

            control = advanced.advanced.control
            debugging_cn_preprocessor = control.debugging_cn_preprocessor
            skipping_cn_preprocessor = control.skipping_cn_preprocessor
            mixing_image_prompt_and_vary_upscale = control.mixing_image_prompt_and_vary_upscale
            mixing_image_prompt_and_inpaint = control.mixing_image_prompt_and_inpaint
            controlnet_softness = control.controlnet_softness
            canny_low_threshold = control.canny_low_threshold
            canny_high_threshold = control.canny_high_threshold

            inpaint_advanced = advanced.advanced.inpaint_advanced
            debugging_inpaint_preprocessor = inpaint_advanced.debugging_inpaint_preprocessor
            debugging_enhance_masks_checkbox = inpaint_advanced.debugging_enhance_masks_checkbox
            debugging_dino = inpaint_advanced.debugging_dino
            inpaint_disable_initial_latent = inpaint_advanced.inpaint_disable_initial_latent
            inpaint_engine = inpaint_advanced.inpaint_engine
            inpaint_strength = inpaint_advanced.inpaint_strength
            inpaint_respective_field = inpaint_advanced.inpaint_respective_field
            inpaint_erode_or_dilate = inpaint_advanced.inpaint_erode_or_dilate
            dino_erode_or_dilate = inpaint_advanced.dino_erode_or_dilate
            inpaint_mask_color = inpaint_advanced.inpaint_mask_color

            freeu = advanced.advanced.freeu
            freeu_enabled = freeu.freeu_enabled
            freeu_b1 = freeu.freeu_b1
            freeu_b2 = freeu.freeu_b2
            freeu_s1 = freeu.freeu_s1
            freeu_s2 = freeu.freeu_s2
            freeu_ctrls = freeu.freeu_ctrls

            inpaint_ctrls = [
                debugging_inpaint_preprocessor, inpaint_disable_initial_latent,
                inpaint_engine, inpaint_strength, inpaint_respective_field,
                inpaint_advanced_masking_checkbox, invert_mask_checkbox,
                inpaint_erode_or_dilate
            ]

    state_is_generating = gr.State(False)

    load_data_outputs = build_load_data_outputs(
        top_checkboxes, prompt_buttons, image_input, enhance_panel,
        advanced, inpaint_engine_state
    )

    current_tab = gr.Textbox(value='uov', visible=False)
    down_js = '(x)=>{setTimeout(()=>{window.scrollTo(0,document.body.scrollHeight);},200);return x;}'
    switch_js = '(x)=>{setTimeout(()=>{window.scrollTo(0,document.body.scrollHeight);},200);return x;}'

    uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    describe_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    enhance_tab.select(lambda: 'enhance', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    metadata_tab.metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
    enhance_checkbox.change(lambda x: gr.update(visible=x), inputs=enhance_checkbox,
                            outputs=enhance_input_panel, queue=False, show_progress=False, _js=switch_js)

    bind_image_input_events(
        image_input, inpaint_engine_state,
        inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts
    )

    bind_metadata_events(
        metadata_tab, state_is_generating, inpaint_mode,
        load_data_outputs, style_selections
    )

    bind_compare_events(
        compare_tab, state_is_generating, inpaint_mode,
        load_data_outputs, style_selections
    )

    bind_advanced_column_events(
        advanced, shared.gradio_root, inpaint_engine_state,
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

    ctrls = build_ctrls(
        currentTask, prompt_buttons, top_checkboxes, image_input,
        enhance_panel, advanced, current_tab
    )

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
