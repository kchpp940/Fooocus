import gradio as gr

import modules.config
import modules.util as util
import modules.shared as shared
import args_manager

from modules.ui.generation import (
    get_task, generate_clicked, sort_enhance_images, inpaint_mode_change,
    stop_clicked, skip_clicked, refresh_seed, random_checked,
    trigger_describe, generate_mask
)
from modules.ui.prompt_area import create_preview_gallery, create_prompt_and_buttons, create_top_checkboxes
from modules.ui.image_input import create_image_input_tabs, create_enhance_input_panel
from modules.ui.advanced_settings import create_all_advanced_tabs, bind_advanced_column_events
from modules.ui.metadata_compare import bind_metadata_events


shared.gradio_root = gr.Blocks(title='Fooocus', css=util.css).queue()

with shared.gradio_root:
    currentTask = gr.State(None)
    inpaint_engine_state = gr.State('empty')

    with gr.Row():
        with gr.Column(scale=2):
            preview_gallery, preview_html = create_preview_gallery(shared.gradio_root)

            prompt, generate_button, stop_button, skip_button = create_prompt_and_buttons(shared.gradio_root)

            advanced_checkbox, enhance_checkbox, input_image_checkbox = create_top_checkboxes()

            with gr.Column(visible=modules.config.default_image_checkbox) as image_input_panel:
                image_components = create_image_input_tabs(shared.gradio_root, inpaint_engine_state)

                uov_input_image = image_components['uov_input_image']
                uov_method = image_components['uov_method']

                ip_ctrls = image_components['ip_ctrls']
                ip_ad_cols = image_components['ip_ad_cols']
                ip_advanced = image_components['ip_advanced']

                inpaint_input_image = image_components['inpaint_input_image']
                inpaint_advanced_masking_checkbox = image_components['inpaint_advanced_masking_checkbox']
                inpaint_mode = image_components['inpaint_mode']
                inpaint_additional_prompt = image_components['inpaint_additional_prompt']
                outpaint_selections = image_components['outpaint_selections']
                example_inpaint_prompts = image_components['example_inpaint_prompts']
                inpaint_mask_generation_col = image_components['inpaint_mask_generation_col']
                inpaint_mask_image = image_components['inpaint_mask_image']
                invert_mask_checkbox = image_components['invert_mask_checkbox']
                inpaint_mask_model = image_components['inpaint_mask_model']
                inpaint_mask_cloth_category = image_components['inpaint_mask_cloth_category']
                inpaint_mask_dino_prompt_text = image_components['inpaint_mask_dino_prompt_text']
                example_inpaint_mask_dino_prompt_text = image_components['example_inpaint_mask_dino_prompt_text']
                inpaint_mask_advanced_options = image_components['inpaint_mask_advanced_options']
                inpaint_mask_sam_model = image_components['inpaint_mask_sam_model']
                inpaint_mask_box_threshold = image_components['inpaint_mask_box_threshold']
                inpaint_mask_text_threshold = image_components['inpaint_mask_text_threshold']
                inpaint_mask_sam_max_detections = image_components['inpaint_mask_sam_max_detections']
                generate_mask_button = image_components['generate_mask_button']

                describe_input_image = image_components['describe_input_image']
                describe_methods = image_components['describe_methods']
                describe_apply_styles = image_components['describe_apply_styles']
                describe_btn = image_components['describe_btn']
                describe_image_size = image_components['describe_image_size']

                enhance_input_image = image_components['enhance_input_image']

                metadata_input_image = image_components['metadata_input_image']
                metadata_json = image_components['metadata_json']
                metadata_import_button = image_components['metadata_import_button']

            with gr.Column(visible=modules.config.default_enhance_checkbox) as enhance_panel:
                enhance_components = create_enhance_input_panel(
                    inpaint_engine_state, inpaint_additional_prompt,
                    outpaint_selections, example_inpaint_prompts
                )
                enhance_ctrls = enhance_components['enhance_ctrls']
                enhance_uov_method = enhance_components['enhance_uov_method']
                enhance_uov_processing_order = enhance_components['enhance_uov_processing_order']
                enhance_uov_prompt_type = enhance_components['enhance_uov_prompt_type']
                enhance_inpaint_mode_ctrls = enhance_components['enhance_inpaint_mode_ctrls']
                enhance_inpaint_engine_ctrls = enhance_components['enhance_inpaint_engine_ctrls']
                enhance_inpaint_update_ctrls = enhance_components['enhance_inpaint_update_ctrls']

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox) as advanced_column:
            output_format_ref = [None]
            advanced_components = create_all_advanced_tabs(shared.gradio_root, output_format_ref)

            performance_selection = advanced_components['performance_selection']
            aspect_ratios_selection = advanced_components['aspect_ratios_selection']
            image_number = advanced_components['image_number']
            output_format = advanced_components['output_format']
            negative_prompt = advanced_components['negative_prompt']
            negative_prompt_advanced = advanced_components['negative_prompt_advanced']
            seed_random = advanced_components['seed_random']
            seed = advanced_components['seed']
            sharpness = advanced_components['sharpness']
            guidance_scale = advanced_components['guidance_scale']
            base_model = advanced_components['base_model']
            refiner_model = advanced_components['refiner_model']
            refiner_switch = advanced_components['refiner_switch']
            styles = advanced_components['styles']
            style_separator = advanced_components['style_separator']
            lora_ctrls = advanced_components['lora_ctrls']
            lora_1_weight = advanced_components['lora_1_weight']
            lora_2_weight = advanced_components['lora_2_weight']
            lora_3_weight = advanced_components['lora_3_weight']
            lora_4_weight = advanced_components['lora_4_weight']
            lora_5_weight = advanced_components['lora_5_weight']
            uov_method_advanced = advanced_components['uov_method_advanced']
            uov_input_image_advanced = advanced_components['uov_input_image_advanced']
            inpaint_disable_initial_latent = advanced_components['inpaint_disable_initial_latent']
            inpaint_engine = advanced_components['inpaint_engine']
            inpaint_strength = advanced_components['inpaint_strength']
            inpaint_respective_field = advanced_components['inpaint_respective_field']
            inpaint_erode_or_dilate = advanced_components['inpaint_erode_or_dilate']
            inpaint_mask_upload_checkbox = advanced_components['inpaint_mask_upload_checkbox']
            invert_mask_checkbox_advanced = advanced_components['invert_mask_checkbox_advanced']
            inpaint_advanced_mask_img_checkbox = advanced_components['inpaint_advanced_mask_img_checkbox']
            controlnet_image = advanced_components['controlnet_image']
            controlnet_canny = advanced_components['controlnet_canny']
            controlnet_cpds = advanced_components['controlnet_cpds']
            controlnet_depth = advanced_components['controlnet_depth']
            controlnet_mask = advanced_components['controlnet_mask']
            controlnet_t2i = advanced_components['controlnet_t2i']
            controlnet_seg = advanced_components['controlnet_seg']
            freeu_b1 = advanced_components['freeu_b1']
            freeu_b2 = advanced_components['freeu_b2']
            freeu_s1 = advanced_components['freeu_s1']
            freeu_s2 = advanced_components['freeu_s2']
            freeu_enabled = advanced_components['freeu_enabled']
            controlnet_softness = advanced_components['controlnet_softness']
            freeu_mode = advanced_components['freeu_mode']
            debug_dino_input = advanced_components['debug_dino_input']
            debug_cn_preprocessor = advanced_components['debug_cn_preprocessor']
            preset_widget = advanced_components.get('preset_selection', None)
            preset_name = advanced_components.get('new_preset_name_input', None)
            save_current_setting_to_preset = advanced_components.get('save_preset_btn', None)
            duplicate_preset = advanced_components.get('duplicate_preset_btn', None)
            rename_preset = advanced_components.get('rename_preset_btn', None)
            delete_preset = advanced_components.get('delete_preset_btn', None)
            preset_status = advanced_components.get('preset_operation_msg', None)
            preset_description = advanced_components.get('preset_details_html', None)
            debug_sam_input = advanced_components['debug_sam_input']
            dev_mode = advanced_components['dev_mode']

    state_is_generating = gr.State(False)

    load_data_outputs = [
        prompt, negative_prompt, style_separator, styles,
        performance_selection, aspect_ratios_selection, image_number,
        negative_prompt_advanced, seed_random, seed, sharpness, guidance_scale,
        base_model, refiner_model, refiner_switch,
        lora_1_weight, lora_2_weight, lora_3_weight, lora_4_weight, lora_5_weight,
        uov_method_advanced,
        inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field, inpaint_erode_or_dilate,
        inpaint_mask_upload_checkbox, invert_mask_checkbox_advanced, inpaint_advanced_mask_img_checkbox,
        controlnet_canny, controlnet_cpds, controlnet_depth, controlnet_mask, controlnet_t2i, controlnet_seg,
        freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_enabled, controlnet_softness,
        freeu_mode, debug_dino_input, state_is_generating, output_format,
        inpaint_mode
    ]

    ctrls = [
        prompt, negative_prompt, style_separator, styles,
        performance_selection, aspect_ratios_selection, image_number, output_format,
        negative_prompt_advanced, seed_random, seed, sharpness, guidance_scale,
        base_model, refiner_model, refiner_switch,
    ] + lora_ctrls + [
        uov_input_image, uov_method, enhance_input_image,
        enhance_uov_method, enhance_uov_processing_order, enhance_uov_prompt_type
    ] + enhance_ctrls + [
        inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image, inpaint_mode,
        inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field,
        inpaint_erode_or_dilate, inpaint_mask_upload_checkbox, invert_mask_checkbox_advanced,
        inpaint_advanced_mask_img_checkbox,
        outpaint_selections
    ] + ip_ctrls + [
        controlnet_image, controlnet_canny, controlnet_cpds, controlnet_depth, controlnet_mask, controlnet_t2i, controlnet_seg,
        freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_enabled, controlnet_softness,
        freeu_mode, debug_dino_input, debug_cn_preprocessor,
        inpaint_mask_model, inpaint_mask_cloth_category,
        inpaint_mask_sam_model, inpaint_mask_text_threshold, inpaint_mask_box_threshold, inpaint_mask_sam_max_detections,
        inpaint_mask_dino_prompt_text, invert_mask_checkbox, inpaint_engine_state, dev_mode, debug_sam_input
    ]

    input_image_checkbox.change(
        lambda x: (gr.update(visible=x), gr.update(visible=x) if modules.config.enhance_expansion else gr.update(visible=False)),
        inputs=input_image_checkbox,
        outputs=[image_input_panel, enhance_panel],
        show_progress=False, queue=False
    )

    ip_advanced.change(
        lambda x: [gr.update(visible=x) for _ in ip_ad_cols],
        inputs=ip_advanced, outputs=ip_ad_cols,
        show_progress=False, queue=False
    )

    advanced_checkbox.change(
        lambda x: (gr.update(visible=x), gr.update(visible=x)),
        inputs=advanced_checkbox, outputs=[advanced_column, input_image_checkbox],
        show_progress=False, queue=False
    )

    enhance_checkbox.change(
        lambda x: gr.update(visible=x),
        inputs=enhance_checkbox, outputs=enhance_panel,
        show_progress=False, queue=False
    )

    inpaint_advanced_masking_checkbox.change(
        lambda x: gr.update(visible=x),
        inputs=inpaint_advanced_masking_checkbox, outputs=inpaint_mask_generation_col,
        show_progress=False, queue=False
    )

    inpaint_mode.change(
        inpaint_mode_change,
        inputs=[inpaint_mode, inpaint_engine_state],
        outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field
        ],
        show_progress=False, queue=False
    )

    inpaint_mask_model.change(
        lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                  [gr.update(visible=x == 'sam')] * 3 +
                  [gr.Dataset.update(visible=x == 'sam',
                                     samples=modules.config.example_enhance_detection_prompts)],
        inputs=inpaint_mask_model,
        outputs=[inpaint_mask_cloth_category, inpaint_mask_dino_prompt_text, inpaint_mask_advanced_options,
                 inpaint_mask_sam_max_detections, example_inpaint_mask_dino_prompt_text],
        queue=False, show_progress=False
    )

    bind_metadata_events(image_components, state_is_generating, inpaint_mode, load_data_outputs)

    bind_advanced_column_events(
        advanced_components, output_format_ref,
        currentTask, state_is_generating, inpaint_mode, load_data_outputs, ctrls
    )

    describe_btn.click(
        trigger_describe,
        inputs=[describe_methods, describe_input_image, describe_apply_styles, style_separator, *styles],
        outputs=[prompt, *styles],
        show_progress=True, queue=True
    )

    generate_mask_button.click(
        generate_mask,
        inputs=[
            inpaint_input_image, inpaint_mask_model, inpaint_mask_cloth_category,
            inpaint_mask_sam_model, inpaint_mask_box_threshold, inpaint_mask_text_threshold,
            inpaint_mask_sam_max_detections, inpaint_mask_dino_prompt_text, invert_mask_checkbox
        ],
        outputs=[inpaint_input_image]
    )

    for enhance_inpaint_mode, _inpaint_disable_initial_latent, _inpaint_engine, \
        _inpaint_strength, _inpaint_respective_field in enhance_inpaint_update_ctrls:
        enhance_inpaint_mode.change(
            inpaint_mode_change, inputs=[enhance_inpaint_mode, inpaint_engine_state],
            outputs=[
                inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
                _inpaint_disable_initial_latent, _inpaint_engine, _inpaint_strength,
                _inpaint_respective_field
            ], show_progress=False, queue=False
        )

    stop_button.click(
        stop_clicked, inputs=[currentTask],
        outputs=[state_is_generating, currentTask]
    )

    skip_button.click(
        skip_clicked, inputs=[currentTask], outputs=[state_is_generating]
    )

    random_seed_event = seed_random.change(
        random_checked, inputs=[seed_random], outputs=[seed]
    )

    for s in [shared.gradio_root.load, preview_gallery.select]:
        s(refresh_seed, inputs=[seed_random, seed], outputs=[seed], show_progress=False, queue=False)

    generate_button.click(
        get_task, inputs=ctrls, outputs=[currentTask],
        show_progress=False, queue=False
    ).then(
        generate_clicked, inputs=[currentTask],
        outputs=[
            state_is_generating,
            prompt,
            negative_prompt,
            preview_gallery,
            preview_html,
            output_format,
        ]
    ).then(
        sort_enhance_images, inputs=[currentTask],
        outputs=[preview_gallery]
    ).success(
        refresh_seed, inputs=[seed_random, seed], outputs=[seed],
        show_progress=False, queue=False
    )

    reset_button = gr.Button(visible=False, elem_id='input_image_reset')
    reset_button.click(
        lambda: [None, 0, None, None],
        outputs=[
            uov_input_image, uov_method,
            inpaint_input_image, describe_input_image
        ],
        queue=False, show_progress=False
    )

shared.demo = shared.gradio_root
shared.gradio_root.queue()


def run(port):
    launch_kwargs = {}
    username = args_manager.args.username
    password = args_manager.args.password
    server_name = args_manager.args.listen
    if username and password:
        launch_kwargs["auth"] = (username, password)
    if server_name is not None:
        launch_kwargs["server_name"] = server_name
    if port is not None:
        launch_kwargs["server_port"] = int(port)
    elif args_manager.args.port is not None:
        launch_kwargs["server_port"] = int(args_manager.args.port)
    else:
        launch_kwargs["server_port"] = 7865
    if args_manager.args.share:
        launch_kwargs["share"] = True
    launch_kwargs["inbrowser"] = args_manager.args.inbrowser
    launch_kwargs["favicon"] = "assets/favicon.svg"
    shared.gradio_root.launch(**launch_kwargs)
