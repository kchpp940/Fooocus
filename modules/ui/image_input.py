import gradio as gr

import modules.config
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.util
import modules.style_sorter as style_sorter
import args_manager
from modules.ui.generation import inpaint_mode_change, generate_mask, trigger_describe


def create_image_input_tabs(gradio_root, inpaint_engine_state):
    components = {}
    ip_ctrls = []
    ip_ad_cols = []
    ip_types = []
    ip_stops = []
    ip_weights = []

    with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id):
        with gr.Tab(label='Upscale or Variation', id='uov_tab') as uov_tab:
            components['uov_tab'] = uov_tab
            with gr.Row():
                with gr.Column():
                    uov_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                    components['uov_input_image'] = uov_input_image
                with gr.Column():
                    uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=modules.config.default_uov_method)
                    components['uov_method'] = uov_method
                    gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>')

        with gr.Tab(label='Image Prompt', id='ip_tab') as ip_tab:
            components['ip_tab'] = ip_tab
            with gr.Row():
                for image_count in range(modules.config.default_controlnet_image_count):
                    image_count += 1
                    with gr.Column():
                        ip_image = grh.Image(
                            label='Image', source='upload', type='numpy', show_label=False, height=300,
                            value=modules.config.default_ip_images[image_count]
                        )
                        ip_ctrls.append(ip_image)
                        with gr.Column(visible=modules.config.default_image_prompt_advanced_checkbox) as ad_col:
                            with gr.Row():
                                ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=modules.config.default_ip_stop_ats[image_count])
                                ip_stops.append(ip_stop)
                                ip_ctrls.append(ip_stop)

                                ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=modules.config.default_ip_weights[image_count])
                                ip_weights.append(ip_weight)
                                ip_ctrls.append(ip_weight)

                            ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=modules.config.default_ip_types[image_count], container=False)
                            ip_types.append(ip_type)
                            ip_ctrls.append(ip_type)

                            ip_type.change(
                                lambda x: flags.default_parameters[x], inputs=[ip_type],
                                outputs=[ip_stop, ip_weight], queue=False, show_progress=False
                            )
                        ip_ad_cols.append(ad_col)

            ip_advanced = gr.Checkbox(label='Advanced', value=modules.config.default_image_prompt_advanced_checkbox, container=False)
            components['ip_advanced'] = ip_advanced
            components['ip_ad_cols'] = ip_ad_cols
            components['ip_types'] = ip_types
            components['ip_stops'] = ip_stops
            components['ip_weights'] = ip_weights
            components['ip_ctrls'] = ip_ctrls
            gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Documentation</a>')

        with gr.Tab(label='Inpaint or Outpaint', id='inpaint_tab') as inpaint_tab:
            components['inpaint_tab'] = inpaint_tab
            with gr.Row():
                with gr.Column():
                    inpaint_input_image = grh.Image(
                        label='Image', source='upload', type='numpy', tool='sketch', height=500,
                        brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False
                    )
                    components['inpaint_input_image'] = inpaint_input_image

                    inpaint_advanced_masking_checkbox = gr.Checkbox(
                        label='Enable Advanced Masking Features',
                        value=modules.config.default_inpaint_advanced_masking_checkbox
                    )
                    components['inpaint_advanced_masking_checkbox'] = inpaint_advanced_masking_checkbox

                    inpaint_mode = gr.Dropdown(
                        choices=modules.flags.inpaint_options,
                        value=modules.config.default_inpaint_method,
                        label='Method'
                    )
                    components['inpaint_mode'] = inpaint_mode

                    inpaint_additional_prompt = gr.Textbox(
                        placeholder="Describe what you want to inpaint.",
                        elem_id='inpaint_additional_prompt',
                        label='Inpaint Additional Prompt',
                        visible=False
                    )
                    components['inpaint_additional_prompt'] = inpaint_additional_prompt

                    outpaint_selections = gr.CheckboxGroup(
                        choices=['Left', 'Right', 'Top', 'Bottom'],
                        value=[], label='Outpaint Direction'
                    )
                    components['outpaint_selections'] = outpaint_selections

                    example_inpaint_prompts = gr.Dataset(
                        samples=modules.config.example_inpaint_prompts,
                        label='Additional Prompt Quick List',
                        components=[inpaint_additional_prompt],
                        visible=False
                    )
                    components['example_inpaint_prompts'] = example_inpaint_prompts
                    gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')
                    example_inpaint_prompts.click(
                        lambda x: x[0], inputs=example_inpaint_prompts,
                        outputs=inpaint_additional_prompt, show_progress=False, queue=False
                    )

                with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:
                    components['inpaint_mask_generation_col'] = inpaint_mask_generation_col

                    inpaint_mask_image = grh.Image(
                        label='Mask Upload', source='upload', type='numpy', tool='sketch', height=500,
                        brush_color="#FFFFFF", mask_opacity=1, elem_id='inpaint_mask_canvas'
                    )
                    components['inpaint_mask_image'] = inpaint_mask_image

                    invert_mask_checkbox = gr.Checkbox(
                        label='Invert Mask When Generating',
                        value=modules.config.default_invert_mask_checkbox
                    )
                    components['invert_mask_checkbox'] = invert_mask_checkbox

                    inpaint_mask_model = gr.Dropdown(
                        label='Mask generation model',
                        choices=flags.inpaint_mask_models,
                        value=modules.config.default_inpaint_mask_model
                    )
                    components['inpaint_mask_model'] = inpaint_mask_model

                    inpaint_mask_cloth_category = gr.Dropdown(
                        label='Cloth category',
                        choices=flags.inpaint_mask_cloth_category,
                        value=modules.config.default_inpaint_mask_cloth_category,
                        visible=False
                    )
                    components['inpaint_mask_cloth_category'] = inpaint_mask_cloth_category

                    inpaint_mask_dino_prompt_text = gr.Textbox(
                        label='Detection prompt', value='', visible=False,
                        info='Use singular whenever possible',
                        placeholder='Describe what you want to detect.'
                    )
                    components['inpaint_mask_dino_prompt_text'] = inpaint_mask_dino_prompt_text

                    example_inpaint_mask_dino_prompt_text = gr.Dataset(
                        samples=modules.config.example_enhance_detection_prompts,
                        label='Detection Prompt Quick List',
                        components=[inpaint_mask_dino_prompt_text],
                        visible=modules.config.default_inpaint_mask_model == 'sam'
                    )
                    components['example_inpaint_mask_dino_prompt_text'] = example_inpaint_mask_dino_prompt_text
                    example_inpaint_mask_dino_prompt_text.click(
                        lambda x: x[0], inputs=example_inpaint_mask_dino_prompt_text,
                        outputs=inpaint_mask_dino_prompt_text, show_progress=False, queue=False
                    )

                    with gr.Accordion("Advanced options", visible=False, open=False) as inpaint_mask_advanced_options:
                        components['inpaint_mask_advanced_options'] = inpaint_mask_advanced_options
                        inpaint_mask_sam_model = gr.Dropdown(
                            label='SAM model', choices=flags.inpaint_mask_sam_model,
                            value=modules.config.default_inpaint_mask_sam_model
                        )
                        components['inpaint_mask_sam_model'] = inpaint_mask_sam_model

                        inpaint_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                        components['inpaint_mask_box_threshold'] = inpaint_mask_box_threshold

                        inpaint_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                        components['inpaint_mask_text_threshold'] = inpaint_mask_text_threshold

                        inpaint_mask_sam_max_detections = gr.Slider(
                            label="Maximum number of detections",
                            info="Set to 0 to detect all", minimum=0, maximum=10,
                            value=modules.config.default_sam_max_detections, step=1, interactive=True
                        )
                        components['inpaint_mask_sam_max_detections'] = inpaint_mask_sam_max_detections

                    generate_mask_button = gr.Button(value='Generate mask from image')
                    components['generate_mask_button'] = generate_mask_button

        with gr.Tab(label='Describe', id='describe_tab') as describe_tab:
            components['describe_tab'] = describe_tab
            with gr.Row():
                with gr.Column():
                    describe_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                    components['describe_input_image'] = describe_input_image
                with gr.Column():
                    describe_methods = gr.CheckboxGroup(
                        label='Content Type',
                        choices=flags.describe_types,
                        value=modules.config.default_describe_content_type
                    )
                    components['describe_methods'] = describe_methods

                    describe_apply_styles = gr.Checkbox(
                        label='Apply Styles',
                        value=modules.config.default_describe_apply_prompts_checkbox
                    )
                    components['describe_apply_styles'] = describe_apply_styles

                    describe_btn = gr.Button(value='Describe this Image into Prompt')
                    components['describe_btn'] = describe_btn

                    describe_image_size = gr.Textbox(
                        label='Image Size and Recommended Size',
                        elem_id='describe_image_size', visible=False
                    )
                    components['describe_image_size'] = describe_image_size
                    gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Documentation</a>')

                    def trigger_show_image_properties(image):
                        value = modules.util.get_image_size_info(image, modules.flags.sdxl_aspect_ratios)
                        return gr.update(value=value, visible=True)

                    describe_input_image.upload(
                        trigger_show_image_properties, inputs=describe_input_image,
                        outputs=describe_image_size, show_progress=False, queue=False
                    )

        with gr.Tab(label='Enhance', id='enhance_tab') as enhance_tab:
            components['enhance_tab'] = enhance_tab
            with gr.Row():
                with gr.Column():
                    enhance_input_image = grh.Image(label='Use with Enhance, skips image generation', source='upload', type='numpy')
                    components['enhance_input_image'] = enhance_input_image
                    gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

        with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
            components['metadata_tab'] = metadata_tab
            with gr.Column():
                metadata_input_image = grh.Image(label='For images created by Fooocus', source='upload', type='pil')
                metadata_json = gr.JSON(label='Metadata')
                metadata_import_button = gr.Button(value='Apply Metadata')

            components['metadata_input_image'] = metadata_input_image
            components['metadata_json'] = metadata_json
            components['metadata_import_button'] = metadata_import_button

    return components


def create_enhance_input_panel(inpaint_engine_state, inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts):
    components = {}
    enhance_ctrls = []
    enhance_inpaint_mode_ctrls = []
    enhance_inpaint_engine_ctrls = []
    enhance_inpaint_update_ctrls = []

    with gr.Tabs():
        with gr.Tab(label='Upscale or Variation'):
            with gr.Row():
                with gr.Column():
                    enhance_uov_method = gr.Radio(
                        label='Upscale or Variation:', choices=flags.uov_list,
                        value=modules.config.default_enhance_uov_method
                    )
                    components['enhance_uov_method'] = enhance_uov_method

                    enhance_uov_processing_order = gr.Radio(
                        label='Order of Processing',
                        info='Use before to enhance small details and after to enhance large areas.',
                        choices=flags.enhancement_uov_processing_order,
                        value=modules.config.default_enhance_uov_processing_order
                    )
                    components['enhance_uov_processing_order'] = enhance_uov_processing_order

                    enhance_uov_prompt_type = gr.Radio(
                        label='Prompt',
                        info='Choose which prompt to use for Upscale or Variation.',
                        choices=flags.enhancement_uov_prompt_types,
                        value=modules.config.default_enhance_uov_prompt_type,
                        visible=modules.config.default_enhance_uov_processing_order == flags.enhancement_uov_after
                    )
                    components['enhance_uov_prompt_type'] = enhance_uov_prompt_type

                    enhance_uov_processing_order.change(
                        lambda x: gr.update(visible=x == flags.enhancement_uov_after),
                        inputs=enhance_uov_processing_order,
                        outputs=enhance_uov_prompt_type,
                        queue=False, show_progress=False
                    )
                    gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

        for index in range(modules.config.default_enhance_tabs):
            with gr.Tab(label=f'#{index + 1}') as enhance_tab_item:
                enhance_enabled = gr.Checkbox(label='Enable', value=False, elem_classes='min_check', container=False)

                enhance_mask_dino_prompt_text = gr.Textbox(
                    label='Detection prompt', info='Use singular whenever possible',
                    placeholder='Describe what you want to detect.',
                    interactive=True,
                    visible=modules.config.default_enhance_inpaint_mask_model == 'sam'
                )
                example_enhance_mask_dino_prompt_text = gr.Dataset(
                    samples=modules.config.example_enhance_detection_prompts,
                    label='Detection Prompt Quick List',
                    components=[enhance_mask_dino_prompt_text],
                    visible=modules.config.default_enhance_inpaint_mask_model == 'sam'
                )
                example_enhance_mask_dino_prompt_text.click(
                    lambda x: x[0], inputs=example_enhance_mask_dino_prompt_text,
                    outputs=enhance_mask_dino_prompt_text, show_progress=False, queue=False
                )

                enhance_prompt = gr.Textbox(
                    label="Enhancement positive prompt",
                    placeholder="Uses original prompt instead if empty.",
                    elem_id='enhance_prompt'
                )
                enhance_negative_prompt = gr.Textbox(
                    label="Enhancement negative prompt",
                    placeholder="Uses original negative prompt instead if empty.",
                    elem_id='enhance_negative_prompt'
                )

                with gr.Accordion("Detection", open=False):
                    enhance_mask_model = gr.Dropdown(
                        label='Mask generation model',
                        choices=flags.inpaint_mask_models,
                        value=modules.config.default_enhance_inpaint_mask_model
                    )
                    enhance_mask_cloth_category = gr.Dropdown(
                        label='Cloth category',
                        choices=flags.inpaint_mask_cloth_category,
                        value=modules.config.default_inpaint_mask_cloth_category,
                        visible=modules.config.default_enhance_inpaint_mask_model == 'u2net_cloth_seg',
                        interactive=True
                    )

                    with gr.Accordion(
                        "SAM Options",
                        visible=modules.config.default_enhance_inpaint_mask_model == 'sam',
                        open=False
                    ) as sam_options:
                        enhance_mask_sam_model = gr.Dropdown(
                            label='SAM model', choices=flags.inpaint_mask_sam_model,
                            value=modules.config.default_inpaint_mask_sam_model,
                            interactive=True
                        )
                        enhance_mask_box_threshold = gr.Slider(
                            label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                            interactive=True
                        )
                        enhance_mask_text_threshold = gr.Slider(
                            label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                            interactive=True
                        )
                        enhance_mask_sam_max_detections = gr.Slider(
                            label="Maximum number of detections",
                            info="Set to 0 to detect all",
                            minimum=0, maximum=10,
                            value=modules.config.default_sam_max_detections,
                            step=1, interactive=True
                        )

                with gr.Accordion("Inpaint", visible=True, open=False):
                    enhance_inpaint_mode = gr.Dropdown(
                        choices=modules.flags.inpaint_options,
                        value=modules.config.default_inpaint_method,
                        label='Method', interactive=True
                    )
                    enhance_inpaint_disable_initial_latent = gr.Checkbox(
                        label='Disable initial latent in inpaint', value=False
                    )
                    enhance_inpaint_engine = gr.Dropdown(
                        label='Inpaint Engine',
                        value=modules.config.default_inpaint_engine_version,
                        choices=flags.inpaint_engine_versions,
                        info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.'
                    )
                    enhance_inpaint_strength = gr.Slider(
                        label='Inpaint Denoising Strength',
                        minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                        info='Same as the denoising strength in A1111 inpaint. '
                             'Only used in inpaint, not used in outpaint. '
                             '(Outpaint always use 1.0)'
                    )
                    enhance_inpaint_respective_field = gr.Slider(
                        label='Inpaint Respective Field',
                        minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                        info='The area to inpaint. '
                             'Value 0 is same as "Only Masked" in A1111. '
                             'Value 1 is same as "Whole Image" in A1111. '
                             'Only used in inpaint, not used in outpaint. '
                             '(Outpaint always use 1.0)'
                    )
                    enhance_inpaint_erode_or_dilate = gr.Slider(
                        label='Mask Erode or Dilate',
                        minimum=-64, maximum=64, step=1, value=0,
                        info='Positive value will make white area in the mask larger, '
                             'negative value will make white area smaller. '
                             '(default is 0, always processed before any mask invert)'
                    )
                    enhance_mask_invert = gr.Checkbox(label='Invert Mask', value=False)

                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

            enhance_ctrls += [
                enhance_enabled,
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
            ]

            enhance_inpaint_mode_ctrls += [enhance_inpaint_mode]
            enhance_inpaint_engine_ctrls += [enhance_inpaint_engine]

            enhance_inpaint_update_ctrls += [[
                enhance_inpaint_mode, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                enhance_inpaint_strength, enhance_inpaint_respective_field
            ]]

            enhance_inpaint_mode.change(
                inpaint_mode_change, inputs=[enhance_inpaint_mode, inpaint_engine_state],
                outputs=[
                    inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
                    enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                    enhance_inpaint_strength, enhance_inpaint_respective_field
                ],
                show_progress=False, queue=False
            )

            enhance_mask_model.change(
                lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                          [gr.update(visible=x == 'sam')] * 2 +
                          [gr.Dataset.update(visible=x == 'sam',
                                             samples=modules.config.example_enhance_detection_prompts)],
                inputs=enhance_mask_model,
                outputs=[enhance_mask_cloth_category, enhance_mask_dino_prompt_text, sam_options,
                         example_enhance_mask_dino_prompt_text],
                queue=False, show_progress=False
            )

    components['enhance_ctrls'] = enhance_ctrls
    components['enhance_inpaint_mode_ctrls'] = enhance_inpaint_mode_ctrls
    components['enhance_inpaint_engine_ctrls'] = enhance_inpaint_engine_ctrls
    components['enhance_inpaint_update_ctrls'] = enhance_inpaint_update_ctrls

    return components
