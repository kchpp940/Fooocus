import gradio as gr

import modules.config
import modules.flags as flags
import args_manager
import copy
from modules.sdxl_styles import legal_style_names
import modules.style_sorter as style_sorter
from modules.ui.generation import (
    random_checked, refresh_seed, update_history_link,
    dev_mode_checked, refresh_files_clicked,
    preset_selection_change, inpaint_engine_state_change,
    preset_selection_update_details, format_preset_details_html,
    save_current_as_preset, duplicate_current_preset,
    rename_current_preset, delete_current_preset,
    inpaint_mode_change
)


def create_settings_tab(gradio_root, output_format_ref):
    components = {}

    if not args_manager.args.disable_preset_selection:
        preset_selection = gr.Dropdown(label='Preset',
                                       choices=modules.config.available_presets,
                                       value=args_manager.args.preset if args_manager.args.preset else "initial",
                                       interactive=True)
        components['preset_selection'] = preset_selection

        with gr.Accordion(label='Preset Details & Management', open=True):
            preset_details_html = gr.HTML(
                value=update_preset_details(
                    args_manager.args.preset if args_manager.args.preset else "initial"
                )
            )
            components['preset_details_html'] = preset_details_html

            with gr.Row():
                new_preset_name_input = gr.Textbox(
                    label='New Preset Name',
                    placeholder='Enter preset name...',
                    interactive=True,
                    scale=3
                )
                save_preset_btn = gr.Button(
                    label='💾 Save Current as Preset',
                    variant='primary',
                    scale=2
                )

            with gr.Row():
                duplicate_preset_btn = gr.Button(
                    label='📋 Duplicate Current',
                    variant='secondary'
                )
                rename_preset_btn = gr.Button(
                    label='✏️ Rename',
                    variant='secondary'
                )
                delete_preset_btn = gr.Button(
                    label='🗑️ Delete',
                    variant='stop'
                )

            preset_operation_msg = gr.Textbox(
                label='Status',
                value='',
                interactive=False,
                show_label=True
            )

            components['new_preset_name_input'] = new_preset_name_input
            components['save_preset_btn'] = save_preset_btn
            components['duplicate_preset_btn'] = duplicate_preset_btn
            components['rename_preset_btn'] = rename_preset_btn
            components['delete_preset_btn'] = delete_preset_btn
            components['preset_operation_msg'] = preset_operation_msg

    performance_selection = gr.Radio(label='Performance',
                                     choices=flags.Performance.values(),
                                     value=modules.config.default_performance,
                                     elem_classes=['performance_selection'])
    components['performance_selection'] = performance_selection

    with gr.Accordion(label='Aspect Ratios', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
        aspect_ratios_selection = gr.Radio(label='Aspect Ratios', show_label=False,
                                           choices=modules.config.available_aspect_ratios_labels,
                                           value=modules.config.default_aspect_ratio,
                                           info='width × height',
                                           elem_classes='aspect_ratios')
        components['aspect_ratios_selection'] = aspect_ratios_selection

        aspect_ratios_selection.change(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
        gradio_root.load(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')

    image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)
    components['image_number'] = image_number

    output_format = gr.Radio(label='Output Format',
                             choices=flags.OutputFormat.list(),
                             value=modules.config.default_output_format)
    components['output_format'] = output_format
    output_format_ref[0] = output_format

    negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.",
                                 info='Describing what you do not want to see.', lines=2,
                                 elem_id='negative_prompt',
                                 value=modules.config.default_prompt_negative)
    components['negative_prompt'] = negative_prompt

    seed_random = gr.Checkbox(label='Random', value=True)
    components['seed_random'] = seed_random

    image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False)
    components['image_seed'] = image_seed

    history_link = gr.HTML()
    components['history_link'] = history_link
    gradio_root.load(lambda: update_history_link(output_format.value), outputs=history_link, queue=False, show_progress=False)

    return components


def update_preset_details(preset_name):
    return format_preset_details_html(preset_name)


def create_styles_tab(gradio_root):
    components = {}

    style_sorter.try_load_sorted_styles(
        style_names=legal_style_names,
        default_selected=modules.config.default_styles)

    style_search_bar = gr.Textbox(show_label=False, container=False,
                                  placeholder="\U0001F50E Type here to search styles ...",
                                  value="",
                                  label='Search Styles')
    components['style_search_bar'] = style_search_bar

    style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                        choices=copy.deepcopy(style_sorter.all_styles),
                                        value=copy.deepcopy(modules.config.default_styles),
                                        label='Selected Styles',
                                        elem_classes=['style_selections'])
    components['style_selections'] = style_selections

    gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)
    components['gradio_receiver_style_selections'] = gradio_receiver_style_selections

    gradio_root.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                     outputs=style_selections)

    style_search_bar.change(style_sorter.search_styles,
                            inputs=[style_selections, style_search_bar],
                            outputs=style_selections,
                            queue=False,
                            show_progress=False).then(
        lambda: None, _js='()=>{refresh_style_localization();}')

    gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                           inputs=style_selections,
                                           outputs=style_selections,
                                           queue=False,
                                           show_progress=False).then(
        lambda: None, _js='()=>{refresh_style_localization();}')

    return components


def create_models_tab(gradio_root):
    components = {}

    with gr.Group():
        with gr.Row():
            base_model = gr.Dropdown(label='Base Model (SDXL only)', choices=modules.config.model_filenames, value=modules.config.default_base_model_name, show_label=True)
            refiner_model = gr.Dropdown(label='Refiner (SDXL or SD 1.5)', choices=['None'] + modules.config.model_filenames, value=modules.config.default_refiner_model_name, show_label=True)

        refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                   info='Use 0.4 for SD1.5 realistic models; '
                                        'or 0.667 for SD1.5 anime models; '
                                        'or 0.8 for XL-refiners; '
                                        'or any value for switching two SDXL models.',
                                   value=modules.config.default_refiner_switch,
                                   visible=modules.config.default_refiner_model_name != 'None')

        refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                             inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)

    components['base_model'] = base_model
    components['refiner_model'] = refiner_model
    components['refiner_switch'] = refiner_switch

    with gr.Group():
        lora_ctrls = []

        for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
            with gr.Row():
                lora_enabled = gr.Checkbox(label='Enable', value=enabled,
                                           elem_classes=['lora_enable', 'min_check'], scale=1)
                lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                         choices=['None'] + modules.config.lora_filenames, value=filename,
                                         elem_classes='lora_model', scale=5)
                lora_weight = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight,
                                        maximum=modules.config.default_loras_max_weight, step=0.01, value=weight,
                                        elem_classes='lora_weight', scale=5)
                lora_ctrls += [lora_enabled, lora_model, lora_weight]

    components['lora_ctrls'] = lora_ctrls

    with gr.Row():
        refresh_files = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
    components['refresh_files'] = refresh_files

    return components


def create_advanced_tab(gradio_root):
    components = {}

    guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                               value=modules.config.default_cfg_scale,
                               info='Higher value means style is cleaner, vivider, and more artistic.')
    components['guidance_scale'] = guidance_scale

    sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                          value=modules.config.default_sample_sharpness,
                          info='Higher value means image and texture are sharper.')
    components['sharpness'] = sharpness

    gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Documentation</a>')

    dev_mode = gr.Checkbox(label='Developer Debug Mode', value=modules.config.default_developer_debug_mode_checkbox, container=False)
    components['dev_mode'] = dev_mode

    with gr.Column(visible=modules.config.default_developer_debug_mode_checkbox) as dev_tools:
        components['dev_tools'] = dev_tools

        with gr.Tab(label='Debug Tools'):
            adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                            step=0.001, value=1.5, info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
            adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                            step=0.001, value=0.8, info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
            adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                       step=0.001, value=0.3,
                                       info='When to end the guidance from positive/negative ADM. ')

            refiner_swap_method = gr.Dropdown(label='Refiner swap method', value=flags.refiner_swap_method,
                                              choices=['joint', 'separate', 'vae'])

            adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                     value=modules.config.default_cfg_tsnr,
                                     info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                          '(effective when real CFG > mimicked CFG).')
            clip_skip = gr.Slider(label='CLIP Skip', minimum=1, maximum=flags.clip_skip_max, step=1,
                                     value=modules.config.default_clip_skip,
                                     info='Bypass CLIP layers to avoid overfitting (use 1 to not skip any layers, 2 is recommended).')
            sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                       value=modules.config.default_sampler)
            scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                         value=modules.config.default_scheduler)
            vae_name = gr.Dropdown(label='VAE', choices=[modules.flags.default_vae] + modules.config.vae_filenames,
                                         value=modules.config.default_vae, show_label=True)

            generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                              info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
                                              value=False)

            overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                       minimum=-1, maximum=200, step=1,
                                       value=modules.config.default_overwrite_step,
                                       info='Set as -1 to disable. For developer debugging.')
            overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                         minimum=-1, maximum=200, step=1,
                                         value=modules.config.default_overwrite_switch,
                                         info='Set as -1 to disable. For developer debugging.')
            overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                        minimum=-1, maximum=2048, step=1, value=-1,
                                        info='Set as -1 to disable. For developer debugging. '
                                             'Results will be worse for non-standard numbers that SDXL is not trained on.')
            overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                         minimum=-1, maximum=2048, step=1, value=-1,
                                         info='Set as -1 to disable. For developer debugging. '
                                              'Results will be worse for non-standard numbers that SDXL is not trained on.')
            overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                info='Set as negative number to disable. For developer debugging.')
            overwrite_upscale_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Upscale"',
                                                   minimum=-1, maximum=1.0, step=0.001,
                                                   value=modules.config.default_overwrite_upscale,
                                                   info='Set as negative number to disable. For developer debugging.')

            disable_preview = gr.Checkbox(label='Disable Preview', value=modules.config.default_black_out_nsfw,
                                          interactive=not modules.config.default_black_out_nsfw,
                                          info='Disable preview during generation.')
            disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results',
                                                          value=flags.Performance.has_restricted_features(modules.config.default_performance),
                                                          info='Disable intermediate results during generation, only show final gallery.')

            disable_seed_increment = gr.Checkbox(label='Disable seed increment',
                                                 info='Disable automatic seed increment when image number is > 1.',
                                                 value=False)
            read_wildcards_in_order = gr.Checkbox(label="Read wildcards in order", value=False)

            black_out_nsfw = gr.Checkbox(label='Black Out NSFW', value=modules.config.default_black_out_nsfw,
                                         interactive=not modules.config.default_black_out_nsfw,
                                         info='Use black image if NSFW is detected.')

            black_out_nsfw.change(lambda x: gr.update(value=x, interactive=not x),
                                  inputs=black_out_nsfw, outputs=disable_preview, queue=False,
                                  show_progress=False)

            save_final_enhanced_image_only = None
            if not args_manager.args.disable_image_log:
                save_final_enhanced_image_only = gr.Checkbox(label='Save only final enhanced image',
                                                             value=modules.config.default_save_only_final_enhanced_image)

            save_metadata_to_images = None
            metadata_scheme = None
            if not args_manager.args.disable_metadata:
                save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images', value=modules.config.default_save_metadata_to_images,
                                                      info='Adds parameters to generated images allowing manual regeneration.')
                metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                           info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
                                           visible=modules.config.default_save_metadata_to_images)

                save_metadata_to_images.change(lambda x: gr.update(visible=x), inputs=[save_metadata_to_images], outputs=[metadata_scheme],
                                               queue=False, show_progress=False)

            components['adm_scaler_positive'] = adm_scaler_positive
            components['adm_scaler_negative'] = adm_scaler_negative
            components['adm_scaler_end'] = adm_scaler_end
            components['refiner_swap_method'] = refiner_swap_method
            components['adaptive_cfg'] = adaptive_cfg
            components['clip_skip'] = clip_skip
            components['sampler_name'] = sampler_name
            components['scheduler_name'] = scheduler_name
            components['vae_name'] = vae_name
            components['generate_image_grid'] = generate_image_grid
            components['overwrite_step'] = overwrite_step
            components['overwrite_switch'] = overwrite_switch
            components['overwrite_width'] = overwrite_width
            components['overwrite_height'] = overwrite_height
            components['overwrite_vary_strength'] = overwrite_vary_strength
            components['overwrite_upscale_strength'] = overwrite_upscale_strength
            components['disable_preview'] = disable_preview
            components['disable_intermediate_results'] = disable_intermediate_results
            components['disable_seed_increment'] = disable_seed_increment
            components['read_wildcards_in_order'] = read_wildcards_in_order
            components['black_out_nsfw'] = black_out_nsfw
            components['save_final_enhanced_image_only'] = save_final_enhanced_image_only
            components['save_metadata_to_images'] = save_metadata_to_images
            components['metadata_scheme'] = metadata_scheme

        with gr.Tab(label='Control'):
            debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                    info='See the results from preprocessors.')
            skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                   info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')

            mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                                                               value=False)
            mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                          value=False)

            controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                            step=0.001, value=0.25,
                                            info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')

            with gr.Tab(label='Canny'):
                canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                step=1, value=64)
                canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1, maximum=255,
                                                 step=1, value=128)

            components['debugging_cn_preprocessor'] = debugging_cn_preprocessor
            components['skipping_cn_preprocessor'] = skipping_cn_preprocessor
            components['mixing_image_prompt_and_vary_upscale'] = mixing_image_prompt_and_vary_upscale
            components['mixing_image_prompt_and_inpaint'] = mixing_image_prompt_and_inpaint
            components['controlnet_softness'] = controlnet_softness
            components['canny_low_threshold'] = canny_low_threshold
            components['canny_high_threshold'] = canny_high_threshold

        with gr.Tab(label='Inpaint'):
            debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
            debugging_enhance_masks_checkbox = gr.Checkbox(label='Debug Enhance Masks', value=False,
                                                           info='Show enhance masks in preview and final results')
            debugging_dino = gr.Checkbox(label='Debug GroundingDINO', value=False,
                                                     info='Use GroundingDINO boxes instead of more detailed SAM masks')
            inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
            inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                         value=modules.config.default_inpaint_engine_version,
                                         choices=flags.inpaint_engine_versions,
                                         info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
            inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                         minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                         info='Same as the denoising strength in A1111 inpaint. '
                                              'Only used in inpaint, not used in outpaint. '
                                              '(Outpaint always use 1.0)')
            inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                 minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                 info='The area to inpaint. '
                                                      'Value 0 is same as "Only Masked" in A1111. '
                                                      'Value 1 is same as "Whole Image" in A1111. '
                                                      'Only used in inpaint, not used in outpaint. '
                                                      '(Outpaint always use 1.0)')
            inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                minimum=-64, maximum=64, step=1, value=0,
                                                info='Positive value will make white area in the mask larger, '
                                                     'negative value will make white area smaller. '
                                                     '(default is 0, always processed before any mask invert)')
            dino_erode_or_dilate = gr.Slider(label='GroundingDINO Box Erode or Dilate',
                                             minimum=-64, maximum=64, step=1, value=0,
                                             info='Positive value will make white area in the mask larger, '
                                                  'negative value will make white area smaller. '
                                                  '(default is 0, processed before SAM)')

            inpaint_mask_color = gr.ColorPicker(label='Inpaint brush color', value='#FFFFFF', elem_id='inpaint_brush_color')

            components['debugging_inpaint_preprocessor'] = debugging_inpaint_preprocessor
            components['debugging_enhance_masks_checkbox'] = debugging_enhance_masks_checkbox
            components['debugging_dino'] = debugging_dino
            components['inpaint_disable_initial_latent'] = inpaint_disable_initial_latent
            components['inpaint_engine'] = inpaint_engine
            components['inpaint_strength'] = inpaint_strength
            components['inpaint_respective_field'] = inpaint_respective_field
            components['inpaint_erode_or_dilate'] = inpaint_erode_or_dilate
            components['dino_erode_or_dilate'] = dino_erode_or_dilate
            components['inpaint_mask_color'] = inpaint_mask_color

        with gr.Tab(label='FreeU'):
            freeu_enabled = gr.Checkbox(label='Enabled', value=False)
            freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
            freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
            freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
            freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
            freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

            components['freeu_enabled'] = freeu_enabled
            components['freeu_b1'] = freeu_b1
            components['freeu_b2'] = freeu_b2
            components['freeu_s1'] = freeu_s1
            components['freeu_s2'] = freeu_s2
            components['freeu_ctrls'] = freeu_ctrls

    return components


def create_all_advanced_tabs(gradio_root, output_format_ref=None):
    components = {}

    if output_format_ref is None:
        output_format_ref = [None]

    with gr.Tab(label='Settings'):
        settings = create_settings_tab(gradio_root, output_format_ref)
        components.update(settings)

    with gr.Tab(label='Styles', elem_classes=['style_selections_tab']):
        styles = create_styles_tab(gradio_root)
        components.update(styles)

    with gr.Tab(label='Models'):
        models = create_models_tab(gradio_root)
        components.update(models)

    with gr.Tab(label='Advanced'):
        adv = create_advanced_tab(gradio_root)
        components.update(adv)

    return components


def bind_advanced_column_events(components, gradio_root, inpaint_engine_state,
                                  inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
                                  inpaint_input_image, inpaint_mask_image, inpaint_mask_generation_col,
                                  inpaint_advanced_masking_checkbox, invert_mask_checkbox,
                                  enhance_inpaint_mode_ctrls, enhance_inpaint_update_ctrls, enhance_inpaint_engine_ctrls,
                                  state_is_generating, inpaint_mode, load_data_outputs,
                                  style_selections, output_format_ref):
    seed_random = components['seed_random']
    image_seed = components['image_seed']

    seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                       queue=False, show_progress=False)

    dev_mode = components['dev_mode']
    dev_tools = components['dev_tools']

    dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                    queue=False, show_progress=False)

    if 'refresh_files' in components:
        refresh_files = components['refresh_files']
        base_model = components['base_model']
        refiner_model = components['refiner_model']
        vae_name = components['vae_name']
        lora_ctrls = components['lora_ctrls']

        refresh_files_output = [base_model, refiner_model, vae_name]
        if not args_manager.args.disable_preset_selection:
            preset_selection = components['preset_selection']
            refresh_files_output += [preset_selection]

        refresh_files.click(refresh_files_clicked, [], refresh_files_output + lora_ctrls,
                            queue=False, show_progress=False)

    if not args_manager.args.disable_preset_selection:
        preset_selection = components['preset_selection']
        new_preset_name_input = components['new_preset_name_input']
        save_preset_btn = components['save_preset_btn']
        duplicate_preset_btn = components['duplicate_preset_btn']
        rename_preset_btn = components['rename_preset_btn']
        delete_preset_btn = components['delete_preset_btn']
        preset_details_html = components['preset_details_html']
        preset_operation_msg = components['preset_operation_msg']

        base_model = components['base_model']
        refiner_model = components['refiner_model']
        refiner_switch = components['refiner_switch']
        guidance_scale = components['guidance_scale']
        sharpness = components['sharpness']
        adaptive_cfg = components['adaptive_cfg']
        clip_skip = components['clip_skip']
        sampler_name = components['sampler_name']
        scheduler_name = components['scheduler_name']
        vae_name = components['vae_name']
        performance_selection = components['performance_selection']
        aspect_ratios_selection = components['aspect_ratios_selection']
        style_selections = components['style_selections']
        overwrite_step = components['overwrite_step']
        inpaint_engine = components['inpaint_engine']
        lora_ctrls = components['lora_ctrls']

        preset_selection.change(
            preset_selection_change,
            inputs=[preset_selection, state_is_generating, inpaint_mode],
            outputs=load_data_outputs, queue=False, show_progress=True
        ).then(
            fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections,
            queue=False, show_progress=False
        ).then(
            lambda: None, _js='()=>{refresh_style_localization();}'
        ).then(
            inpaint_engine_state_change,
            inputs=[inpaint_engine_state] + enhance_inpaint_mode_ctrls,
            outputs=enhance_inpaint_engine_ctrls, queue=False, show_progress=False
        )

        preset_selection.change(
            preset_selection_update_details,
            inputs=[preset_selection],
            outputs=[preset_details_html],
            queue=False,
            show_progress=False
        )

        save_preset_inputs = [new_preset_name_input]
        save_preset_inputs += [
            base_model, refiner_model, refiner_switch,
            guidance_scale, sharpness, adaptive_cfg, clip_skip,
            sampler_name, scheduler_name, vae_name,
            performance_selection, aspect_ratios_selection,
            style_selections, overwrite_step, inpaint_engine
        ]
        save_preset_inputs += lora_ctrls

        save_preset_btn.click(
            save_current_as_preset,
            inputs=save_preset_inputs,
            outputs=[preset_selection, preset_details_html, preset_operation_msg],
            queue=False,
            show_progress=False
        )

        duplicate_preset_btn.click(
            duplicate_current_preset,
            inputs=[preset_selection, new_preset_name_input],
            outputs=[preset_selection, preset_details_html, preset_operation_msg],
            queue=False,
            show_progress=False
        )

        rename_preset_btn.click(
            rename_current_preset,
            inputs=[preset_selection, new_preset_name_input],
            outputs=[preset_selection, preset_details_html, preset_operation_msg],
            queue=False,
            show_progress=False
        )

        delete_preset_btn.click(
            delete_current_preset,
            inputs=[preset_selection],
            outputs=[preset_selection, preset_details_html, preset_operation_msg],
            queue=False,
            show_progress=False
        )

    if 'inpaint_advanced_masking_checkbox' in components:
        pass

    if 'inpaint_mask_color' in components:
        inpaint_mask_color = components['inpaint_mask_color']
        inpaint_mask_color.change(lambda x: gr.update(brush_color=x), inputs=inpaint_mask_color,
                                  outputs=inpaint_input_image,
                                  queue=False, show_progress=False)

    inpaint_advanced_masking_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                             inputs=inpaint_advanced_masking_checkbox,
                                             outputs=[inpaint_mask_image, inpaint_mask_generation_col],
                                             queue=False, show_progress=False)

    inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state], outputs=[
        inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
        components['inpaint_disable_initial_latent'], components['inpaint_engine'],
        components['inpaint_strength'], components['inpaint_respective_field']
    ], show_progress=False, queue=False)

    default_inpaint_ctrls = [inpaint_mode, components['inpaint_disable_initial_latent'],
                             components['inpaint_engine'], components['inpaint_strength'],
                             components['inpaint_respective_field']]

    for mode, disable_initial_latent, engine, strength, respective_field in [default_inpaint_ctrls] + enhance_inpaint_update_ctrls:
        gradio_root.load(inpaint_mode_change, inputs=[mode, inpaint_engine_state], outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts, disable_initial_latent,
            engine, strength, respective_field
        ], show_progress=False, queue=False)

    output_format = output_format_ref[0]
    if output_format is not None:
        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

    history_link = components['history_link']
    if output_format is not None:
        output_format.change(lambda: update_history_link(output_format.value), outputs=history_link, queue=False, show_progress=False)
