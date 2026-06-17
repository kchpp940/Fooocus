import args_manager
from modules.ui.types import (
    TopCheckboxes, PromptAndButtons,
    ImageInputTabs, EnhancePanelComponents,
    AdvancedColumnComponents
)


def build_load_data_outputs(
    top_checkboxes: TopCheckboxes,
    prompt_buttons: PromptAndButtons,
    image_input: ImageInputTabs,
    enhance_panel: EnhancePanelComponents,
    advanced: AdvancedColumnComponents,
    inpaint_engine_state,
):
    s = advanced.settings
    a = advanced.advanced

    outputs = [
        top_checkboxes.advanced_checkbox,
        s.image_number,
        prompt_buttons.prompt,
        s.negative_prompt,
        advanced.styles.style_selections,
        s.performance_selection,
        a.debug_tools.overwrite_step,
        a.debug_tools.overwrite_switch,
        s.aspect_ratios_selection,
        a.debug_tools.overwrite_width,
        a.debug_tools.overwrite_height,
        a.guidance_scale,
        a.sharpness,
        a.debug_tools.adm_scaler_positive,
        a.debug_tools.adm_scaler_negative,
        a.debug_tools.adm_scaler_end,
        a.debug_tools.refiner_swap_method,
        a.debug_tools.adaptive_cfg,
        a.debug_tools.clip_skip,
        advanced.models.base_model,
        advanced.models.refiner_model,
        advanced.models.refiner_switch,
        a.debug_tools.sampler_name,
        a.debug_tools.scheduler_name,
        a.debug_tools.vae_name,
        s.seed_random,
        s.image_seed,
        a.inpaint_advanced.inpaint_engine,
        inpaint_engine_state,
        image_input.inpaint.inpaint_mode,
    ]
    outputs += enhance_panel.enhance_inpaint_mode_ctrls
    outputs += [
        prompt_buttons.generate_button,
        prompt_buttons.load_parameter_button,
    ]
    outputs += a.freeu.freeu_ctrls
    outputs += advanced.models.lora_ctrls

    return outputs


def build_ctrls(
    currentTask,
    prompt_buttons: PromptAndButtons,
    top_checkboxes: TopCheckboxes,
    image_input: ImageInputTabs,
    enhance_panel: EnhancePanelComponents,
    advanced: AdvancedColumnComponents,
    current_tab,
):
    s = advanced.settings
    a = advanced.advanced

    ctrls = [currentTask, a.debug_tools.generate_image_grid]
    ctrls += [
        prompt_buttons.prompt,
        s.negative_prompt,
        advanced.styles.style_selections,
        s.performance_selection,
        s.aspect_ratios_selection,
        s.image_number,
        s.output_format,
        s.image_seed,
        a.debug_tools.read_wildcards_in_order,
        a.sharpness,
        a.guidance_scale,
    ]

    ctrls += [
        advanced.models.base_model,
        advanced.models.refiner_model,
        advanced.models.refiner_switch,
    ]
    ctrls += advanced.models.lora_ctrls

    ctrls += [
        top_checkboxes.input_image_checkbox,
        current_tab,
    ]
    ctrls += [
        image_input.uov_method,
        image_input.uov_input_image,
    ]
    ctrls += [
        image_input.inpaint.outpaint_selections,
        image_input.inpaint.inpaint_input_image,
        image_input.inpaint.inpaint_additional_prompt,
        image_input.inpaint.inpaint_mask_image,
    ]
    ctrls += [
        a.debug_tools.disable_preview,
        a.debug_tools.disable_intermediate_results,
        a.debug_tools.disable_seed_increment,
        a.debug_tools.black_out_nsfw,
    ]
    ctrls += [
        a.debug_tools.adm_scaler_positive,
        a.debug_tools.adm_scaler_negative,
        a.debug_tools.adm_scaler_end,
        a.debug_tools.adaptive_cfg,
        a.debug_tools.clip_skip,
    ]
    ctrls += [
        a.debug_tools.sampler_name,
        a.debug_tools.scheduler_name,
        a.debug_tools.vae_name,
    ]
    ctrls += [
        a.debug_tools.overwrite_step,
        a.debug_tools.overwrite_switch,
        a.debug_tools.overwrite_width,
        a.debug_tools.overwrite_height,
        a.debug_tools.overwrite_vary_strength,
    ]
    ctrls += [
        a.debug_tools.overwrite_upscale_strength,
        a.control.mixing_image_prompt_and_vary_upscale,
        a.control.mixing_image_prompt_and_inpaint,
    ]
    ctrls += [
        a.control.debugging_cn_preprocessor,
        a.control.skipping_cn_preprocessor,
        a.control.canny_low_threshold,
        a.control.canny_high_threshold,
    ]
    ctrls += [
        a.debug_tools.refiner_swap_method,
        a.control.controlnet_softness,
    ]
    ctrls += a.freeu.freeu_ctrls

    inpaint_ctrls = [
        a.inpaint_advanced.debugging_inpaint_preprocessor,
        a.inpaint_advanced.inpaint_disable_initial_latent,
        a.inpaint_advanced.inpaint_engine,
        a.inpaint_advanced.inpaint_strength,
        a.inpaint_advanced.inpaint_respective_field,
        image_input.inpaint.inpaint_advanced_masking_checkbox,
        image_input.inpaint.invert_mask_checkbox,
        a.inpaint_advanced.inpaint_erode_or_dilate,
    ]
    ctrls += inpaint_ctrls

    if not args_manager.args.disable_image_log:
        ctrls += [a.debug_tools.save_final_enhanced_image_only]

    if not args_manager.args.disable_metadata:
        ctrls += [
            a.debug_tools.save_metadata_to_images,
            a.debug_tools.metadata_scheme,
        ]

    ctrls += image_input.ip.ip_ctrls
    ctrls += [
        a.inpaint_advanced.debugging_dino,
        a.inpaint_advanced.dino_erode_or_dilate,
        a.inpaint_advanced.debugging_enhance_masks_checkbox,
        image_input.enhance_input_image,
        top_checkboxes.enhance_checkbox,
        enhance_panel.enhance_uov_method,
        enhance_panel.enhance_uov_processing_order,
        enhance_panel.enhance_uov_prompt_type,
    ]
    ctrls += enhance_panel.enhance_ctrls

    return ctrls
