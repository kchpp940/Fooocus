from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class PreviewComponents:
    progress_window: Any
    progress_gallery: Any
    progress_html: Any
    gallery: Any


@dataclass
class PromptAndButtons:
    prompt: Any
    generate_button: Any
    reset_button: Any
    load_parameter_button: Any
    skip_button: Any
    stop_button: Any


@dataclass
class TopCheckboxes:
    input_image_checkbox: Any
    enhance_checkbox: Any
    advanced_checkbox: Any


@dataclass
class ImagePromptComponents:
    ip_images: List[Any]
    ip_types: List[Any]
    ip_stops: List[Any]
    ip_weights: List[Any]
    ip_ctrls: List[Any]
    ip_ad_cols: List[Any]
    ip_advanced: Any


@dataclass
class InpaintComponents:
    inpaint_input_image: Any
    inpaint_advanced_masking_checkbox: Any
    inpaint_mode: Any
    inpaint_additional_prompt: Any
    outpaint_selections: Any
    example_inpaint_prompts: Any
    inpaint_mask_generation_col: Any
    inpaint_mask_image: Any
    invert_mask_checkbox: Any
    inpaint_mask_model: Any
    inpaint_mask_cloth_category: Any
    inpaint_mask_dino_prompt_text: Any
    example_inpaint_mask_dino_prompt_text: Any
    inpaint_mask_advanced_options: Any
    inpaint_mask_sam_model: Any
    inpaint_mask_box_threshold: Any
    inpaint_mask_text_threshold: Any
    inpaint_mask_sam_max_detections: Any
    generate_mask_button: Any


@dataclass
class DescribeComponents:
    describe_input_image: Any
    describe_methods: Any
    describe_apply_styles: Any
    describe_btn: Any
    describe_image_size: Any


@dataclass
class ImageInputTabs:
    uov_tab: Any
    uov_input_image: Any
    uov_method: Any
    ip_tab: Any
    ip: ImagePromptComponents
    inpaint_tab: Any
    inpaint: InpaintComponents
    describe_tab: Any
    describe: DescribeComponents
    enhance_tab: Any
    enhance_input_image: Any


@dataclass
class MetadataTab:
    metadata_tab: Any
    metadata_input_image: Any
    metadata_json: Any
    metadata_import_button: Any


@dataclass
class CompareTab:
    compare_tab: Any
    compare_image_left: Any
    compare_image_right: Any
    compare_run_button: Any
    compare_fill_diff_button: Any
    compare_fill_all_button: Any
    compare_summary_html: Any
    compare_diff_json: Any


@dataclass
class EnhancePanelComponents:
    enhance_ctrls: List[Any]
    enhance_inpaint_mode_ctrls: List[Any]
    enhance_inpaint_engine_ctrls: List[Any]
    enhance_inpaint_update_ctrls: List[Any]
    enhance_uov_method: Any
    enhance_uov_processing_order: Any
    enhance_uov_prompt_type: Any


@dataclass
class SettingsTabComponents:
    preset_selection: Optional[Any] = None
    preset_details_html: Optional[Any] = None
    new_preset_name_input: Optional[Any] = None
    save_preset_btn: Optional[Any] = None
    duplicate_preset_btn: Optional[Any] = None
    rename_preset_btn: Optional[Any] = None
    delete_preset_btn: Optional[Any] = None
    preset_operation_msg: Optional[Any] = None
    performance_selection: Any = None
    aspect_ratios_selection: Any = None
    image_number: Any = None
    output_format: Any = None
    negative_prompt: Any = None
    seed_random: Any = None
    image_seed: Any = None
    history_link: Any = None


@dataclass
class StylesTabComponents:
    style_search_bar: Any
    style_selections: Any
    gradio_receiver_style_selections: Any


@dataclass
class ModelsTabComponents:
    base_model: Any
    refiner_model: Any
    refiner_switch: Any
    lora_ctrls: List[Any]
    refresh_files: Any


@dataclass
class DebugToolsComponents:
    adm_scaler_positive: Any
    adm_scaler_negative: Any
    adm_scaler_end: Any
    refiner_swap_method: Any
    adaptive_cfg: Any
    clip_skip: Any
    sampler_name: Any
    scheduler_name: Any
    vae_name: Any
    generate_image_grid: Any
    overwrite_step: Any
    overwrite_switch: Any
    overwrite_width: Any
    overwrite_height: Any
    overwrite_vary_strength: Any
    overwrite_upscale_strength: Any
    disable_preview: Any
    disable_intermediate_results: Any
    disable_seed_increment: Any
    read_wildcards_in_order: Any
    black_out_nsfw: Any
    save_final_enhanced_image_only: Optional[Any] = None
    save_metadata_to_images: Optional[Any] = None
    metadata_scheme: Optional[Any] = None


@dataclass
class ControlTabComponents:
    debugging_cn_preprocessor: Any
    skipping_cn_preprocessor: Any
    mixing_image_prompt_and_vary_upscale: Any
    mixing_image_prompt_and_inpaint: Any
    controlnet_softness: Any
    canny_low_threshold: Any
    canny_high_threshold: Any


@dataclass
class AdvancedInpaintComponents:
    debugging_inpaint_preprocessor: Any
    debugging_enhance_masks_checkbox: Any
    debugging_dino: Any
    inpaint_disable_initial_latent: Any
    inpaint_engine: Any
    inpaint_strength: Any
    inpaint_respective_field: Any
    inpaint_erode_or_dilate: Any
    dino_erode_or_dilate: Any
    inpaint_mask_color: Any


@dataclass
class FreeUComponents:
    freeu_enabled: Any
    freeu_b1: Any
    freeu_b2: Any
    freeu_s1: Any
    freeu_s2: Any
    freeu_ctrls: List[Any]


@dataclass
class AdvancedTabComponents:
    guidance_scale: Any
    sharpness: Any
    dev_mode: Any
    dev_tools: Any
    debug_tools: DebugToolsComponents
    control: ControlTabComponents
    inpaint_advanced: AdvancedInpaintComponents
    freeu: FreeUComponents


@dataclass
class AdvancedColumnComponents:
    settings: SettingsTabComponents
    styles: StylesTabComponents
    models: ModelsTabComponents
    advanced: AdvancedTabComponents
