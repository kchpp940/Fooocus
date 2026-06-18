from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
import copy


@dataclass
class GenerationContext:
    performance_selection: Any = None
    base_model_name: str = ''
    refiner_model_name: str = ''
    vae_name: str = ''
    sampler_name: str = ''
    scheduler_name: str = ''
    final_scheduler_name: str = ''
    steps: int = 0
    original_steps: int = 0
    switch: int = 0
    width: int = 0
    height: int = 0
    aspect_ratios_selection: str = '1024×1024'
    cfg_scale: float = 0.0
    sharpness: float = 0.0
    adm_scaler_positive: float = 0.0
    adm_scaler_negative: float = 0.0
    adm_scaler_end: float = 0.0
    adaptive_cfg: float = 0.0
    clip_skip: int = 0
    refiner_switch: float = 0.0
    refiner_swap_method: str = ''
    seed: int = 0
    image_number: int = 0
    output_format: str = ''
    prompt: str = ''
    negative_prompt: str = ''
    style_selections: List[str] = field(default_factory=list)
    loras: List[Tuple[str, float]] = field(default_factory=list)
    performance_loras: List[Tuple[str, float]] = field(default_factory=list)
    use_expansion: bool = False
    use_style: bool = False
    use_synthetic_refiner: bool = False
    disable_seed_increment: bool = False
    read_wildcards_in_order: bool = False
    freeu_enabled: bool = False
    freeu_b1: float = 0.0
    freeu_b2: float = 0.0
    freeu_s1: float = 0.0
    freeu_s2: float = 0.0
    black_out_nsfw: bool = False
    save_metadata_to_images: bool = False
    metadata_scheme: Any = None
    save_final_enhanced_image_only: bool = False
    disable_preview: bool = False
    disable_intermediate_results: bool = False
    current_tab: str = ''
    uov_method: str = ''
    uov_input_image: Any = None
    outpaint_selections: List[str] = field(default_factory=list)
    inpaint_input_image: Any = None
    inpaint_additional_prompt: str = ''
    inpaint_mask_image_upload: Any = None
    inpaint_engine: str = 'None'
    inpaint_strength: float = 0.0
    inpaint_respective_field: float = 0.0
    inpaint_disable_initial_latent: bool = False
    inpaint_advanced_masking_checkbox: bool = False
    invert_mask_checkbox: bool = False
    inpaint_erode_or_dilate: int = 0
    debugging_inpaint_preprocessor: bool = False
    cn_tasks: Dict[str, List[List[Any]]] = field(default_factory=dict)
    debugging_cn_preprocessor: bool = False
    skipping_cn_preprocessor: bool = False
    canny_low_threshold: float = 0.0
    canny_high_threshold: float = 0.0
    controlnet_softness: float = 0.0
    mixing_image_prompt_and_vary_upscale: bool = False
    mixing_image_prompt_and_inpaint: bool = False
    input_image_checkbox: bool = False
    overwrite_step: int = 0
    overwrite_switch: float = 0.0
    overwrite_width: int = 0
    overwrite_height: int = 0
    overwrite_vary_strength: float = 0.0
    overwrite_upscale_strength: float = 0.0
    enhance_checkbox: bool = False
    enhance_input_image: Any = None
    enhance_uov_method: str = ''
    enhance_uov_processing_order: str = ''
    enhance_uov_prompt_type: str = ''
    enhance_ctrls: List[List[Any]] = field(default_factory=list)
    should_enhance: bool = False
    debugging_dino: bool = False
    dino_erode_or_dilate: int = 0
    debugging_enhance_masks_checkbox: bool = False
    denoising_strength: float = 1.0
    tiled: bool = False
    initial_latent: Any = None
    goals: List[str] = field(default_factory=list)
    base_model_additional_loras: List[Tuple[str, float]] = field(default_factory=list)
    inpaint_head_model_path: Optional[str] = None
    inpaint_image: Any = None
    inpaint_mask: Any = None
    inpaint_parameterized: bool = False
    skip_prompt_processing: bool = False
    controlnet_canny_path: Optional[str] = None
    controlnet_cpds_path: Optional[str] = None
    clip_vision_path: Optional[str] = None
    ip_negative_path: Optional[str] = None
    ip_adapter_path: Optional[str] = None
    ip_adapter_face_path: Optional[str] = None

    def clone(self) -> 'GenerationContext':
        new_ctx = GenerationContext()
        for k, v in self.__dict__.items():
            if isinstance(v, (list, dict)):
                setattr(new_ctx, k, copy.deepcopy(v))
            else:
                setattr(new_ctx, k, v)
        return new_ctx


@dataclass
class SingleTaskContext:
    task_seed: int = 0
    task_prompt: str = ''
    task_negative_prompt: str = ''
    positive: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)
    expansion: str = ''
    c: Any = None
    uc: Any = None
    positive_top_k: int = 0
    negative_top_k: int = 0
    log_positive_prompt: str = ''
    log_negative_prompt: str = ''
    styles: List[str] = field(default_factory=list)


@dataclass
class EnhanceCtrl:
    mask_dino_prompt_text: str = ''
    prompt: str = ''
    negative_prompt: str = ''
    mask_model: str = ''
    mask_cloth_category: str = ''
    mask_sam_model: str = ''
    mask_text_threshold: float = 0.0
    mask_box_threshold: float = 0.0
    mask_sam_max_detections: int = 0
    inpaint_disable_initial_latent: bool = False
    inpaint_engine: str = 'None'
    inpaint_strength: float = 0.0
    inpaint_respective_field: float = 0.0
    inpaint_erode_or_dilate: int = 0
    mask_invert: bool = False


@dataclass
class WorkerRuntime:
    pid: int
    pipeline: Any
    inpaint_worker: Any
    flags: Any
    ldm_model_management: Any
    ip_adapter: Any
    default_censor: Callable
    fooocus_expansion: str
    log: Callable
    time_module: Any
