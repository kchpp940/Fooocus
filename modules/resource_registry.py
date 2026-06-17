from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class ResourceCategory(Enum):
    CHECKPOINT = "checkpoint"
    LORA = "lora"
    VAE = "vae"
    VAE_APPROX = "vae_approx"
    EMBEDDING = "embedding"
    INPAINT = "inpaint"
    CONTROLNET = "controlnet"
    CLIP_VISION = "clip_vision"
    UPSCALE_MODEL = "upscale_model"
    FOOOCUS_EXPANSION = "fooocus_expansion"
    SAFETY_CHECKER = "safety_checker"
    SAM = "sam"
    WILDCARD = "wildcard"


@dataclass(frozen=True)
class ResourceType:
    category: ResourceCategory
    config_key: str
    default_relative_paths: List[str]
    is_array_path: bool
    extensions: List[str]
    known_urls: Dict[str, str] = field(default_factory=dict)
    name_filter: Optional[str] = None
    make_directory: bool = True

    @property
    def registry_id(self) -> str:
        return self.category.value


@dataclass(frozen=True)
class ResourceDefinition:
    resource_id: str
    category: ResourceCategory
    filename: str
    url: str
    required: bool = False
    expected_hash: Optional[str] = None
    group: Optional[str] = None
    description: Optional[str] = None

    @property
    def registry_id(self) -> str:
        return f"{self.category.value}:{self.resource_id}"


CHECKPOINT = ResourceType(
    category=ResourceCategory.CHECKPOINT,
    config_key="path_checkpoints",
    default_relative_paths=["../models/checkpoints/"],
    is_array_path=True,
    extensions=[".pth", ".ckpt", ".bin", ".safetensors", ".fooocus.patch"],
)

LORA = ResourceType(
    category=ResourceCategory.LORA,
    config_key="path_loras",
    default_relative_paths=["../models/loras/"],
    is_array_path=True,
    extensions=[".pth", ".ckpt", ".bin", ".safetensors", ".fooocus.patch"],
)

VAE = ResourceType(
    category=ResourceCategory.VAE,
    config_key="path_vae",
    default_relative_paths=["../models/vae/"],
    is_array_path=False,
    extensions=[".pth", ".ckpt", ".bin", ".safetensors"],
)

VAE_APPROX = ResourceType(
    category=ResourceCategory.VAE_APPROX,
    config_key="path_vae_approx",
    default_relative_paths=["../models/vae_approx/"],
    is_array_path=False,
    extensions=[".pth", ".safetensors"],
)

EMBEDDING = ResourceType(
    category=ResourceCategory.EMBEDDING,
    config_key="path_embeddings",
    default_relative_paths=["../models/embeddings/"],
    is_array_path=False,
    extensions=[".pt", ".bin", ".safetensors"],
)

INPAINT = ResourceType(
    category=ResourceCategory.INPAINT,
    config_key="path_inpaint",
    default_relative_paths=["../models/inpaint/"],
    is_array_path=False,
    extensions=[".pth", ".fooocus.patch"],
)

CONTROLNET = ResourceType(
    category=ResourceCategory.CONTROLNET,
    config_key="path_controlnet",
    default_relative_paths=["../models/controlnet/"],
    is_array_path=False,
    extensions=[".safetensors", ".bin"],
)

CLIP_VISION = ResourceType(
    category=ResourceCategory.CLIP_VISION,
    config_key="path_clip_vision",
    default_relative_paths=["../models/clip_vision/"],
    is_array_path=False,
    extensions=[".safetensors", ".bin"],
)

UPSCALE_MODEL = ResourceType(
    category=ResourceCategory.UPSCALE_MODEL,
    config_key="path_upscale_models",
    default_relative_paths=["../models/upscale_models/"],
    is_array_path=False,
    extensions=[".pth", ".bin"],
)

FOOOCUS_EXPANSION = ResourceType(
    category=ResourceCategory.FOOOCUS_EXPANSION,
    config_key="path_fooocus_expansion",
    default_relative_paths=["../models/prompt_expansion/fooocus_expansion"],
    is_array_path=False,
    extensions=[".bin"],
)

SAFETY_CHECKER = ResourceType(
    category=ResourceCategory.SAFETY_CHECKER,
    config_key="path_safety_checker",
    default_relative_paths=["../models/safety_checker/"],
    is_array_path=False,
    extensions=[".bin"],
)

SAM = ResourceType(
    category=ResourceCategory.SAM,
    config_key="path_sam",
    default_relative_paths=["../models/sam/"],
    is_array_path=False,
    extensions=[".pth"],
)

WILDCARD = ResourceType(
    category=ResourceCategory.WILDCARD,
    config_key="path_wildcards",
    default_relative_paths=["../wildcards/"],
    is_array_path=False,
    extensions=[".txt"],
    make_directory=True,
)


ALL_RESOURCE_TYPES: List[ResourceType] = [
    CHECKPOINT,
    LORA,
    VAE,
    VAE_APPROX,
    EMBEDDING,
    INPAINT,
    CONTROLNET,
    CLIP_VISION,
    UPSCALE_MODEL,
    FOOOCUS_EXPANSION,
    SAFETY_CHECKER,
    SAM,
    WILDCARD,
]

RESOURCE_TYPE_MAP: Dict[ResourceCategory, ResourceType] = {rt.category: rt for rt in ALL_RESOURCE_TYPES}

HASHED_CATEGORIES = {ResourceCategory.CHECKPOINT, ResourceCategory.LORA}

INPAINT_VERSION_PATCH_URLS: Dict[str, str] = {
    "v1": "inpaint.fooocus.patch",
    "v2.5": "inpaint_v25.fooocus.patch",
    "v2.6": "inpaint_v26.fooocus.patch",
}

PERFORMANCE_LORA_URLS: Dict[str, Tuple[str, str]] = {
    "EXTREME_SPEED": ("sdxl_lcm_lora.safetensors", "https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors"),
    "LIGHTNING": ("sdxl_lightning_4step_lora.safetensors", "https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors"),
    "HYPER_SD": ("sdxl_hyper_sd_4step_lora.safetensors", "https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors"),
}

IP_ADAPTER_URLS: Dict[str, List[Tuple[str, str, ResourceCategory]]] = {
    "ip": [
        ("clip_vision_vit_h.safetensors", "https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors", ResourceCategory.CLIP_VISION),
        ("fooocus_ip_negative.safetensors", "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors", ResourceCategory.CONTROLNET),
        ("ip-adapter-plus_sdxl_vit-h.bin", "https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin", ResourceCategory.CONTROLNET),
    ],
    "face": [
        ("clip_vision_vit_h.safetensors", "https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors", ResourceCategory.CLIP_VISION),
        ("fooocus_ip_negative.safetensors", "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors", ResourceCategory.CONTROLNET),
        ("ip-adapter-plus-face_sdxl_vit-h.bin", "https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin", ResourceCategory.CONTROLNET),
    ],
}


BUILTIN_RESOURCE_DEFINITIONS: List[ResourceDefinition] = [
    ResourceDefinition(
        resource_id="vae_approx_xlvaeapp",
        category=ResourceCategory.VAE_APPROX,
        filename="xlvaeapp.pth",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth",
        required=True,
        group="vae_approx",
        description="XL VAE Approx",
    ),
    ResourceDefinition(
        resource_id="vae_approx_sd15",
        category=ResourceCategory.VAE_APPROX,
        filename="vaeapp_sd15.pth",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt",
        required=True,
        group="vae_approx",
        description="SD1.5 VAE Approx",
    ),
    ResourceDefinition(
        resource_id="vae_approx_interposer",
        category=ResourceCategory.VAE_APPROX,
        filename="xl-to-v1_interposer-v4.0.safetensors",
        url="https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors",
        required=True,
        group="vae_approx",
        description="XL-to-V1 Interposer",
    ),
    ResourceDefinition(
        resource_id="expansion",
        category=ResourceCategory.FOOOCUS_EXPANSION,
        filename="pytorch_model.bin",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
        required=True,
        group="expansion",
        description="Fooocus Prompt Expansion",
    ),
    ResourceDefinition(
        resource_id="inpaint_head",
        category=ResourceCategory.INPAINT,
        filename="fooocus_inpaint_head.pth",
        url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
        required=True,
        group="inpaint",
        description="Inpaint Head",
    ),
    ResourceDefinition(
        resource_id="inpaint_patch_v1",
        category=ResourceCategory.INPAINT,
        filename="inpaint.fooocus.patch",
        url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch",
        required=False,
        group="inpaint_v1",
        description="Inpaint Patch v1",
    ),
    ResourceDefinition(
        resource_id="inpaint_patch_v25",
        category=ResourceCategory.INPAINT,
        filename="inpaint_v25.fooocus.patch",
        url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch",
        required=False,
        group="inpaint_v25",
        description="Inpaint Patch v2.5",
    ),
    ResourceDefinition(
        resource_id="inpaint_patch_v26",
        category=ResourceCategory.INPAINT,
        filename="inpaint_v26.fooocus.patch",
        url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch",
        required=False,
        group="inpaint_v26",
        description="Inpaint Patch v2.6",
    ),
    ResourceDefinition(
        resource_id="controlnet_canny",
        category=ResourceCategory.CONTROLNET,
        filename="control-lora-canny-rank128.safetensors",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors",
        required=False,
        group="controlnet_canny",
        description="ControlNet Canny",
    ),
    ResourceDefinition(
        resource_id="controlnet_cpds",
        category=ResourceCategory.CONTROLNET,
        filename="fooocus_xl_cpds_128.safetensors",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors",
        required=False,
        group="controlnet_cpds",
        description="ControlNet CPDS",
    ),
    ResourceDefinition(
        resource_id="ip_negative",
        category=ResourceCategory.CONTROLNET,
        filename="fooocus_ip_negative.safetensors",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors",
        required=False,
        group="ip_adapter",
        description="IP Adapter Negative",
    ),
    ResourceDefinition(
        resource_id="ip_adapter_plus",
        category=ResourceCategory.CONTROLNET,
        filename="ip-adapter-plus_sdxl_vit-h.bin",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin",
        required=False,
        group="ip_adapter",
        description="IP Adapter Plus",
    ),
    ResourceDefinition(
        resource_id="ip_adapter_face",
        category=ResourceCategory.CONTROLNET,
        filename="ip-adapter-plus-face_sdxl_vit-h.bin",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin",
        required=False,
        group="ip_adapter_face",
        description="IP Adapter Face",
    ),
    ResourceDefinition(
        resource_id="clip_vision_vit_h",
        category=ResourceCategory.CLIP_VISION,
        filename="clip_vision_vit_h.safetensors",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors",
        required=False,
        group="ip_adapter",
        description="CLIP Vision ViT-H",
    ),
    ResourceDefinition(
        resource_id="upscale",
        category=ResourceCategory.UPSCALE_MODEL,
        filename="fooocus_upscaler_s409985e5.bin",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
        required=False,
        group="upscale",
        description="Fooocus Upscaler",
    ),
    ResourceDefinition(
        resource_id="safety_checker",
        category=ResourceCategory.SAFETY_CHECKER,
        filename="stable-diffusion-safety-checker.bin",
        url="https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin",
        required=False,
        group="safety",
        description="Safety Checker",
    ),
    ResourceDefinition(
        resource_id="sam_vit_b",
        category=ResourceCategory.SAM,
        filename="sam_vit_b_01ec64.pth",
        url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth",
        required=False,
        group="sam",
        description="SAM ViT-B",
    ),
    ResourceDefinition(
        resource_id="sam_vit_l",
        category=ResourceCategory.SAM,
        filename="sam_vit_l_0b3195.pth",
        url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth",
        required=False,
        group="sam",
        description="SAM ViT-L",
    ),
    ResourceDefinition(
        resource_id="sam_vit_h",
        category=ResourceCategory.SAM,
        filename="sam_vit_h_4b8939.pth",
        url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth",
        required=False,
        group="sam",
        description="SAM ViT-H",
    ),
    ResourceDefinition(
        resource_id="lora_lcm",
        category=ResourceCategory.LORA,
        filename="sdxl_lcm_lora.safetensors",
        url="https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors",
        required=False,
        group="performance_extreme",
        description="LCM LoRA (Extreme Speed)",
    ),
    ResourceDefinition(
        resource_id="lora_lightning",
        category=ResourceCategory.LORA,
        filename="sdxl_lightning_4step_lora.safetensors",
        url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors",
        required=False,
        group="performance_lightning",
        description="Lightning LoRA",
    ),
    ResourceDefinition(
        resource_id="lora_hyper_sd",
        category=ResourceCategory.LORA,
        filename="sdxl_hyper_sd_4step_lora.safetensors",
        url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors",
        required=False,
        group="performance_hyper_sd",
        description="Hyper-SD LoRA",
    ),
]

BUILTIN_DEFINITIONS_MAP: Dict[str, ResourceDefinition] = {d.resource_id: d for d in BUILTIN_RESOURCE_DEFINITIONS}

CATEGORY_LABELS: Dict[ResourceCategory, str] = {
    ResourceCategory.CHECKPOINT: "Checkpoints",
    ResourceCategory.LORA: "LoRA",
    ResourceCategory.VAE: "VAE",
    ResourceCategory.VAE_APPROX: "VAE Approx",
    ResourceCategory.EMBEDDING: "Embeddings",
    ResourceCategory.INPAINT: "Inpaint",
    ResourceCategory.CONTROLNET: "ControlNet",
    ResourceCategory.CLIP_VISION: "CLIP Vision",
    ResourceCategory.UPSCALE_MODEL: "Upscale Models",
    ResourceCategory.FOOOCUS_EXPANSION: "Prompt Expansion",
    ResourceCategory.SAFETY_CHECKER: "Safety Checker",
    ResourceCategory.SAM: "SAM",
    ResourceCategory.WILDCARD: "Wildcards",
}
