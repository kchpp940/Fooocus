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
    known_urls={
        "xlvaeapp.pth": "https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth",
        "vaeapp_sd15.pth": "https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt",
        "xl-to-v1_interposer-v4.0.safetensors": "https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors",
    },
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
    known_urls={
        "fooocus_inpaint_head.pth": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
        "inpaint.fooocus.patch": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch",
        "inpaint_v25.fooocus.patch": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch",
        "inpaint_v26.fooocus.patch": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch",
    },
)

CONTROLNET = ResourceType(
    category=ResourceCategory.CONTROLNET,
    config_key="path_controlnet",
    default_relative_paths=["../models/controlnet/"],
    is_array_path=False,
    extensions=[".safetensors", ".bin"],
    known_urls={
        "control-lora-canny-rank128.safetensors": "https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors",
        "fooocus_xl_cpds_128.safetensors": "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors",
        "fooocus_ip_negative.safetensors": "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors",
        "ip-adapter-plus_sdxl_vit-h.bin": "https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin",
        "ip-adapter-plus-face_sdxl_vit-h.bin": "https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin",
    },
)

CLIP_VISION = ResourceType(
    category=ResourceCategory.CLIP_VISION,
    config_key="path_clip_vision",
    default_relative_paths=["../models/clip_vision/"],
    is_array_path=False,
    extensions=[".safetensors", ".bin"],
    known_urls={
        "clip_vision_vit_h.safetensors": "https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors",
    },
)

UPSCALE_MODEL = ResourceType(
    category=ResourceCategory.UPSCALE_MODEL,
    config_key="path_upscale_models",
    default_relative_paths=["../models/upscale_models/"],
    is_array_path=False,
    extensions=[".pth", ".bin"],
    known_urls={
        "fooocus_upscaler_s409985e5.bin": "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
    },
)

FOOOCUS_EXPANSION = ResourceType(
    category=ResourceCategory.FOOOCUS_EXPANSION,
    config_key="path_fooocus_expansion",
    default_relative_paths=["../models/prompt_expansion/fooocus_expansion"],
    is_array_path=False,
    extensions=[".bin"],
    known_urls={
        "pytorch_model.bin": "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
    },
)

SAFETY_CHECKER = ResourceType(
    category=ResourceCategory.SAFETY_CHECKER,
    config_key="path_safety_checker",
    default_relative_paths=["../models/safety_checker/"],
    is_array_path=False,
    extensions=[".bin"],
    known_urls={
        "stable-diffusion-safety-checker.bin": "https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin",
    },
)

SAM = ResourceType(
    category=ResourceCategory.SAM,
    config_key="path_sam",
    default_relative_paths=["../models/sam/"],
    is_array_path=False,
    extensions=[".pth"],
    known_urls={
        "sam_vit_b_01ec64.pth": "https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth",
        "sam_vit_l_0b3195.pth": "https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth",
        "sam_vit_h_4b8939.pth": "https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth",
    },
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
