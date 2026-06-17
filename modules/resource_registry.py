from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict


class ResourceType(Enum):
    CHECKPOINT = "checkpoint"
    LORA = "lora"
    VAE = "vae"
    VAE_APPROX = "vae_approx"
    INPAINT = "inpaint"
    CONTROLNET = "controlnet"
    CLIP_VISION = "clip_vision"
    UPSCALE = "upscale"
    EMBEDDING = "embedding"
    SAFETY_CHECKER = "safety_checker"
    SAM = "sam"
    FOOOCUS_EXPANSION = "fooocus_expansion"


RESOURCE_TYPE_CONFIG: Dict[ResourceType, Dict] = {
    ResourceType.CHECKPOINT: {
        "path_config_key": "path_checkpoints",
        "default_path": "../models/checkpoints/",
        "extensions": ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch'],
        "is_multi_dir": True,
    },
    ResourceType.LORA: {
        "path_config_key": "path_loras",
        "default_path": "../models/loras/",
        "extensions": ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch'],
        "is_multi_dir": True,
    },
    ResourceType.VAE: {
        "path_config_key": "path_vae",
        "default_path": "../models/vae/",
        "extensions": ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch'],
        "is_multi_dir": False,
    },
    ResourceType.VAE_APPROX: {
        "path_config_key": "path_vae_approx",
        "default_path": "../models/vae_approx/",
        "extensions": ['.pth', '.ckpt', '.bin', '.safetensors'],
        "is_multi_dir": False,
    },
    ResourceType.INPAINT: {
        "path_config_key": "path_inpaint",
        "default_path": "../models/inpaint/",
        "extensions": ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch'],
        "is_multi_dir": False,
    },
    ResourceType.CONTROLNET: {
        "path_config_key": "path_controlnet",
        "default_path": "../models/controlnet/",
        "extensions": ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch'],
        "is_multi_dir": False,
    },
    ResourceType.CLIP_VISION: {
        "path_config_key": "path_clip_vision",
        "default_path": "../models/clip_vision/",
        "extensions": ['.pth', '.ckpt', '.bin', '.safetensors'],
        "is_multi_dir": False,
    },
    ResourceType.UPSCALE: {
        "path_config_key": "path_upscale_models",
        "default_path": "../models/upscale_models/",
        "extensions": ['.pth', '.ckpt', '.bin', '.safetensors'],
        "is_multi_dir": False,
    },
    ResourceType.EMBEDDING: {
        "path_config_key": "path_embeddings",
        "default_path": "../models/embeddings/",
        "extensions": ['.pth', '.ckpt', '.bin', '.safetensors'],
        "is_multi_dir": False,
    },
    ResourceType.SAFETY_CHECKER: {
        "path_config_key": "path_safety_checker",
        "default_path": "../models/safety_checker/",
        "extensions": ['.pth', '.ckpt', '.bin'],
        "is_multi_dir": False,
    },
    ResourceType.SAM: {
        "path_config_key": "path_sam",
        "default_path": "../models/sam/",
        "extensions": ['.pth', '.ckpt', '.bin'],
        "is_multi_dir": False,
    },
    ResourceType.FOOOCUS_EXPANSION: {
        "path_config_key": "path_fooocus_expansion",
        "default_path": "../models/prompt_expansion/fooocus_expansion",
        "extensions": ['.pth', '.ckpt', '.bin'],
        "is_multi_dir": False,
    },
}


@dataclass(frozen=True)
class ResourceDefinition:
    resource_id: str
    resource_type: ResourceType
    name: str
    urls: List[str] = field(default_factory=list)
    expected_hash: Optional[str] = None
    description: Optional[str] = None
    optional: bool = False
    version: Optional[str] = None


VAE_APPROX_RESOURCES: List[ResourceDefinition] = [
    ResourceDefinition(
        resource_id="vae_approx_xl",
        resource_type=ResourceType.VAE_APPROX,
        name="xlvaeapp.pth",
        urls=["https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth"],
        description="VAE Approximation for SDXL",
    ),
    ResourceDefinition(
        resource_id="vae_approx_sd15",
        resource_type=ResourceType.VAE_APPROX,
        name="vaeapp_sd15.pth",
        urls=["https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt"],
        description="VAE Approximation for SD 1.5",
    ),
    ResourceDefinition(
        resource_id="vae_approx_interposer",
        resource_type=ResourceType.VAE_APPROX,
        name="xl-to-v1_interposer-v4.0.safetensors",
        urls=["https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors"],
        description="XL to V1 Interposer",
    ),
]


INPAINT_RESOURCES: Dict[str, List[ResourceDefinition]] = {
    "v1": [
        ResourceDefinition(
            resource_id="inpaint_head",
            resource_type=ResourceType.INPAINT,
            name="fooocus_inpaint_head.pth",
            urls=["https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"],
            description="Fooocus Inpaint Head Model",
        ),
        ResourceDefinition(
            resource_id="inpaint_patch_v1",
            resource_type=ResourceType.INPAINT,
            name="inpaint.fooocus.patch",
            urls=["https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch"],
            description="Fooocus Inpaint Patch v1",
            version="v1",
        ),
    ],
    "v2.5": [
        ResourceDefinition(
            resource_id="inpaint_head",
            resource_type=ResourceType.INPAINT,
            name="fooocus_inpaint_head.pth",
            urls=["https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"],
            description="Fooocus Inpaint Head Model",
        ),
        ResourceDefinition(
            resource_id="inpaint_patch_v25",
            resource_type=ResourceType.INPAINT,
            name="inpaint_v25.fooocus.patch",
            urls=["https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch"],
            description="Fooocus Inpaint Patch v2.5",
            version="v2.5",
        ),
    ],
    "v2.6": [
        ResourceDefinition(
            resource_id="inpaint_head",
            resource_type=ResourceType.INPAINT,
            name="fooocus_inpaint_head.pth",
            urls=["https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"],
            description="Fooocus Inpaint Head Model",
        ),
        ResourceDefinition(
            resource_id="inpaint_patch_v26",
            resource_type=ResourceType.INPAINT,
            name="inpaint_v26.fooocus.patch",
            urls=["https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch"],
            description="Fooocus Inpaint Patch v2.6",
            version="v2.6",
        ),
    ],
}


PERFORMANCE_LORA_RESOURCES: List[ResourceDefinition] = [
    ResourceDefinition(
        resource_id="lora_lcm",
        resource_type=ResourceType.LORA,
        name="sdxl_lcm_lora.safetensors",
        urls=["https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors"],
        description="SDXL LCM LoRA for Extreme Speed",
    ),
    ResourceDefinition(
        resource_id="lora_lightning",
        resource_type=ResourceType.LORA,
        name="sdxl_lightning_4step_lora.safetensors",
        urls=["https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors"],
        description="SDXL Lightning 4-step LoRA",
    ),
    ResourceDefinition(
        resource_id="lora_hyper_sd",
        resource_type=ResourceType.LORA,
        name="sdxl_hyper_sd_4step_lora.safetensors",
        urls=["https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors"],
        description="SDXL Hyper-SD 4-step LoRA",
    ),
]


CONTROLNET_RESOURCES: List[ResourceDefinition] = [
    ResourceDefinition(
        resource_id="controlnet_canny",
        resource_type=ResourceType.CONTROLNET,
        name="control-lora-canny-rank128.safetensors",
        urls=["https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors"],
        description="ControlNet Canny Model",
    ),
    ResourceDefinition(
        resource_id="controlnet_cpds",
        resource_type=ResourceType.CONTROLNET,
        name="fooocus_xl_cpds_128.safetensors",
        urls=["https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors"],
        description="Fooocus XL CPDS ControlNet",
    ),
]


IP_ADAPTER_RESOURCES: Dict[str, List[ResourceDefinition]] = {
    "ip": [
        ResourceDefinition(
            resource_id="clip_vision_vit_h",
            resource_type=ResourceType.CLIP_VISION,
            name="clip_vision_vit_h.safetensors",
            urls=["https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors"],
            description="CLIP Vision ViT-H Model",
        ),
        ResourceDefinition(
            resource_id="ip_negative",
            resource_type=ResourceType.CONTROLNET,
            name="fooocus_ip_negative.safetensors",
            urls=["https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors"],
            description="Fooocus IP Negative Embedding",
        ),
        ResourceDefinition(
            resource_id="ip_adapter_plus",
            resource_type=ResourceType.CONTROLNET,
            name="ip-adapter-plus_sdxl_vit-h.bin",
            urls=["https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin"],
            description="IP-Adapter Plus SDXL",
        ),
    ],
    "face": [
        ResourceDefinition(
            resource_id="clip_vision_vit_h",
            resource_type=ResourceType.CLIP_VISION,
            name="clip_vision_vit_h.safetensors",
            urls=["https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors"],
            description="CLIP Vision ViT-H Model",
        ),
        ResourceDefinition(
            resource_id="ip_negative",
            resource_type=ResourceType.CONTROLNET,
            name="fooocus_ip_negative.safetensors",
            urls=["https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors"],
            description="Fooocus IP Negative Embedding",
        ),
        ResourceDefinition(
            resource_id="ip_adapter_face",
            resource_type=ResourceType.CONTROLNET,
            name="ip-adapter-plus-face_sdxl_vit-h.bin",
            urls=["https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin"],
            description="IP-Adapter Plus Face SDXL",
        ),
    ],
}


UPSCALE_RESOURCES: List[ResourceDefinition] = [
    ResourceDefinition(
        resource_id="upscaler_fooocus",
        resource_type=ResourceType.UPSCALE,
        name="fooocus_upscaler_s409985e5.bin",
        urls=["https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin"],
        description="Fooocus Upscaler Model",
    ),
]


SAFETY_CHECKER_RESOURCES: List[ResourceDefinition] = [
    ResourceDefinition(
        resource_id="safety_checker_stable_diffusion",
        resource_type=ResourceType.SAFETY_CHECKER,
        name="stable-diffusion-safety-checker.bin",
        urls=["https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin"],
        description="Stable Diffusion Safety Checker",
    ),
]


SAM_RESOURCES: Dict[str, ResourceDefinition] = {
    "vit_b": ResourceDefinition(
        resource_id="sam_vit_b",
        resource_type=ResourceType.SAM,
        name="sam_vit_b_01ec64.pth",
        urls=["https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth"],
        description="SAM ViT-B Model",
    ),
    "vit_l": ResourceDefinition(
        resource_id="sam_vit_l",
        resource_type=ResourceType.SAM,
        name="sam_vit_l_0b3195.pth",
        urls=["https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth"],
        description="SAM ViT-L Model",
    ),
    "vit_h": ResourceDefinition(
        resource_id="sam_vit_h",
        resource_type=ResourceType.SAM,
        name="sam_vit_h_4b8939.pth",
        urls=["https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth"],
        description="SAM ViT-H Model",
    ),
}


FOOOCUS_EXPANSION_RESOURCES: List[ResourceDefinition] = [
    ResourceDefinition(
        resource_id="fooocus_expansion",
        resource_type=ResourceType.FOOOCUS_EXPANSION,
        name="pytorch_model.bin",
        urls=["https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin"],
        description="Fooocus Prompt Expansion Model",
    ),
]


RESOURCE_INDEX: Dict[str, ResourceDefinition] = {}


def _build_index():
    all_resources = []
    all_resources.extend(VAE_APPROX_RESOURCES)
    all_resources.extend(PERFORMANCE_LORA_RESOURCES)
    all_resources.extend(CONTROLNET_RESOURCES)
    all_resources.extend(UPSCALE_RESOURCES)
    all_resources.extend(SAFETY_CHECKER_RESOURCES)
    all_resources.extend(FOOOCUS_EXPANSION_RESOURCES)
    all_resources.extend(SAM_RESOURCES.values())
    
    for version_resources in INPAINT_RESOURCES.values():
        all_resources.extend(version_resources)
    for type_resources in IP_ADAPTER_RESOURCES.values():
        all_resources.extend(type_resources)
    
    for resource in all_resources:
        if resource.resource_id not in RESOURCE_INDEX:
            RESOURCE_INDEX[resource.resource_id] = resource


_build_index()


def get_resource_definition(resource_id: str) -> Optional[ResourceDefinition]:
    return RESOURCE_INDEX.get(resource_id)


def get_resources_by_type(resource_type: ResourceType) -> List[ResourceDefinition]:
    return [r for r in RESOURCE_INDEX.values() if r.resource_type == resource_type]


def get_resource_type_config(resource_type: ResourceType) -> Dict:
    return RESOURCE_TYPE_CONFIG.get(resource_type, {})
