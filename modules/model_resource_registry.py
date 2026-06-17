import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


class ResourceType(Enum):
    CHECKPOINT = "checkpoint"
    LORA = "lora"
    VAE = "vae"
    CONTROLNET = "controlnet"
    INPAINT = "inpaint"
    VAE_APPROX = "vae_approx"
    CLIP_VISION = "clip_vision"
    UPSCALE = "upscale"
    SAFETY_CHECKER = "safety_checker"
    SAM = "sam"
    EXPANSION = "expansion"
    EMBEDDING = "embedding"


@dataclass
class ResourceDef:
    name: str
    resource_type: ResourceType
    filename: str
    source_url: Optional[str] = None
    dir_key: str = ""
    expected_hash: Optional[str] = None
    is_builtin: bool = False
    autodownload: bool = False
    description: str = ""

    @property
    def key(self) -> str:
        return f"{self.resource_type.value}:{self.filename}"


_registry: Dict[str, ResourceDef] = {}
_initialized = False


def register(rdef: ResourceDef):
    _registry[rdef.key] = rdef


def _init_registry():
    global _initialized
    if _initialized:
        return

    register(ResourceDef(
        name="VAE Approx XL",
        resource_type=ResourceType.VAE_APPROX,
        filename="xlvaeapp.pth",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth",
        dir_key="path_vae_approx",
        is_builtin=True,
        autodownload=True,
        description="SDXL VAE Approximation Model",
    ))

    register(ResourceDef(
        name="VAE Approx SD1.5",
        resource_type=ResourceType.VAE_APPROX,
        filename="vaeapp_sd15.pth",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt",
        dir_key="path_vae_approx",
        is_builtin=True,
        autodownload=True,
        description="SD 1.5 VAE Approximation Model",
    ))

    register(ResourceDef(
        name="XL-to-V1 Interposer v4.0",
        resource_type=ResourceType.VAE_APPROX,
        filename="xl-to-v1_interposer-v4.0.safetensors",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors",
        dir_key="path_vae_approx",
        is_builtin=True,
        autodownload=True,
        description="XL to SD 1.5 VAE Interposer",
    ))

    register(ResourceDef(
        name="Fooocus Prompt Expansion",
        resource_type=ResourceType.EXPANSION,
        filename="pytorch_model.bin",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
        dir_key="path_fooocus_expansion",
        is_builtin=True,
        autodownload=True,
        description="Fooocus prompt expansion model",
    ))

    register(ResourceDef(
        name="Inpaint Head Model",
        resource_type=ResourceType.INPAINT,
        filename="fooocus_inpaint_head.pth",
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
        dir_key="path_inpaint",
        is_builtin=True,
        description="Fooocus inpaint head (shared by all inpaint engines)",
    ))

    register(ResourceDef(
        name="Inpaint Patch v1",
        resource_type=ResourceType.INPAINT,
        filename="inpaint.fooocus.patch",
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch",
        dir_key="path_inpaint",
        is_builtin=True,
        description="Inpaint engine v1 patch",
    ))

    register(ResourceDef(
        name="Inpaint Patch v2.5",
        resource_type=ResourceType.INPAINT,
        filename="inpaint_v25.fooocus.patch",
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch",
        dir_key="path_inpaint",
        is_builtin=True,
        description="Inpaint engine v2.5 patch",
    ))

    register(ResourceDef(
        name="Inpaint Patch v2.6",
        resource_type=ResourceType.INPAINT,
        filename="inpaint_v26.fooocus.patch",
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch",
        dir_key="path_inpaint",
        is_builtin=True,
        description="Inpaint engine v2.6 patch (latest)",
    ))

    register(ResourceDef(
        name="ControlNet Canny (rank128)",
        resource_type=ResourceType.CONTROLNET,
        filename="control-lora-canny-rank128.safetensors",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors",
        dir_key="path_controlnet",
        is_builtin=True,
        description="ControlNet Canny LoRA",
    ))

    register(ResourceDef(
        name="ControlNet CPDS",
        resource_type=ResourceType.CONTROLNET,
        filename="fooocus_xl_cpds_128.safetensors",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors",
        dir_key="path_controlnet",
        is_builtin=True,
        description="Fooocus CPDS ControlNet",
    ))

    register(ResourceDef(
        name="CLIP Vision ViT-H",
        resource_type=ResourceType.CLIP_VISION,
        filename="clip_vision_vit_h.safetensors",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors",
        dir_key="path_clip_vision",
        is_builtin=True,
        description="CLIP Vision ViT-H for IP-Adapter",
    ))

    register(ResourceDef(
        name="IP-Adapter Negative",
        resource_type=ResourceType.CONTROLNET,
        filename="fooocus_ip_negative.safetensors",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors",
        dir_key="path_controlnet",
        is_builtin=True,
        description="IP-Adapter negative conditioning",
    ))

    register(ResourceDef(
        name="IP-Adapter Plus SDXL",
        resource_type=ResourceType.CONTROLNET,
        filename="ip-adapter-plus_sdxl_vit-h.bin",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin",
        dir_key="path_controlnet",
        is_builtin=True,
        description="IP-Adapter Plus for SDXL",
    ))

    register(ResourceDef(
        name="IP-Adapter Plus Face SDXL",
        resource_type=ResourceType.CONTROLNET,
        filename="ip-adapter-plus-face_sdxl_vit-h.bin",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin",
        dir_key="path_controlnet",
        is_builtin=True,
        description="IP-Adapter Plus Face for SDXL",
    ))

    register(ResourceDef(
        name="Fooocus Upscaler",
        resource_type=ResourceType.UPSCALE,
        filename="fooocus_upscaler_s409985e5.bin",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
        dir_key="path_upscale_models",
        is_builtin=True,
        description="Fooocus 4x upscaler model",
    ))

    register(ResourceDef(
        name="Stable Diffusion Safety Checker",
        resource_type=ResourceType.SAFETY_CHECKER,
        filename="stable-diffusion-safety-checker.bin",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin",
        dir_key="path_safety_checker",
        is_builtin=True,
        description="SD safety checker model",
    ))

    register(ResourceDef(
        name="SAM ViT-B",
        resource_type=ResourceType.SAM,
        filename="sam_vit_b_01ec64.pth",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth",
        dir_key="path_sam",
        is_builtin=True,
        description="Segment Anything Model ViT-Base",
    ))

    register(ResourceDef(
        name="SAM ViT-L",
        resource_type=ResourceType.SAM,
        filename="sam_vit_l_0b3195.pth",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth",
        dir_key="path_sam",
        is_builtin=True,
        description="Segment Anything Model ViT-Large",
    ))

    register(ResourceDef(
        name="SAM ViT-H",
        resource_type=ResourceType.SAM,
        filename="sam_vit_h_4b8939.pth",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth",
        dir_key="path_sam",
        is_builtin=True,
        description="Segment Anything Model ViT-Huge",
    ))

    try:
        from modules.flags import PerformanceLoRA
        register(ResourceDef(
            name="LCM LoRA (Extreme Speed)",
            resource_type=ResourceType.LORA,
            filename=PerformanceLoRA.EXTREME_SPEED.value,
            source_url="https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors",
            dir_key="paths_loras",
            is_builtin=True,
            description="Latency Consistency Model LoRA for extreme speed",
        ))
        register(ResourceDef(
            name="Lightning LoRA (4-step)",
            resource_type=ResourceType.LORA,
            filename=PerformanceLoRA.LIGHTNING.value,
            source_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors",
            dir_key="paths_loras",
            is_builtin=True,
            description="Lightning 4-step LoRA for fast inference",
        ))
        register(ResourceDef(
            name="Hyper-SD LoRA (4-step)",
            resource_type=ResourceType.LORA,
            filename=PerformanceLoRA.HYPER_SD.value,
            source_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors",
            dir_key="paths_loras",
            is_builtin=True,
            description="Hyper-SD 4-step LoRA for ultra-fast inference",
        ))
    except ImportError:
        register(ResourceDef(
            name="LCM LoRA (Extreme Speed)",
            resource_type=ResourceType.LORA,
            filename="sdxl_lcm_lora.safetensors",
            source_url="https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors",
            dir_key="paths_loras",
            is_builtin=True,
            description="Latency Consistency Model LoRA for extreme speed",
        ))
        register(ResourceDef(
            name="Lightning LoRA (4-step)",
            resource_type=ResourceType.LORA,
            filename="sdxl_lightning_4step_lora.safetensors",
            source_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors",
            dir_key="paths_loras",
            is_builtin=True,
            description="Lightning 4-step LoRA for fast inference",
        ))
        register(ResourceDef(
            name="Hyper-SD LoRA (4-step)",
            resource_type=ResourceType.LORA,
            filename="sdxl_hyper_sd_4step_lora.safetensors",
            source_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors",
            dir_key="paths_loras",
            is_builtin=True,
            description="Hyper-SD 4-step LoRA for ultra-fast inference",
        ))

    _initialized = True


def get_registry() -> Dict[str, ResourceDef]:
    _init_registry()
    return dict(_registry)


def get_resource_def(key: str) -> Optional[ResourceDef]:
    _init_registry()
    return _registry.get(key)


def get_resources_by_type(rtype: ResourceType) -> List[ResourceDef]:
    _init_registry()
    return [r for r in _registry.values() if r.resource_type == rtype]


def get_autodownload_list() -> List[ResourceDef]:
    _init_registry()
    return [r for r in _registry.values() if r.autodownload and r.source_url]


def resolve_dir(rdef: ResourceDef, config_module) -> str:
    if rdef.dir_key:
        val = getattr(config_module, rdef.dir_key, None)
        if val:
            if isinstance(val, list):
                return val[0] if val else ""
            return val

    type_to_keys = {
        ResourceType.CHECKPOINT: ["paths_checkpoints"],
        ResourceType.LORA: ["paths_loras"],
        ResourceType.VAE: ["path_vae"],
        ResourceType.CONTROLNET: ["path_controlnet"],
        ResourceType.INPAINT: ["path_inpaint"],
        ResourceType.VAE_APPROX: ["path_vae_approx"],
        ResourceType.CLIP_VISION: ["path_clip_vision"],
        ResourceType.UPSCALE: ["path_upscale_models"],
        ResourceType.SAFETY_CHECKER: ["path_safety_checker"],
        ResourceType.SAM: ["path_sam"],
        ResourceType.EXPANSION: ["path_fooocus_expansion"],
        ResourceType.EMBEDDING: ["path_embeddings"],
    }

    for k in type_to_keys.get(rdef.resource_type, []):
        val = getattr(config_module, k, None)
        if val:
            if isinstance(val, list):
                return val[0] if val else ""
            return val
    return ""


def resolve_all_dirs(rtype: ResourceType, config_module) -> List[str]:
    type_to_keys = {
        ResourceType.CHECKPOINT: ["paths_checkpoints"],
        ResourceType.LORA: ["paths_loras"],
        ResourceType.VAE: ["path_vae"],
        ResourceType.CONTROLNET: ["path_controlnet"],
        ResourceType.INPAINT: ["path_inpaint"],
        ResourceType.VAE_APPROX: ["path_vae_approx"],
        ResourceType.CLIP_VISION: ["path_clip_vision"],
        ResourceType.UPSCALE: ["path_upscale_models"],
        ResourceType.SAFETY_CHECKER: ["path_safety_checker"],
        ResourceType.SAM: ["path_sam"],
        ResourceType.EXPANSION: ["path_fooocus_expansion"],
        ResourceType.EMBEDDING: ["path_embeddings"],
    }

    dirs = []
    for k in type_to_keys.get(rtype, []):
        val = getattr(config_module, k, None)
        if val:
            if isinstance(val, list):
                dirs.extend(val)
            else:
                dirs.append(val)
    return dirs


def resolve_all_dirs_for_def(rdef: ResourceDef, config_module) -> List[str]:
    dirs = []
    seen = set()

    def add_dir(d):
        if d and d not in seen:
            dirs.append(d)
            seen.add(d)

    if rdef.dir_key:
        val = getattr(config_module, rdef.dir_key, None)
        if val:
            if isinstance(val, list):
                for d in val:
                    add_dir(d)
            else:
                add_dir(val)

    type_to_keys = {
        ResourceType.CHECKPOINT: ["paths_checkpoints"],
        ResourceType.LORA: ["paths_loras"],
        ResourceType.VAE: ["path_vae"],
        ResourceType.CONTROLNET: ["path_controlnet"],
        ResourceType.INPAINT: ["path_inpaint"],
        ResourceType.VAE_APPROX: ["path_vae_approx"],
        ResourceType.CLIP_VISION: ["path_clip_vision"],
        ResourceType.UPSCALE: ["path_upscale_models"],
        ResourceType.SAFETY_CHECKER: ["path_safety_checker"],
        ResourceType.SAM: ["path_sam"],
        ResourceType.EXPANSION: ["path_fooocus_expansion"],
        ResourceType.EMBEDDING: ["path_embeddings"],
    }

    for k in type_to_keys.get(rdef.resource_type, []):
        val = getattr(config_module, k, None)
        if val:
            if isinstance(val, list):
                for d in val:
                    add_dir(d)
            else:
                add_dir(val)

    return dirs


def resolve_all_possible_paths(rdef: ResourceDef, config_module) -> List[str]:
    dirs = resolve_all_dirs_for_def(rdef, config_module)
    return [os.path.join(d, rdef.filename) for d in dirs if d]


def get_type_keys(rtype: ResourceType) -> List[str]:
    type_to_keys = {
        ResourceType.CHECKPOINT: ["paths_checkpoints"],
        ResourceType.LORA: ["paths_loras"],
        ResourceType.VAE: ["path_vae"],
        ResourceType.CONTROLNET: ["path_controlnet"],
        ResourceType.INPAINT: ["path_inpaint"],
        ResourceType.VAE_APPROX: ["path_vae_approx"],
        ResourceType.CLIP_VISION: ["path_clip_vision"],
        ResourceType.UPSCALE: ["path_upscale_models"],
        ResourceType.SAFETY_CHECKER: ["path_safety_checker"],
        ResourceType.SAM: ["path_sam"],
        ResourceType.EXPANSION: ["path_fooocus_expansion"],
        ResourceType.EMBEDDING: ["path_embeddings"],
    }
    return type_to_keys.get(rtype, [])
