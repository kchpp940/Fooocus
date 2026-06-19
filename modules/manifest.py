import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import modules.diagnostics as diagnostics
from modules.diagnostics import (
    DiagnosticStage, DiagnosticErrorCategory, get_current_context,
    log_info, log_warning, log_error,
)


class ResourceCategory(str, Enum):
    CHECKPOINTS = "checkpoints"
    LORAS = "loras"
    EMBEDDINGS = "embeddings"
    VAE = "vae"
    VAE_APPROX = "vae_approx"
    UPSCALE_MODELS = "upscale_models"
    INPAINT = "inpaint"
    CONTROLNET = "controlnet"
    CLIP_VISION = "clip_vision"
    FOOOCUS_EXPANSION = "prompt_expansion"
    SAFETY_CHECKER = "safety_checker"
    SAM = "sam"
    WILDCARDS = "wildcards"
    CONFIGS = "configs"
    UNET = "unet"
    GLIGEN = "gligen"
    HYPERNETWORKS = "hypernetworks"
    CLIP = "clip"
    DIFFUSERS = "diffusers"
    STYLE_MODELS = "style_models"


@dataclass
class ResourceItem:
    name: str
    category: ResourceCategory
    url: str = ""
    sha256: Optional[str] = None
    required: bool = False
    description: str = ""
    file_name: Optional[str] = None
    sub_path: Optional[str] = None

    def get_filename(self) -> str:
        if self.file_name:
            return self.file_name
        if self.url:
            return os.path.basename(self.url)
        return self.name


@dataclass
class Manifest:
    version: str = "1.0"
    fooocus_version: str = ""
    resources: Dict[str, ResourceItem] = field(default_factory=dict)

    def add_resource(self, item: ResourceItem):
        self.resources[item.name] = item

    def get_resources_by_category(self, category: ResourceCategory) -> List[ResourceItem]:
        return [r for r in self.resources.values() if r.category == category]

    def get_required_resources(self) -> List[ResourceItem]:
        return [r for r in self.resources.values() if r.required]


def load_manifest(manifest_path: str) -> Manifest:
    if not os.path.exists(manifest_path):
        print(f"[Manifest] Manifest file not found: {manifest_path}")
        return Manifest()

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = Manifest(
            version=data.get("version", "1.0"),
            fooocus_version=data.get("fooocus_version", ""),
        )

        for name, item_data in data.get("resources", {}).items():
            category_str = item_data.get("category", "checkpoints")
            try:
                category = ResourceCategory(category_str)
            except ValueError:
                category = ResourceCategory.CHECKPOINTS

            item = ResourceItem(
                name=name,
                category=category,
                url=item_data.get("url", ""),
                sha256=item_data.get("sha256"),
                required=item_data.get("required", False),
                description=item_data.get("description", ""),
                file_name=item_data.get("file_name"),
                sub_path=item_data.get("sub_path"),
            )
            manifest.add_resource(item)

        print(f"[Manifest] Loaded manifest with {len(manifest.resources)} resources")
        return manifest

    except Exception as e:
        print(f"[Manifest] Failed to load manifest: {e}")
        return Manifest()


def get_category_dir(category: ResourceCategory) -> str:
    mapping = {
        ResourceCategory.CHECKPOINTS: "checkpoints",
        ResourceCategory.LORAS: "loras",
        ResourceCategory.EMBEDDINGS: "embeddings",
        ResourceCategory.VAE: "vae",
        ResourceCategory.VAE_APPROX: "vae_approx",
        ResourceCategory.UPSCALE_MODELS: "upscale_models",
        ResourceCategory.INPAINT: "inpaint",
        ResourceCategory.CONTROLNET: "controlnet",
        ResourceCategory.CLIP_VISION: "clip_vision",
        ResourceCategory.FOOOCUS_EXPANSION: "prompt_expansion/fooocus_expansion",
        ResourceCategory.SAFETY_CHECKER: "safety_checker",
        ResourceCategory.SAM: "sam",
        ResourceCategory.WILDCARDS: "wildcards",
        ResourceCategory.CONFIGS: "configs",
        ResourceCategory.UNET: "unet",
        ResourceCategory.GLIGEN: "gligen",
        ResourceCategory.HYPERNETWORKS: "hypernetworks",
        ResourceCategory.CLIP: "clip",
        ResourceCategory.DIFFUSERS: "diffusers",
        ResourceCategory.STYLE_MODELS: "style_models",
    }
    return mapping.get(category, "checkpoints")


def resolve_resource_path(
    item: ResourceItem,
    models_root: str,
) -> str:
    category_dir = get_category_dir(item.category)
    base_dir = os.path.join(models_root, category_dir)
    if item.sub_path:
        base_dir = os.path.join(base_dir, item.sub_path)
    filename = item.get_filename()
    return os.path.join(base_dir, filename)


def sha256_file(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class CheckResult:
    def __init__(self):
        self.present: List[ResourceItem] = []
        self.missing_required: List[ResourceItem] = []
        self.missing_optional: List[ResourceItem] = []
        self.hash_mismatch: List[Tuple[ResourceItem, str, str]] = []
        self.errors: List[str] = []

    @property
    def is_ok(self) -> bool:
        return len(self.missing_required) == 0 and len(self.errors) == 0

    def summary(self) -> str:
        parts = []
        if self.present:
            parts.append(f"present={len(self.present)}")
        if self.missing_required:
            parts.append(f"missing_required={len(self.missing_required)}")
        if self.missing_optional:
            parts.append(f"missing_optional={len(self.missing_optional)}")
        if self.hash_mismatch:
            parts.append(f"hash_mismatch={len(self.hash_mismatch)}")
        return ", ".join(parts) if parts else "empty"


def check_manifest_resources(
    manifest: Manifest,
    models_root: str,
    check_hash: bool = False,
) -> CheckResult:
    ctx = get_current_context()
    result = CheckResult()

    if not manifest.resources:
        print("[Manifest] No resources in manifest, skipping check.")
        return result

    log_info(
        DiagnosticStage.RESOURCE_SCAN,
        f"开始校验 manifest 资源清单 (共 {len(manifest.resources)} 个)",
        ctx=ctx,
        extra_data={"resources_total": len(manifest.resources)},
    )

    for item in manifest.resources.values():
        file_path = resolve_resource_path(item, models_root)

        if not os.path.exists(file_path):
            if item.required:
                result.missing_required.append(item)
                log_warning(
                    DiagnosticStage.RESOURCE_SCAN,
                    f"必需模型缺失: {item.name}",
                    extra_data={
                        "resource_name": item.name,
                        "category": item.category.value,
                        "expected_path": file_path,
                        "download_url": item.url,
                    },
                )
            else:
                result.missing_optional.append(item)
            continue

        if check_hash and item.sha256:
            try:
                actual_hash = sha256_file(file_path)
                if actual_hash.lower() != item.sha256.lower():
                    result.hash_mismatch.append((item, item.sha256, actual_hash))
                    log_warning(
                        DiagnosticStage.RESOURCE_SCAN,
                        f"模型文件 hash 不匹配: {item.name}",
                        extra_data={
                            "resource_name": item.name,
                            "expected_sha256": item.sha256,
                            "actual_sha256": actual_hash,
                            "file_path": file_path,
                        },
                    )
                    continue
            except Exception as e:
                result.errors.append(f"Failed to hash {file_path}: {e}")

        result.present.append(item)

    log_info(
        DiagnosticStage.RESOURCE_SCAN,
        f"Manifest 校验完成: {result.summary()}",
        ctx=ctx,
        extra_data={
            "present_count": len(result.present),
            "missing_required_count": len(result.missing_required),
            "missing_optional_count": len(result.missing_optional),
            "hash_mismatch_count": len(result.hash_mismatch),
        },
    )

    return result


def print_manifest_report(result: CheckResult, manifest: Manifest) -> bool:
    if not manifest.resources:
        return True

    print()
    print("=" * 70)
    print("  Resource Manifest Check")
    print("=" * 70)

    if result.present:
        print(f"\n  Present ({len(result.present)}):")
        for item in result.present:
            print(f"    [OK] {item.name}")

    if result.missing_optional:
        print(f"\n  Missing - Optional ({len(result.missing_optional)}):")
        for item in result.missing_optional:
            print(f"    [--] {item.name} ({item.category.value})")
            if item.url:
                print(f"         URL: {item.url}")

    if result.missing_required:
        print(f"\n  Missing - Required ({len(result.missing_required)}):")
        for item in result.missing_required:
            print(f"    [!!] {item.name} ({item.category.value})")
            if item.description:
                print(f"         {item.description}")
            if item.url:
                print(f"         Download: {item.url}")

    if result.hash_mismatch:
        print(f"\n  Hash Mismatch ({len(result.hash_mismatch)}):")
        for item, expected, actual in result.hash_mismatch:
            print(f"    [!!] {item.name}")
            print(f"         Expected: {expected}")
            print(f"         Actual:   {actual}")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"    [!!] {err}")

    print()
    print("=" * 70)

    if result.is_ok:
        print("  Status: PASSED - All required resources are present.")
        print("=" * 70)
        return True
    else:
        print("  Status: FAILED - Missing required resources!")
        print("=" * 70)
        return False


def generate_manifest_from_config(
    checkpoint_downloads: Dict[str, str],
    lora_downloads: Dict[str, str],
    embeddings_downloads: Dict[str, str],
    vae_downloads: Dict[str, str],
    vae_approx_filenames: List[Tuple[str, str]],
    fooocus_expansion_url: str,
) -> Manifest:
    manifest = Manifest()

    for name, url in checkpoint_downloads.items():
        manifest.add_resource(ResourceItem(
            name=name,
            category=ResourceCategory.CHECKPOINTS,
            url=url,
            required=True,
            description="Base checkpoint model",
        ))

    for name, url in lora_downloads.items():
        manifest.add_resource(ResourceItem(
            name=name,
            category=ResourceCategory.LORAS,
            url=url,
            required=False,
            description="LoRA model",
        ))

    for name, url in embeddings_downloads.items():
        manifest.add_resource(ResourceItem(
            name=name,
            category=ResourceCategory.EMBEDDINGS,
            url=url,
            required=False,
            description="Embedding / Textual Inversion",
        ))

    for name, url in vae_downloads.items():
        manifest.add_resource(ResourceItem(
            name=name,
            category=ResourceCategory.VAE,
            url=url,
            required=False,
            description="VAE model",
        ))

    for filename, url in vae_approx_filenames:
        manifest.add_resource(ResourceItem(
            name=filename,
            category=ResourceCategory.VAE_APPROX,
            url=url,
            required=False,
            description="VAE approximation model (TAESD)",
            file_name=filename,
        ))

    manifest.add_resource(ResourceItem(
        name="fooocus_expansion",
        category=ResourceCategory.FOOOCUS_EXPANSION,
        url=fooocus_expansion_url,
        required=False,
        description="Fooocus prompt expansion model",
        file_name="pytorch_model.bin",
    ))

    return manifest


def save_manifest(manifest: Manifest, output_path: str):
    data = {
        "version": manifest.version,
        "fooocus_version": manifest.fooocus_version,
        "resources": {},
    }

    for name, item in manifest.resources.items():
        item_data = {
            "category": item.category.value,
            "url": item.url,
            "required": item.required,
            "description": item.description,
        }
        if item.sha256:
            item_data["sha256"] = item.sha256
        if item.file_name:
            item_data["file_name"] = item.file_name
        if item.sub_path:
            item_data["sub_path"] = item.sub_path
        data["resources"][name] = item_data

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[Manifest] Saved manifest to {output_path}")


def add_inpaint_resources(manifest: Manifest, version: str = "v2.6"):
    manifest.add_resource(ResourceItem(
        name="fooocus_inpaint_head",
        category=ResourceCategory.INPAINT,
        url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
        required=False,
        description="Fooocus inpaint head model",
        file_name="fooocus_inpaint_head.pth",
    ))

    patch_files = {
        "v1": "inpaint.fooocus.patch",
        "v2.5": "inpaint_v25.fooocus.patch",
        "v2.6": "inpaint_v26.fooocus.patch",
    }
    patch_name = patch_files.get(version, "inpaint_v26.fooocus.patch")
    manifest.add_resource(ResourceItem(
        name=f"fooocus_inpaint_patch_{version}",
        category=ResourceCategory.INPAINT,
        url=f"https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/{patch_name}",
        required=False,
        description=f"Fooocus inpaint patch ({version})",
        file_name=patch_name,
    ))


def add_controlnet_resources(manifest: Manifest):
    manifest.add_resource(ResourceItem(
        name="control_lora_canny_rank128",
        category=ResourceCategory.CONTROLNET,
        url="https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors",
        required=False,
        description="ControlNet canny model",
        file_name="control-lora-canny-rank128.safetensors",
    ))
    manifest.add_resource(ResourceItem(
        name="fooocus_xl_cpds_128",
        category=ResourceCategory.CONTROLNET,
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors",
        required=False,
        description="Fooocus XL CPDS controlnet",
        file_name="fooocus_xl_cpds_128.safetensors",
    ))


def add_ip_adapter_resources(manifest: Manifest):
    manifest.add_resource(ResourceItem(
        name="clip_vision_vit_h",
        category=ResourceCategory.CLIP_VISION,
        url="https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors",
        required=False,
        description="CLIP Vision ViT-H model for IP-Adapter",
        file_name="clip_vision_vit_h.safetensors",
    ))
    manifest.add_resource(ResourceItem(
        name="fooocus_ip_negative",
        category=ResourceCategory.CONTROLNET,
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors",
        required=False,
        description="Fooocus IP negative model",
        file_name="fooocus_ip_negative.safetensors",
    ))
    manifest.add_resource(ResourceItem(
        name="ip_adapter_plus_sdxl_vit_h",
        category=ResourceCategory.CONTROLNET,
        url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin",
        required=False,
        description="IP-Adapter Plus SDXL ViT-H",
        file_name="ip-adapter-plus_sdxl_vit-h.bin",
    ))
    manifest.add_resource(ResourceItem(
        name="ip_adapter_plus_face_sdxl_vit_h",
        category=ResourceCategory.CONTROLNET,
        url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin",
        required=False,
        description="IP-Adapter Plus Face SDXL ViT-H",
        file_name="ip-adapter-plus-face_sdxl_vit-h.bin",
    ))


def add_upscale_resources(manifest: Manifest):
    manifest.add_resource(ResourceItem(
        name="fooocus_upscaler",
        category=ResourceCategory.UPSCALE_MODELS,
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
        required=False,
        description="Fooocus upscaler model",
        file_name="fooocus_upscaler_s409985e5.bin",
    ))


def add_performance_lora_resources(manifest: Manifest):
    manifest.add_resource(ResourceItem(
        name="sdxl_lcm_lora",
        category=ResourceCategory.LORAS,
        url="https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors",
        required=False,
        description="SDXL LCM LoRA (extreme speed performance)",
        file_name="sdxl_lcm_lora.safetensors",
    ))
    manifest.add_resource(ResourceItem(
        name="sdxl_lightning_4step_lora",
        category=ResourceCategory.LORAS,
        url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors",
        required=False,
        description="SDXL Lightning 4-step LoRA (lightning performance)",
        file_name="sdxl_lightning_4step_lora.safetensors",
    ))
    manifest.add_resource(ResourceItem(
        name="sdxl_hyper_sd_4step_lora",
        category=ResourceCategory.LORAS,
        url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors",
        required=False,
        description="SDXL Hyper-SD 4-step LoRA (hyper_sd performance)",
        file_name="sdxl_hyper_sd_4step_lora.safetensors",
    ))


def add_sam_resources(manifest: Manifest):
    manifest.add_resource(ResourceItem(
        name="sam_vit_b",
        category=ResourceCategory.SAM,
        url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth",
        required=False,
        description="SAM ViT-B model",
        file_name="sam_vit_b_01ec64.pth",
    ))
    manifest.add_resource(ResourceItem(
        name="sam_vit_l",
        category=ResourceCategory.SAM,
        url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth",
        required=False,
        description="SAM ViT-L model",
        file_name="sam_vit_l_0b3195.pth",
    ))
    manifest.add_resource(ResourceItem(
        name="sam_vit_h",
        category=ResourceCategory.SAM,
        url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth",
        required=False,
        description="SAM ViT-H model",
        file_name="sam_vit_h_4b8939.pth",
    ))


def add_safety_checker_resources(manifest: Manifest):
    manifest.add_resource(ResourceItem(
        name="safety_checker",
        category=ResourceCategory.SAFETY_CHECKER,
        url="https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin",
        required=False,
        description="Stable diffusion safety checker model",
        file_name="stable-diffusion-safety-checker.bin",
    ))


def build_default_manifest(
    checkpoint_downloads: Dict[str, str],
    lora_downloads: Dict[str, str],
    embeddings_downloads: Dict[str, str],
    vae_downloads: Dict[str, str],
    vae_approx_filenames: List[Tuple[str, str]],
    fooocus_expansion_url: str,
    fooocus_version: str = "",
    include_all: bool = False,
) -> Manifest:
    manifest = generate_manifest_from_config(
        checkpoint_downloads,
        lora_downloads,
        embeddings_downloads,
        vae_downloads,
        vae_approx_filenames,
        fooocus_expansion_url,
    )
    manifest.fooocus_version = fooocus_version

    add_inpaint_resources(manifest)
    add_controlnet_resources(manifest)
    add_ip_adapter_resources(manifest)
    add_upscale_resources(manifest)
    add_performance_lora_resources(manifest)
    add_sam_resources(manifest)
    add_safety_checker_resources(manifest)

    return manifest


def get_manifest_path(user_data_dir: str) -> str:
    return os.path.join(user_data_dir, "manifest.json")
