import os
import json
import time
import threading
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Callable

import modules.config
from modules.model_loader import load_file_from_url
from modules.hash_cache import hash_cache, sha256_from_cache, load_cache_from_file, save_cache_to_file
from modules.util import get_file_from_folder_list, sha256, HASH_SHA256_LENGTH
from modules.extra_utils import get_files_from_folder


class ResourceStatus(Enum):
    MISSING = "missing"
    EXISTS = "exists"
    DOWNLOADING = "downloading"
    DOWNLOAD_FAILED = "download_failed"
    HASH_VERIFYING = "hash_verifying"
    HASH_MISMATCH = "hash_mismatch"
    HASH_VERIFIED = "hash_verified"


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
class ResourceInfo:
    name: str
    resource_type: ResourceType
    filename: str
    directory: str
    full_path: str = ""
    source_url: Optional[str] = None
    expected_hash: Optional[str] = None
    status: ResourceStatus = ResourceStatus.MISSING
    file_size: int = 0
    current_hash: Optional[str] = None
    download_progress: float = 0.0
    download_speed: float = 0.0
    error_message: str = ""
    hash_trusted: bool = False
    is_builtin: bool = False
    last_updated: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d["status"] = self.status.value
        d["resource_type"] = self.resource_type.value
        d["file_size_human"] = format_size(self.file_size)
        return d


def format_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.2f} {units[i]}"


class DownloadProgressTracker:
    def __init__(self):
        self._downloads: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._listeners: List[Callable] = []

    def start(self, key: str, filename: str):
        with self._lock:
            self._downloads[key] = {
                "filename": filename,
                "progress": 0.0,
                "speed": 0.0,
                "start_time": time.time(),
                "status": "downloading"
            }
        self._notify()

    def update(self, key: str, progress: float, speed: float = 0.0):
        with self._lock:
            if key in self._downloads:
                self._downloads[key]["progress"] = progress
                self._downloads[key]["speed"] = speed
        self._notify()

    def finish(self, key: str, success: bool, error: str = ""):
        with self._lock:
            if key in self._downloads:
                self._downloads[key]["status"] = "success" if success else "failed"
                self._downloads[key]["error"] = error
                self._downloads[key]["progress"] = 1.0 if success else self._downloads[key].get("progress", 0)
        self._notify()

    def get(self, key: str) -> Optional[Dict]:
        with self._lock:
            return self._downloads.get(key)

    def add_listener(self, callback: Callable):
        self._listeners.append(callback)

    def _notify(self):
        for listener in self._listeners:
            try:
                listener()
            except Exception:
                pass


download_tracker = DownloadProgressTracker()

builtin_resources: Dict[str, ResourceInfo] = {}


VAE_APPROX_FILENAMES = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]


def _get_dir_for_type(rtype: ResourceType) -> str:
    mapping = {
        ResourceType.CHECKPOINT: modules.config.paths_checkpoints[0] if modules.config.paths_checkpoints else "",
        ResourceType.LORA: modules.config.paths_loras[0] if modules.config.paths_loras else "",
        ResourceType.VAE: modules.config.path_vae,
        ResourceType.CONTROLNET: modules.config.path_controlnet,
        ResourceType.INPAINT: modules.config.path_inpaint,
        ResourceType.VAE_APPROX: modules.config.path_vae_approx,
        ResourceType.CLIP_VISION: modules.config.path_clip_vision,
        ResourceType.UPSCALE: modules.config.path_upscale_models,
        ResourceType.SAFETY_CHECKER: modules.config.path_safety_checker,
        ResourceType.SAM: modules.config.path_sam,
        ResourceType.EXPANSION: modules.config.path_fooocus_expansion,
        ResourceType.EMBEDDING: modules.config.path_embeddings,
    }
    return mapping.get(rtype, "")


def register_builtin_resources():

    resources = []

    for filename, url in VAE_APPROX_FILENAMES:
        resources.append(ResourceInfo(
            name=f"VAE Approx - {filename}",
            resource_type=ResourceType.VAE_APPROX,
            filename=filename,
            directory=modules.config.path_vae_approx,
            source_url=url,
            is_builtin=True
        ))

    resources.append(ResourceInfo(
        name="Fooocus Expansion Model",
        resource_type=ResourceType.EXPANSION,
        filename="pytorch_model.bin",
        directory=modules.config.path_fooocus_expansion,
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="Inpaint Head Model",
        resource_type=ResourceType.INPAINT,
        filename="fooocus_inpaint_head.pth",
        directory=modules.config.path_inpaint,
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="Inpaint Patch v1",
        resource_type=ResourceType.INPAINT,
        filename="inpaint.fooocus.patch",
        directory=modules.config.path_inpaint,
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="Inpaint Patch v2.5",
        resource_type=ResourceType.INPAINT,
        filename="inpaint_v25.fooocus.patch",
        directory=modules.config.path_inpaint,
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="Inpaint Patch v2.6",
        resource_type=ResourceType.INPAINT,
        filename="inpaint_v26.fooocus.patch",
        directory=modules.config.path_inpaint,
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="ControlNet Canny (rank128)",
        resource_type=ResourceType.CONTROLNET,
        filename="control-lora-canny-rank128.safetensors",
        directory=modules.config.path_controlnet,
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="ControlNet CPDS",
        resource_type=ResourceType.CONTROLNET,
        filename="fooocus_xl_cpds_128.safetensors",
        directory=modules.config.path_controlnet,
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="CLIP Vision ViT-H",
        resource_type=ResourceType.CLIP_VISION,
        filename="clip_vision_vit_h.safetensors",
        directory=modules.config.path_clip_vision,
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="IP Adapter Negative",
        resource_type=ResourceType.CONTROLNET,
        filename="fooocus_ip_negative.safetensors",
        directory=modules.config.path_controlnet,
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="IP Adapter Plus SDXL",
        resource_type=ResourceType.CONTROLNET,
        filename="ip-adapter-plus_sdxl_vit-h.bin",
        directory=modules.config.path_controlnet,
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="IP Adapter Plus Face SDXL",
        resource_type=ResourceType.CONTROLNET,
        filename="ip-adapter-plus-face_sdxl_vit-h.bin",
        directory=modules.config.path_controlnet,
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="Fooocus Upscaler",
        resource_type=ResourceType.UPSCALE,
        filename="fooocus_upscaler_s409985e5.bin",
        directory=modules.config.path_upscale_models,
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="Stable Diffusion Safety Checker",
        resource_type=ResourceType.SAFETY_CHECKER,
        filename="stable-diffusion-safety-checker.bin",
        directory=modules.config.path_safety_checker,
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="SAM ViT-B",
        resource_type=ResourceType.SAM,
        filename="sam_vit_b_01ec64.pth",
        directory=modules.config.path_sam,
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="SAM ViT-L",
        resource_type=ResourceType.SAM,
        filename="sam_vit_l_0b3195.pth",
        directory=modules.config.path_sam,
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="SAM ViT-H",
        resource_type=ResourceType.SAM,
        filename="sam_vit_h_4b8939.pth",
        directory=modules.config.path_sam,
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="LCM LoRA (Extreme Speed)",
        resource_type=ResourceType.LORA,
        filename="sdxl_lcm_lora.safetensors",
        directory=modules.config.paths_loras[0] if modules.config.paths_loras else "",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="Lightning LoRA (4-step)",
        resource_type=ResourceType.LORA,
        filename="sdxl_lightning_4step_lora.safetensors",
        directory=modules.config.paths_loras[0] if modules.config.paths_loras else "",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors",
        is_builtin=True
    ))

    resources.append(ResourceInfo(
        name="Hyper-SD LoRA (4-step)",
        resource_type=ResourceType.LORA,
        filename="sdxl_hyper_sd_4step_lora.safetensors",
        directory=modules.config.paths_loras[0] if modules.config.paths_loras else "",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors",
        is_builtin=True
    ))

    for r in resources:
        key = f"{r.resource_type.value}:{r.filename}"
        builtin_resources[key] = r


def _scan_directory_for_type(rtype: ResourceType, dirs) -> List[ResourceInfo]:
    result = []
    extensions = ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch', '.pt']
    if not isinstance(dirs, list):
        dirs = [dirs]

    for directory in dirs:
        if not directory or not os.path.exists(directory):
            continue
        try:
            filenames = get_files_from_folder(directory, extensions)
            for fn in filenames:
                full_path = os.path.abspath(os.path.join(directory, fn))
                key = f"{rtype.value}:{os.path.basename(fn)}"
                if key in builtin_resources:
                    continue
                result.append(ResourceInfo(
                    name=os.path.basename(fn),
                    resource_type=rtype,
                    filename=os.path.basename(fn),
                    directory=directory,
                    full_path=full_path,
                    is_builtin=False
                ))
        except Exception:
            continue
    return result


def _scan_user_downloads_from_config() -> List[ResourceInfo]:
    result = []

    def _add_from_dict(download_dict: Dict, rtype: ResourceType, dirs):
        dir_path = dirs[0] if isinstance(dirs, list) and len(dirs) > 0 else dirs
        for fn, url in download_dict.items():
            key = f"{rtype.value}:{fn}"
            if key in builtin_resources:
                builtin_resources[key].source_url = url
                continue
            result.append(ResourceInfo(
                name=fn,
                resource_type=rtype,
                filename=fn,
                directory=dir_path,
                source_url=url,
                is_builtin=False
            ))

    _add_from_dict(modules.config.checkpoint_downloads, ResourceType.CHECKPOINT, modules.config.paths_checkpoints)
    _add_from_dict(modules.config.lora_downloads, ResourceType.LORA, modules.config.paths_loras)
    _add_from_dict(modules.config.vae_downloads, ResourceType.VAE, modules.config.path_vae)
    _add_from_dict(modules.config.embeddings_downloads, ResourceType.EMBEDDING, modules.config.path_embeddings)

    return result


def _update_resource_status(resource: ResourceInfo):
    if not resource.full_path:
        if resource.directory and resource.filename:
            resource.full_path = os.path.abspath(os.path.join(resource.directory, resource.filename))
        elif resource.resource_type in (ResourceType.CHECKPOINT, ResourceType.LORA):
            paths = modules.config.paths_checkpoints if resource.resource_type == ResourceType.CHECKPOINT else modules.config.paths_loras
            resource.full_path = get_file_from_folder_list(resource.filename, paths)

    key = f"{resource.resource_type.value}:{resource.filename}"
    download_info = download_tracker.get(key)
    if download_info:
        if download_info.get("status") == "downloading":
            resource.status = ResourceStatus.DOWNLOADING
            resource.download_progress = download_info.get("progress", 0)
            resource.download_speed = download_info.get("speed", 0)
            return
        elif download_info.get("status") == "failed":
            resource.status = ResourceStatus.DOWNLOAD_FAILED
            resource.error_message = download_info.get("error", "")
            return

    if resource.full_path and os.path.isfile(resource.full_path):
        try:
            stat = os.stat(resource.full_path)
            resource.file_size = stat.st_size
            resource.last_updated = stat.st_mtime
        except Exception:
            pass

        load_cache_from_file()
        abs_path = os.path.abspath(resource.full_path)
        if abs_path in hash_cache:
            resource.current_hash = hash_cache[abs_path]
            resource.hash_trusted = True
            if resource.expected_hash and resource.current_hash:
                if resource.current_hash.startswith(resource.expected_hash) or resource.expected_hash.startswith(resource.current_hash):
                    resource.status = ResourceStatus.HASH_VERIFIED
                else:
                    resource.status = ResourceStatus.HASH_MISMATCH
            else:
                resource.status = ResourceStatus.EXISTS
        else:
            resource.status = ResourceStatus.EXISTS
    else:
        resource.status = ResourceStatus.MISSING
        resource.file_size = 0
        resource.current_hash = None
        resource.hash_trusted = False


def get_all_resources() -> Dict[str, ResourceInfo]:
    all_resources: Dict[str, ResourceInfo] = {}

    if not builtin_resources:
        register_builtin_resources()

    for key, res in builtin_resources.items():
        all_resources[key] = ResourceInfo(**asdict(res))

    user_from_config = _scan_user_downloads_from_config()
    for res in user_from_config:
        key = f"{res.resource_type.value}:{res.filename}"
        if key not in all_resources:
            all_resources[key] = res

    type_dir_mapping = [
        (ResourceType.CHECKPOINT, modules.config.paths_checkpoints),
        (ResourceType.LORA, modules.config.paths_loras),
        (ResourceType.VAE, modules.config.path_vae),
        (ResourceType.CONTROLNET, modules.config.path_controlnet),
        (ResourceType.INPAINT, modules.config.path_inpaint),
        (ResourceType.VAE_APPROX, modules.config.path_vae_approx),
        (ResourceType.CLIP_VISION, modules.config.path_clip_vision),
        (ResourceType.UPSCALE, modules.config.path_upscale_models),
        (ResourceType.SAFETY_CHECKER, modules.config.path_safety_checker),
        (ResourceType.SAM, modules.config.path_sam),
        (ResourceType.EXPANSION, modules.config.path_fooocus_expansion),
        (ResourceType.EMBEDDING, modules.config.path_embeddings),
    ]

    for rtype, dirs in type_dir_mapping:
        scanned = _scan_directory_for_type(rtype, dirs)
        for res in scanned:
            key = f"{res.resource_type.value}:{res.filename}"
            if key not in all_resources:
                all_resources[key] = res

    for key in all_resources:
        _update_resource_status(all_resources[key])

    return all_resources


def get_resources_by_type(rtype: ResourceType) -> List[ResourceInfo]:
    all_res = get_all_resources()
    return [r for r in all_res.values() if r.resource_type == rtype]


def get_resources_summary() -> Dict:
    all_res = get_all_resources()
    summary = {"total": len(all_res), "by_type": {}, "by_status": {}}
    for r in all_res.values():
        t = r.resource_type.value
        s = r.status.value
        summary["by_type"][t] = summary["by_type"].get(t, 0) + 1
        summary["by_status"][s] = summary["by_status"].get(s, 0) + 1
    return summary


def download_resource(resource_key: str, force: bool = False) -> bool:
    all_res = get_all_resources()
    if resource_key not in all_res:
        return False

    res = all_res[resource_key]
    if not res.source_url:
        return False

    if not force and res.status in (ResourceStatus.EXISTS, ResourceStatus.HASH_VERIFIED):
        return True

    key = resource_key
    filename = res.filename
    model_dir = res.directory

    if not model_dir:
        model_dir = _get_dir_for_type(res.resource_type)

    if not model_dir:
        return False

    download_tracker.start(key, filename)

    def progress_cb(downloaded: int, total: int, speed: float):
        progress = downloaded / total if total > 0 else 0
        download_tracker.update(key, progress, speed)

    def run_download():
        try:
            result_path = load_file_from_url(
                url=res.source_url,
                model_dir=model_dir,
                file_name=filename,
                progress=True,
                progress_callback=progress_cb
            )
            if result_path and os.path.exists(result_path):
                try:
                    h = sha256(result_path, length=None)
                    abs_p = os.path.abspath(result_path)
                    hash_cache[abs_p] = h[:HASH_SHA256_LENGTH]
                    save_cache_to_file(abs_p, h[:HASH_SHA256_LENGTH])
                except Exception:
                    pass
                download_tracker.finish(key, True)
            else:
                download_tracker.finish(key, False, "Download returned empty path")
        except Exception as e:
            download_tracker.finish(key, False, str(e))

    thread = threading.Thread(target=run_download, daemon=True)
    thread.start()
    return True


def rehash_resource(resource_key: str) -> bool:
    all_res = get_all_resources()
    if resource_key not in all_res:
        return False

    res = all_res[resource_key]
    if not res.full_path or not os.path.isfile(res.full_path):
        return False

    key = resource_key
    abs_p = os.path.abspath(res.full_path)

    def run_hash():
        try:
            if abs_p in hash_cache:
                del hash_cache[abs_p]
            h = sha256_from_cache(abs_p)
            save_cache_to_file()
        except Exception:
            pass

    thread = threading.Thread(target=run_hash, daemon=True)
    thread.start()
    return True


def get_all_resources_json() -> str:
    all_res = get_all_resources()
    result = []
    for key, res in all_res.items():
        d = res.to_dict()
        d["key"] = key
        result.append(d)
    return json.dumps(result, ensure_ascii=False, indent=2)


def refresh_all_files():
    modules.config.update_files()
    return True
