import os
import json
import time
import threading
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Callable, Tuple


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


class HashStatus(Enum):
    NOT_COMPUTED = "not_computed"
    COMPUTED_UNVERIFIED = "computed_unverified"
    VERIFIED = "verified"
    MISMATCH = "mismatch"


class DownloadStatus(Enum):
    IDLE = "idle"
    PENDING = "pending"
    DOWNLOADING = "downloading"
    SUCCESS = "success"
    FAILED = "failed"


class ResourceStatus(Enum):
    MISSING = "missing"
    EXISTS = "exists"
    DOWNLOADING = "downloading"
    DOWNLOAD_FAILED = "download_failed"
    HASH_VERIFIED = "hash_verified"
    HASH_MISMATCH = "hash_mismatch"


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


@dataclass
class ResourceRuntime:
    full_path: str = ""
    file_size: int = 0
    current_hash: Optional[str] = None
    hash_status: HashStatus = HashStatus.NOT_COMPUTED
    download_status: DownloadStatus = DownloadStatus.IDLE
    download_progress: float = 0.0
    download_speed: float = 0.0
    error_message: str = ""
    last_checked: float = 0.0


@dataclass
class ResourceInfo:
    def_: ResourceDef
    rt: ResourceRuntime = field(default_factory=ResourceRuntime)

    @property
    def key(self) -> str:
        return self.def_.key

    @property
    def status(self) -> ResourceStatus:
        if self.rt.download_status == DownloadStatus.DOWNLOADING:
            return ResourceStatus.DOWNLOADING
        if self.rt.download_status == DownloadStatus.FAILED:
            return ResourceStatus.DOWNLOAD_FAILED
        if not self.rt.full_path or not os.path.isfile(self.rt.full_path):
            return ResourceStatus.MISSING
        if self.rt.hash_status == HashStatus.VERIFIED:
            return ResourceStatus.HASH_VERIFIED
        if self.rt.hash_status == HashStatus.MISMATCH:
            return ResourceStatus.HASH_MISMATCH
        return ResourceStatus.EXISTS

    @property
    def directory(self) -> str:
        return os.path.dirname(self.rt.full_path) if self.rt.full_path else _get_dir_for_key(self.def_.dir_key, self.def_.resource_type)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.def_.name,
            "resource_type": self.def_.resource_type.value,
            "filename": self.def_.filename,
            "source_url": self.def_.source_url,
            "expected_hash": self.def_.expected_hash,
            "is_builtin": self.def_.is_builtin,
            "description": self.def_.description,
            "directory": self.directory,
            "full_path": self.rt.full_path,
            "status": self.status.value,
            "file_size": self.rt.file_size,
            "file_size_human": format_size(self.rt.file_size),
            "current_hash": self.rt.current_hash,
            "hash_status": self.rt.hash_status.value,
            "download_status": self.rt.download_status.value,
            "download_progress": self.rt.download_progress,
            "download_speed": self.rt.download_speed,
            "error_message": self.rt.error_message,
            "last_checked": self.rt.last_checked,
        }


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


def format_speed(speed_bytes: float) -> str:
    if speed_bytes <= 0:
        return "-"
    units = ["B/s", "KB/s", "MB/s", "GB/s"]
    i = 0
    s = float(speed_bytes)
    while s >= 1024 and i < len(units) - 1:
        s /= 1024
        i += 1
    return f"{s:.2f} {units[i]}"


_builtin_registry: Dict[str, ResourceDef] = {}
_runtime_cache: Dict[str, ResourceRuntime] = {}
_cache_lock = threading.Lock()
_initialized = False


def _get_dir_for_key(dir_key: str, rtype: ResourceType) -> str:
    import modules.config as cfg

    if dir_key:
        val = getattr(cfg, dir_key, None)
        if val:
            if isinstance(val, list):
                return val[0] if val else ""
            return val

    type_to_keys: Dict[ResourceType, List[str]] = {
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

    keys = type_to_keys.get(rtype, [])
    for k in keys:
        val = getattr(cfg, k, None)
        if val:
            if isinstance(val, list):
                return val[0] if val else ""
            return val
    return ""


def _get_all_dirs_for_type(rtype: ResourceType) -> List[str]:
    import modules.config as cfg

    type_to_keys: Dict[ResourceType, List[str]] = {
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
        val = getattr(cfg, k, None)
        if val:
            if isinstance(val, list):
                dirs.extend(val)
            else:
                dirs.append(val)
    return dirs


def register_resource(rdef: ResourceDef):
    _builtin_registry[rdef.key] = rdef


def register_builtin_resources():
    global _initialized
    if _initialized:
        return

    register_resource(ResourceDef(
        name="VAE Approx XL",
        resource_type=ResourceType.VAE_APPROX,
        filename="xlvaeapp.pth",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth",
        dir_key="path_vae_approx",
        is_builtin=True,
        autodownload=True,
        description="SDXL VAE Approximation Model",
    ))

    register_resource(ResourceDef(
        name="VAE Approx SD1.5",
        resource_type=ResourceType.VAE_APPROX,
        filename="vaeapp_sd15.pth",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt",
        dir_key="path_vae_approx",
        is_builtin=True,
        autodownload=True,
        description="SD 1.5 VAE Approximation Model",
    ))

    register_resource(ResourceDef(
        name="XL-to-V1 Interposer v4.0",
        resource_type=ResourceType.VAE_APPROX,
        filename="xl-to-v1_interposer-v4.0.safetensors",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors",
        dir_key="path_vae_approx",
        is_builtin=True,
        autodownload=True,
        description="XL to SD 1.5 VAE Interposer",
    ))

    register_resource(ResourceDef(
        name="Fooocus Prompt Expansion",
        resource_type=ResourceType.EXPANSION,
        filename="pytorch_model.bin",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
        dir_key="path_fooocus_expansion",
        is_builtin=True,
        autodownload=True,
        description="Fooocus prompt expansion model",
    ))

    register_resource(ResourceDef(
        name="Inpaint Head Model",
        resource_type=ResourceType.INPAINT,
        filename="fooocus_inpaint_head.pth",
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
        dir_key="path_inpaint",
        is_builtin=True,
        description="Fooocus inpaint head (shared by all inpaint engines)",
    ))

    register_resource(ResourceDef(
        name="Inpaint Patch v1",
        resource_type=ResourceType.INPAINT,
        filename="inpaint.fooocus.patch",
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch",
        dir_key="path_inpaint",
        is_builtin=True,
        description="Inpaint engine v1 patch",
    ))

    register_resource(ResourceDef(
        name="Inpaint Patch v2.5",
        resource_type=ResourceType.INPAINT,
        filename="inpaint_v25.fooocus.patch",
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch",
        dir_key="path_inpaint",
        is_builtin=True,
        description="Inpaint engine v2.5 patch",
    ))

    register_resource(ResourceDef(
        name="Inpaint Patch v2.6",
        resource_type=ResourceType.INPAINT,
        filename="inpaint_v26.fooocus.patch",
        source_url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch",
        dir_key="path_inpaint",
        is_builtin=True,
        description="Inpaint engine v2.6 patch (latest)",
    ))

    register_resource(ResourceDef(
        name="ControlNet Canny (rank128)",
        resource_type=ResourceType.CONTROLNET,
        filename="control-lora-canny-rank128.safetensors",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors",
        dir_key="path_controlnet",
        is_builtin=True,
        description="ControlNet Canny LoRA",
    ))

    register_resource(ResourceDef(
        name="ControlNet CPDS",
        resource_type=ResourceType.CONTROLNET,
        filename="fooocus_xl_cpds_128.safetensors",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors",
        dir_key="path_controlnet",
        is_builtin=True,
        description="Fooocus CPDS ControlNet",
    ))

    register_resource(ResourceDef(
        name="CLIP Vision ViT-H",
        resource_type=ResourceType.CLIP_VISION,
        filename="clip_vision_vit_h.safetensors",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors",
        dir_key="path_clip_vision",
        is_builtin=True,
        description="CLIP Vision ViT-H for IP-Adapter",
    ))

    register_resource(ResourceDef(
        name="IP-Adapter Negative",
        resource_type=ResourceType.CONTROLNET,
        filename="fooocus_ip_negative.safetensors",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors",
        dir_key="path_controlnet",
        is_builtin=True,
        description="IP-Adapter negative conditioning",
    ))

    register_resource(ResourceDef(
        name="IP-Adapter Plus SDXL",
        resource_type=ResourceType.CONTROLNET,
        filename="ip-adapter-plus_sdxl_vit-h.bin",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin",
        dir_key="path_controlnet",
        is_builtin=True,
        description="IP-Adapter Plus for SDXL",
    ))

    register_resource(ResourceDef(
        name="IP-Adapter Plus Face SDXL",
        resource_type=ResourceType.CONTROLNET,
        filename="ip-adapter-plus-face_sdxl_vit-h.bin",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin",
        dir_key="path_controlnet",
        is_builtin=True,
        description="IP-Adapter Plus Face for SDXL",
    ))

    register_resource(ResourceDef(
        name="Fooocus Upscaler",
        resource_type=ResourceType.UPSCALE,
        filename="fooocus_upscaler_s409985e5.bin",
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
        dir_key="path_upscale_models",
        is_builtin=True,
        description="Fooocus 4x upscaler model",
    ))

    register_resource(ResourceDef(
        name="Stable Diffusion Safety Checker",
        resource_type=ResourceType.SAFETY_CHECKER,
        filename="stable-diffusion-safety-checker.bin",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin",
        dir_key="path_safety_checker",
        is_builtin=True,
        description="SD safety checker model",
    ))

    register_resource(ResourceDef(
        name="SAM ViT-B",
        resource_type=ResourceType.SAM,
        filename="sam_vit_b_01ec64.pth",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth",
        dir_key="path_sam",
        is_builtin=True,
        description="Segment Anything Model ViT-Base",
    ))

    register_resource(ResourceDef(
        name="SAM ViT-L",
        resource_type=ResourceType.SAM,
        filename="sam_vit_l_0b3195.pth",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth",
        dir_key="path_sam",
        is_builtin=True,
        description="Segment Anything Model ViT-Large",
    ))

    register_resource(ResourceDef(
        name="SAM ViT-H",
        resource_type=ResourceType.SAM,
        filename="sam_vit_h_4b8939.pth",
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth",
        dir_key="path_sam",
        is_builtin=True,
        description="Segment Anything Model ViT-Huge",
    ))

    import modules.flags as flags

    register_resource(ResourceDef(
        name="LCM LoRA (Extreme Speed)",
        resource_type=ResourceType.LORA,
        filename=flags.PerformanceLoRA.EXTREME_SPEED.value,
        source_url="https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors",
        dir_key="paths_loras",
        is_builtin=True,
        description="Latent Consistency Model LoRA for extreme speed",
    ))

    register_resource(ResourceDef(
        name="Lightning LoRA (4-step)",
        resource_type=ResourceType.LORA,
        filename=flags.PerformanceLoRA.LIGHTNING.value,
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors",
        dir_key="paths_loras",
        is_builtin=True,
        description="Lightning 4-step LoRA for fast inference",
    ))

    register_resource(ResourceDef(
        name="Hyper-SD LoRA (4-step)",
        resource_type=ResourceType.LORA,
        filename=flags.PerformanceLoRA.HYPER_SD.value,
        source_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors",
        dir_key="paths_loras",
        is_builtin=True,
        description="Hyper-SD 4-step LoRA for ultra-fast inference",
    ))

    _initialized = True


def get_builtin_registry() -> Dict[str, ResourceDef]:
    if not _initialized:
        register_builtin_resources()
    return dict(_builtin_registry)


def get_resource_def(key: str) -> Optional[ResourceDef]:
    if not _initialized:
        register_builtin_resources()
    return _builtin_registry.get(key)


def get_resources_by_type(rtype: ResourceType) -> List[ResourceDef]:
    if not _initialized:
        register_builtin_resources()
    return [r for r in _builtin_registry.values() if r.resource_type == rtype]


class DownloadProgressTracker:
    def __init__(self):
        self._downloads: Dict[str, ResourceRuntime] = {}
        self._lock = threading.Lock()
        self._listeners: List[Callable] = []

    def start(self, key: str):
        with self._lock:
            if key not in self._downloads:
                self._downloads[key] = ResourceRuntime()
            rt = self._downloads[key]
            rt.download_status = DownloadStatus.DOWNLOADING
            rt.download_progress = 0.0
            rt.download_speed = 0.0
            rt.error_message = ""
        self._notify()

    def update(self, key: str, progress: float, speed: float = 0.0):
        with self._lock:
            if key in self._downloads:
                rt = self._downloads[key]
                rt.download_progress = progress
                rt.download_speed = speed
        self._notify()

    def finish(self, key: str, success: bool, error: str = ""):
        with self._lock:
            if key in self._downloads:
                rt = self._downloads[key]
                rt.download_status = DownloadStatus.SUCCESS if success else DownloadStatus.FAILED
                rt.error_message = error
                rt.download_progress = 1.0 if success else rt.download_progress
                if not success:
                    rt.last_checked = time.time()
        self._notify()

    def get(self, key: str) -> Optional[ResourceRuntime]:
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


def _find_file_in_dirs(filename: str, dirs: List[str]) -> Optional[str]:
    from modules.util import get_file_from_folder_list
    return get_file_from_folder_list(filename, dirs)


def _compute_hash_status(rdef: ResourceDef, current_hash: Optional[str]) -> HashStatus:
    if not current_hash:
        return HashStatus.NOT_COMPUTED

    if rdef.expected_hash:
        expected = rdef.expected_hash.lower()
        actual = current_hash.lower()
        if actual.startswith(expected) or expected.startswith(actual):
            return HashStatus.VERIFIED
        else:
            return HashStatus.MISMATCH

    if rdef.is_builtin and rdef.source_url:
        return HashStatus.COMPUTED_UNVERIFIED

    return HashStatus.COMPUTED_UNVERIFIED


def _update_runtime_for_resource(info: ResourceInfo):
    rdef = info.def_
    rt = info.rt

    dirs = _get_all_dirs_for_type(rdef.resource_type)
    filepath = _find_file_in_dirs(rdef.filename, dirs)

    if not filepath and rdef.dir_key:
        base_dir = _get_dir_for_key(rdef.dir_key, rdef.resource_type)
        candidate = os.path.join(base_dir, rdef.filename)
        if os.path.isfile(candidate):
            filepath = os.path.abspath(candidate)

    if filepath:
        rt.full_path = filepath
        try:
            stat = os.stat(filepath)
            rt.file_size = stat.st_size
            rt.last_checked = stat.st_mtime
        except Exception:
            pass

        from modules.hash_cache import get_cached_hash, load_cache_from_file
        load_cache_from_file()
        rt.current_hash = get_cached_hash(filepath)
        rt.hash_status = _compute_hash_status(rdef, rt.current_hash)
    else:
        rt.full_path = ""
        rt.file_size = 0
        rt.current_hash = None
        rt.hash_status = HashStatus.NOT_COMPUTED

    dl_rt = download_tracker.get(info.key)
    if dl_rt:
        if dl_rt.download_status == DownloadStatus.DOWNLOADING:
            rt.download_status = DownloadStatus.DOWNLOADING
            rt.download_progress = dl_rt.download_progress
            rt.download_speed = dl_rt.download_speed
        elif dl_rt.download_status == DownloadStatus.FAILED:
            rt.download_status = DownloadStatus.FAILED
            rt.error_message = dl_rt.error_message


def _scan_directory_extra(rtype: ResourceType) -> List[ResourceInfo]:
    import modules.config as cfg
    from modules.extra_utils import get_files_from_folder

    extensions = ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch', '.pt']
    dirs = _get_all_dirs_for_type(rtype)
    result = []

    builtin_filenames = {r.filename for r in get_resources_by_type(rtype)}

    user_downloads_map = {}
    if rtype == ResourceType.CHECKPOINT:
        user_downloads_map = cfg.checkpoint_downloads
    elif rtype == ResourceType.LORA:
        user_downloads_map = cfg.lora_downloads
    elif rtype == ResourceType.VAE:
        user_downloads_map = cfg.vae_downloads
    elif rtype == ResourceType.EMBEDDING:
        user_downloads_map = cfg.embeddings_downloads

    for directory in dirs:
        if not directory or not os.path.exists(directory):
            continue
        try:
            filenames = get_files_from_folder(directory, extensions)
            for fn in filenames:
                base_fn = os.path.basename(fn)
                if base_fn in builtin_filenames:
                    continue
                is_user_configured = base_fn in user_downloads_map
                source_url = user_downloads_map.get(base_fn) if is_user_configured else None

                rdef = ResourceDef(
                    name=base_fn,
                    resource_type=rtype,
                    filename=base_fn,
                    source_url=source_url,
                    dir_key="",
                    expected_hash=None,
                    is_builtin=False,
                    description="User model" if not is_user_configured else "User configured download",
                )
                result.append(ResourceInfo(def_=rdef))
        except Exception:
            continue

    return result


def get_all_resources() -> Dict[str, ResourceInfo]:
    register_builtin_resources()
    result: Dict[str, ResourceInfo] = {}

    for key, rdef in _builtin_registry.items():
        rt = ResourceRuntime()
        info = ResourceInfo(def_=rdef, rt=rt)
        _update_runtime_for_resource(info)
        result[key] = info

    for rtype in ResourceType:
        extras = _scan_directory_extra(rtype)
        for info in extras:
            if info.key not in result:
                _update_runtime_for_resource(info)
                result[info.key] = info

    return result


def get_resources_list_by_type(rtype: ResourceType) -> List[ResourceInfo]:
    all_res = get_all_resources()
    return [r for r in all_res.values() if r.def_.resource_type == rtype]


def get_resources_summary() -> dict:
    all_res = get_all_resources()
    summary = {
        "total": len(all_res),
        "by_type": {},
        "by_status": {},
        "by_hash_status": {},
    }
    for r in all_res.values():
        t = r.def_.resource_type.value
        s = r.status.value
        h = r.rt.hash_status.value
        summary["by_type"][t] = summary["by_type"].get(t, 0) + 1
        summary["by_status"][s] = summary["by_status"].get(s, 0) + 1
        summary["by_hash_status"][h] = summary["by_hash_status"].get(h, 0) + 1
    return summary


def download_resource(resource_key: str, force: bool = False) -> bool:
    register_builtin_resources()

    info = None
    if resource_key in _builtin_registry:
        info = ResourceInfo(def_=_builtin_registry[resource_key], rt=ResourceRuntime())
        _update_runtime_for_resource(info)
    else:
        all_res = get_all_resources()
        if resource_key in all_res:
            info = all_res[resource_key]

    if not info:
        return False

    if not info.def_.source_url:
        return False

    if not force and info.status in (ResourceStatus.EXISTS, ResourceStatus.HASH_VERIFIED):
        return True

    key = resource_key
    model_dir = _get_dir_for_key(info.def_.dir_key, info.def_.resource_type)
    if not model_dir:
        dirs = _get_all_dirs_for_type(info.def_.resource_type)
        model_dir = dirs[0] if dirs else ""
    if not model_dir:
        return False

    download_tracker.start(key)

    def progress_cb(downloaded: int, total: int, speed: float):
        progress = downloaded / total if total > 0 else 0
        download_tracker.update(key, progress, speed)

    def run_download():
        from modules.model_loader import load_file_from_url
        from modules.hash_cache import sha256_from_cache, save_cache_to_file

        try:
            result_path = load_file_from_url(
                url=info.def_.source_url,
                model_dir=model_dir,
                file_name=info.def_.filename,
                progress=True,
                progress_callback=progress_cb,
            )
            if result_path and os.path.exists(result_path):
                try:
                    abs_p = os.path.abspath(result_path)
                    sha256_from_cache(abs_p)
                    save_cache_to_file()
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

    info = all_res[resource_key]
    if not info.rt.full_path or not os.path.isfile(info.rt.full_path):
        return False

    def run_hash():
        from modules.hash_cache import invalidate_cache, sha256_from_cache, save_cache_to_file
        try:
            abs_p = os.path.abspath(info.rt.full_path)
            invalidate_cache(abs_p)
            sha256_from_cache(abs_p)
            save_cache_to_file()
        except Exception:
            pass

    thread = threading.Thread(target=run_hash, daemon=True)
    thread.start()
    return True


def refresh_all_files() -> bool:
    import modules.config as cfg
    cfg.update_files()
    return True


def get_startup_download_list() -> List[Tuple[str, str, str]]:
    register_builtin_resources()
    result = []

    for key, rdef in _builtin_registry.items():
        if rdef.autodownload and rdef.source_url:
            model_dir = _get_dir_for_key(rdef.dir_key, rdef.resource_type)
            if model_dir:
                result.append((rdef.filename, rdef.source_url, model_dir))

    return result


def get_inpaint_head_path() -> str:
    register_builtin_resources()
    import modules.config as cfg
    return os.path.join(cfg.path_inpaint, "fooocus_inpaint_head.pth")


def get_inpaint_patch_path(version: str) -> str:
    register_builtin_resources()
    import modules.config as cfg
    filename_map = {
        "v1": "inpaint.fooocus.patch",
        "v2.5": "inpaint_v25.fooocus.patch",
        "v2.6": "inpaint_v26.fooocus.patch",
    }
    filename = filename_map.get(version, "inpaint_v26.fooocus.patch")
    return os.path.join(cfg.path_inpaint, filename)
