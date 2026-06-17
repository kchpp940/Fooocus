import os
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple

from modules.model_resource_registry import (
    ResourceType, ResourceDef,
    get_registry, get_resource_def, get_resources_by_type, get_autodownload_list,
    resolve_dir, resolve_all_dirs,
)


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
        if self.rt.full_path:
            return os.path.dirname(self.rt.full_path)
        import modules.config as cfg
        return resolve_dir(self.def_, cfg)

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
                self._downloads[key].download_progress = progress
                self._downloads[key].download_speed = speed
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

    return HashStatus.COMPUTED_UNVERIFIED


def _update_runtime(info: ResourceInfo):
    rdef = info.def_
    rt = info.rt

    import modules.config as cfg
    from modules.util import get_file_from_folder_list

    dirs = resolve_all_dirs(rdef.resource_type, cfg)
    filepath = get_file_from_folder_list(rdef.filename, dirs)

    if not filepath and rdef.dir_key:
        base_dir = resolve_dir(rdef, cfg)
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
    dirs = resolve_all_dirs(rtype, cfg)
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
    registry = get_registry()
    result: Dict[str, ResourceInfo] = {}

    for key, rdef in registry.items():
        rt = ResourceRuntime()
        info = ResourceInfo(def_=rdef, rt=rt)
        _update_runtime(info)
        result[key] = info

    for rtype in ResourceType:
        extras = _scan_directory_extra(rtype)
        for info in extras:
            if info.key not in result:
                _update_runtime(info)
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
    rdef = get_resource_def(resource_key)
    if not rdef:
        all_res = get_all_resources()
        if resource_key in all_res:
            info = all_res[resource_key]
            rdef = info.def_
        else:
            return False

    if not rdef.source_url:
        return False

    import modules.config as cfg

    if not force:
        rt = ResourceRuntime()
        info = ResourceInfo(def_=rdef, rt=rt)
        _update_runtime(info)
        if info.status in (ResourceStatus.EXISTS, ResourceStatus.HASH_VERIFIED):
            return True

    key = resource_key
    model_dir = resolve_dir(rdef, cfg)
    if not model_dir:
        dirs = resolve_all_dirs(rdef.resource_type, cfg)
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
                url=rdef.source_url,
                model_dir=model_dir,
                file_name=rdef.filename,
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
    import modules.config as cfg

    result = []
    for rdef in get_autodownload_list():
        model_dir = resolve_dir(rdef, cfg)
        if model_dir:
            result.append((rdef.filename, rdef.source_url, model_dir))

    return result
