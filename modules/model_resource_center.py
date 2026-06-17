import os
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple

from modules.model_resource_registry import (
    ResourceType, ResourceDef,
    get_registry, get_resource_def, get_resources_by_type, get_autodownload_list,
    resolve_dir, resolve_all_dirs, resolve_all_dirs_for_def, resolve_all_possible_paths,
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
class PathMatch:
    full_path: str = ""
    directory: str = ""
    dir_priority: int = 0
    exists: bool = False
    file_size: int = 0
    current_hash: Optional[str] = None
    hash_status: HashStatus = HashStatus.NOT_COMPUTED
    last_checked: float = 0.0
    is_primary: bool = False


@dataclass
class ResourceRuntime:
    all_matches: List[PathMatch] = field(default_factory=list)
    primary_index: int = -1
    download_status: DownloadStatus = DownloadStatus.IDLE
    download_progress: float = 0.0
    download_speed: float = 0.0
    error_message: str = ""
    download_target_path: str = ""

    @property
    def primary(self) -> Optional[PathMatch]:
        if 0 <= self.primary_index < len(self.all_matches):
            return self.all_matches[self.primary_index]
        for m in self.all_matches:
            if m.exists:
                return m
        return None

    @property
    def existing_matches(self) -> List[PathMatch]:
        return [m for m in self.all_matches if m.exists]

    @property
    def has_duplicates(self) -> bool:
        return len(self.existing_matches) > 1


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
        primary = self.rt.primary
        if not primary or not primary.exists:
            return ResourceStatus.MISSING
        if primary.hash_status == HashStatus.VERIFIED:
            return ResourceStatus.HASH_VERIFIED
        if primary.hash_status == HashStatus.MISMATCH:
            return ResourceStatus.HASH_MISMATCH
        return ResourceStatus.EXISTS

    @property
    def primary_path(self) -> str:
        p = self.rt.primary
        return p.full_path if p else ""

    @property
    def primary_directory(self) -> str:
        p = self.rt.primary
        return p.directory if p else ""

    @property
    def primary_file_size(self) -> int:
        p = self.rt.primary
        return p.file_size if p else 0

    @property
    def primary_hash(self) -> Optional[str]:
        p = self.rt.primary
        return p.current_hash if p else None

    @property
    def primary_hash_status(self) -> HashStatus:
        p = self.rt.primary
        return p.hash_status if p else HashStatus.NOT_COMPUTED

    def to_dict(self) -> dict:
        primary = self.rt.primary
        return {
            "key": self.key,
            "name": self.def_.name,
            "resource_type": self.def_.resource_type.value,
            "filename": self.def_.filename,
            "source_url": self.def_.source_url,
            "expected_hash": self.def_.expected_hash,
            "is_builtin": self.def_.is_builtin,
            "description": self.def_.description,
            "status": self.status.value,
            "has_duplicates": self.rt.has_duplicates,
            "num_matches": len(self.rt.existing_matches),
            "num_possible_paths": len(self.rt.all_matches),
            "primary": {
                "full_path": primary.full_path if primary else "",
                "directory": primary.directory if primary else "",
                "dir_priority": primary.dir_priority if primary else -1,
                "file_size": primary.file_size if primary else 0,
                "file_size_human": format_size(primary.file_size) if primary else "0 B",
                "current_hash": primary.current_hash if primary else None,
                "hash_status": primary.hash_status.value if primary else HashStatus.NOT_COMPUTED.value,
                "last_checked": primary.last_checked if primary else 0,
            },
            "all_matches": [
                {
                    "full_path": m.full_path,
                    "directory": m.directory,
                    "dir_priority": m.dir_priority,
                    "exists": m.exists,
                    "is_primary": m.is_primary,
                    "file_size": m.file_size,
                    "file_size_human": format_size(m.file_size),
                    "current_hash": m.current_hash,
                    "hash_status": m.hash_status.value,
                    "last_checked": m.last_checked,
                }
                for m in self.rt.all_matches
            ],
            "all_possible_dirs": [m.directory for m in self.rt.all_matches],
            "download_status": self.rt.download_status.value,
            "download_progress": self.rt.download_progress,
            "download_speed": self.rt.download_speed,
            "download_target_path": self.rt.download_target_path,
            "error_message": self.rt.error_message,
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

    def start(self, key: str, target_path: str = ""):
        with self._lock:
            if key not in self._downloads:
                self._downloads[key] = ResourceRuntime()
            rt = self._downloads[key]
            rt.download_status = DownloadStatus.DOWNLOADING
            rt.download_progress = 0.0
            rt.download_speed = 0.0
            rt.error_message = ""
            rt.download_target_path = target_path
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


def _scan_path_match(rdef: ResourceDef, full_path: str, dir_priority: int, hash_cache_module) -> PathMatch:
    match = PathMatch(
        full_path=os.path.abspath(os.path.realpath(full_path)) if os.path.exists(full_path) else full_path,
        directory=os.path.dirname(full_path),
        dir_priority=dir_priority,
    )

    if os.path.isfile(full_path):
        match.exists = True
        try:
            stat = os.stat(full_path)
            match.file_size = stat.st_size
            match.last_checked = stat.st_mtime
        except Exception:
            pass

        if hash_cache_module:
            try:
                match.current_hash = hash_cache_module.get_cached_hash(match.full_path)
            except Exception:
                match.current_hash = None
            match.hash_status = _compute_hash_status(rdef, match.current_hash)

    return match


def _update_runtime(info: ResourceInfo):
    rdef = info.def_
    rt = info.rt

    import modules.config as cfg
    from modules.hash_cache import get_cached_hash, load_cache_from_file
    load_cache_from_file()

    possible_paths = resolve_all_possible_paths(rdef, cfg)

    rt.all_matches = []
    rt.primary_index = -1

    for idx, path in enumerate(possible_paths):
        match = _scan_path_match(rdef, path, idx, hash_cache_module=None)
        if match.exists:
            try:
                match.current_hash = get_cached_hash(match.full_path)
            except Exception:
                match.current_hash = None
            match.hash_status = _compute_hash_status(rdef, match.current_hash)
        rt.all_matches.append(match)

    for idx, match in enumerate(rt.all_matches):
        if match.exists and rt.primary_index == -1:
            rt.primary_index = idx
            match.is_primary = True
            break

    dl_rt = download_tracker.get(info.key)
    if dl_rt:
        if dl_rt.download_status == DownloadStatus.DOWNLOADING:
            rt.download_status = DownloadStatus.DOWNLOADING
            rt.download_progress = dl_rt.download_progress
            rt.download_speed = dl_rt.download_speed
            rt.download_target_path = dl_rt.download_target_path
        elif dl_rt.download_status == DownloadStatus.FAILED:
            rt.download_status = DownloadStatus.FAILED
            rt.error_message = dl_rt.error_message
            rt.download_target_path = dl_rt.download_target_path
        elif dl_rt.download_status == DownloadStatus.SUCCESS:
            rt.download_status = DownloadStatus.IDLE
            rt.download_target_path = ""


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

    seen_files = set()

    for dir_idx, directory in enumerate(dirs):
        if not directory or not os.path.exists(directory):
            continue
        try:
            filenames = get_files_from_folder(directory, extensions)
            for fn in filenames:
                base_fn = os.path.basename(fn)
                if base_fn in builtin_filenames:
                    continue
                abs_fn = os.path.abspath(os.path.realpath(fn))
                if abs_fn in seen_files:
                    continue
                seen_files.add(abs_fn)

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
        "with_duplicates": 0,
    }
    for r in all_res.values():
        t = r.def_.resource_type.value
        s = r.status.value
        primary = r.rt.primary
        h = primary.hash_status.value if primary else HashStatus.NOT_COMPUTED.value
        summary["by_type"][t] = summary["by_type"].get(t, 0) + 1
        summary["by_status"][s] = summary["by_status"].get(s, 0) + 1
        summary["by_hash_status"][h] = summary["by_hash_status"].get(h, 0) + 1
        if r.rt.has_duplicates:
            summary["with_duplicates"] += 1
    return summary


def download_resource(resource_key: str, target_path: Optional[str] = None, force: bool = False) -> bool:
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
            if not target_path or target_path == info.primary_path:
                return True

    dirs = resolve_all_dirs_for_def(rdef, cfg)
    if not dirs:
        return False

    model_dir = dirs[0]
    if target_path:
        abs_target = os.path.abspath(target_path)
        if os.path.isdir(abs_target):
            model_dir = abs_target
        else:
            model_dir = os.path.dirname(abs_target)
    else:
        for d in dirs:
            if d and os.path.isdir(d):
                model_dir = d
                break

    if not model_dir:
        return False

    target_full_path = os.path.join(model_dir, rdef.filename)
    download_tracker.start(resource_key, target_full_path)

    def progress_cb(downloaded: int, total: int, speed: float):
        progress = downloaded / total if total > 0 else 0
        download_tracker.update(resource_key, progress, speed)

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
                download_tracker.finish(resource_key, True)
            else:
                download_tracker.finish(resource_key, False, "Download returned empty path")
        except Exception as e:
            download_tracker.finish(resource_key, False, str(e))

    thread = threading.Thread(target=run_download, daemon=True)
    thread.start()
    return True


def rehash_resource(resource_key: str, target_path: Optional[str] = None) -> bool:
    all_res = get_all_resources()
    if resource_key not in all_res:
        return False

    info = all_res[resource_key]

    paths_to_rehash = []
    if target_path:
        abs_target = os.path.abspath(target_path)
        for m in info.rt.all_matches:
            if os.path.abspath(m.full_path) == abs_target and m.exists:
                paths_to_rehash.append(m.full_path)
                break
        if not paths_to_rehash:
            if os.path.isfile(abs_target):
                paths_to_rehash.append(abs_target)
    else:
        for m in info.rt.existing_matches:
            paths_to_rehash.append(m.full_path)

    if not paths_to_rehash:
        return False

    def run_hash():
        from modules.hash_cache import invalidate_cache, sha256_from_cache, save_cache_to_file
        for p in paths_to_rehash:
            try:
                abs_p = os.path.abspath(p)
                invalidate_cache(abs_p)
                sha256_from_cache(abs_p)
            except Exception:
                pass
        try:
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
        dirs = resolve_all_dirs_for_def(rdef, cfg)
        if dirs:
            model_dir = dirs[0]
            for d in dirs:
                if d and os.path.isdir(d):
                    model_dir = d
                    break
            result.append((rdef.filename, rdef.source_url, model_dir))

    return result
