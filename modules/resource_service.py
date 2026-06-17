import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Optional, List, Dict, Tuple, Callable, Any, Literal

import args_manager
from modules.resource_registry import (
    ResourceType,
    ResourceDefinition,
    get_resource_definition,
    get_resources_by_type,
    get_resource_type_config,
    RESOURCE_TYPE_CONFIG,
)
from modules.util import sha256, HASH_SHA256_LENGTH, get_files_from_folder
from modules.model_loader import load_file_from_url


DownloadStatus = Literal["idle", "downloading", "completed", "failed"]


@dataclass
class DirectoryMatch:
    name: str
    path: str
    directory_index: int
    directory_path: str
    size: Optional[int] = None
    last_modified: Optional[float] = None


@dataclass
class ResourceInstance:
    name: str
    resource_type: ResourceType
    path: str
    directory_index: int
    directory_path: str
    is_primary: bool
    all_matches: List[DirectoryMatch]
    hash: Optional[str] = None
    hash_verified: bool = False
    size: Optional[int] = None
    last_modified: Optional[float] = None


@dataclass
class DownloadState:
    resource_id: str
    status: DownloadStatus = "idle"
    progress: float = 0.0
    current_url: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_size: Optional[int] = None
    downloaded_size: Optional[int] = None


@dataclass
class HashCacheEntry:
    filepath: str
    hash: str
    last_modified: float
    size: int


class ResourceService:
    _instance: Optional["ResourceService"] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "ResourceService":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._config_provider: Optional[Any] = None
        self._scan_lock = threading.Lock()
        self._hash_lock = threading.Lock()
        self._download_locks: Dict[str, threading.Lock] = {}
        self._download_lock_lock = threading.Lock()

        self._scanned_instances: Dict[ResourceType, List[ResourceInstance]] = {}
        self._instance_name_index: Dict[Tuple[ResourceType, str], ResourceInstance] = {}
        self._hash_cache: Dict[str, HashCacheEntry] = {}
        self._download_states: Dict[str, DownloadState] = {}

        self._hash_cache_filename = "hash_cache.txt"

        self._observers: List[Callable[[], None]] = []

    def set_config_provider(self, config_provider: Any) -> None:
        self._config_provider = config_provider

    def add_observer(self, observer: Callable[[], None]) -> None:
        self._observers.append(observer)

    def remove_observer(self, observer: Callable[[], None]) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(self) -> None:
        for observer in self._observers:
            try:
                observer()
            except Exception as e:
                print(f"[ResourceService] Observer notification failed: {e}")

    def _get_directories_for_type(self, resource_type: ResourceType) -> List[str]:
        type_config = RESOURCE_TYPE_CONFIG.get(resource_type, {})
        path_config_key = type_config.get("path_config_key")
        if not path_config_key or self._config_provider is None:
            return []

        raw_value = getattr(self._config_provider, path_config_key, None)
        if raw_value is None:
            return []

        directories: List[str] = []
        if isinstance(raw_value, list):
            for path in raw_value:
                abs_path = os.path.abspath(path)
                if abs_path not in directories:
                    directories.append(abs_path)
        elif isinstance(raw_value, str):
            abs_path = os.path.abspath(raw_value)
            directories.append(abs_path)

        return directories

    def get_directories_for_type(self, resource_type: ResourceType) -> List[str]:
        return self._get_directories_for_type(resource_type)

    def _get_target_download_directory(self, resource_type: ResourceType) -> Optional[str]:
        directories = self._get_directories_for_type(resource_type)
        if not directories:
            return None
        target_dir = directories[0]
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def scan_all(self, force: bool = False) -> None:
        with self._scan_lock:
            for resource_type in ResourceType:
                self._scan_type_internal(resource_type, force)
            self._notify_observers()

    def scan_type(self, resource_type: ResourceType, force: bool = False) -> None:
        with self._scan_lock:
            self._scan_type_internal(resource_type, force)
            self._notify_observers()

    def _scan_type_internal(self, resource_type: ResourceType, force: bool = False) -> None:
        if not force and resource_type in self._scanned_instances:
            return

        type_config = RESOURCE_TYPE_CONFIG.get(resource_type, {})
        extensions = type_config.get("extensions", [])
        directories = self._get_directories_for_type(resource_type)

        matches_by_name: Dict[str, List[DirectoryMatch]] = {}

        for dir_index, directory in enumerate(directories):
            if not os.path.isdir(directory):
                continue
            try:
                files = get_files_from_folder(directory, extensions)
                for filename in files:
                    full_path = os.path.join(directory, filename)
                    if not os.path.isfile(full_path):
                        continue

                    try:
                        stat = os.stat(full_path)
                        match = DirectoryMatch(
                            name=filename,
                            path=full_path,
                            directory_index=dir_index,
                            directory_path=directory,
                            size=stat.st_size,
                            last_modified=stat.st_mtime,
                        )
                    except OSError:
                        match = DirectoryMatch(
                            name=filename,
                            path=full_path,
                            directory_index=dir_index,
                            directory_path=directory,
                        )

                    if filename not in matches_by_name:
                        matches_by_name[filename] = []
                    matches_by_name[filename].append(match)

            except Exception as e:
                print(f"[ResourceService] Scan error for {resource_type.value} in {directory}: {e}")

        instances: List[ResourceInstance] = []
        for filename, matches in matches_by_name.items():
            matches.sort(key=lambda m: m.directory_index)

            primary_match = matches[0]
            instance = ResourceInstance(
                name=filename,
                resource_type=resource_type,
                path=primary_match.path,
                directory_index=primary_match.directory_index,
                directory_path=primary_match.directory_path,
                is_primary=True,
                all_matches=matches,
                size=primary_match.size,
                last_modified=primary_match.last_modified,
            )
            instances.append(instance)

        instances.sort(key=lambda x: x.name.casefold())

        self._scanned_instances[resource_type] = instances
        self._instance_name_index.update(
            {(resource_type, inst.name): inst for inst in instances}
        )

    def get_instances_by_type(self, resource_type: ResourceType) -> List[ResourceInstance]:
        if resource_type not in self._scanned_instances:
            self.scan_type(resource_type)
        return self._scanned_instances.get(resource_type, [])

    def get_filenames_by_type(self, resource_type: ResourceType) -> List[str]:
        instances = self.get_instances_by_type(resource_type)
        return [inst.name for inst in instances]

    def get_resource_instance(
        self, resource_type: ResourceType, filename: str
    ) -> Optional[ResourceInstance]:
        key = (resource_type, filename)
        if key not in self._instance_name_index:
            self.scan_type(resource_type)
        return self._instance_name_index.get(key)

    def get_all_matches_for_name(
        self, resource_type: ResourceType, filename: str
    ) -> List[DirectoryMatch]:
        instance = self.get_resource_instance(resource_type, filename)
        if instance is None:
            return []
        return list(instance.all_matches)

    def get_filepath(
        self, resource_type: ResourceType, filename: str
    ) -> Optional[str]:
        instance = self.get_resource_instance(resource_type, filename)
        return instance.path if instance else None

    def find_filepath_by_name(
        self, resource_type: ResourceType, filename: str
    ) -> Optional[str]:
        directories = self._get_directories_for_type(resource_type)
        if not directories:
            return None

        for directory in directories:
            full_path = os.path.abspath(os.path.realpath(os.path.join(directory, filename)))
            if os.path.isfile(full_path):
                return full_path

        first_dir = directories[0]
        return os.path.abspath(os.path.realpath(os.path.join(first_dir, filename)))

    def find_filepath_by_name_all(
        self, resource_type: ResourceType, filename: str
    ) -> List[Tuple[str, bool]]:
        directories = self._get_directories_for_type(resource_type)
        if not directories:
            return []

        results: List[Tuple[str, bool]] = []
        found_first = False
        for directory in directories:
            full_path = os.path.abspath(os.path.realpath(os.path.join(directory, filename)))
            exists = os.path.isfile(full_path)
            is_primary = exists and not found_first
            if exists:
                found_first = True
                results.append((full_path, is_primary))
            else:
                results.append((full_path, False))
        return results

    def load_hash_cache(self) -> None:
        with self._hash_lock:
            self._hash_cache.clear()
            try:
                if os.path.exists(self._hash_cache_filename):
                    with open(self._hash_cache_filename, "rt", encoding="utf-8") as fp:
                        for line in fp:
                            try:
                                entry = json.loads(line)
                                for filepath, hash_value in entry.items():
                                    if not os.path.exists(filepath):
                                        continue
                                    if not isinstance(hash_value, str) or len(hash_value) != HASH_SHA256_LENGTH:
                                        continue
                                    try:
                                        stat = os.stat(filepath)
                                        self._hash_cache[filepath] = HashCacheEntry(
                                            filepath=filepath,
                                            hash=hash_value,
                                            last_modified=stat.st_mtime,
                                            size=stat.st_size,
                                        )
                                    except OSError:
                                        continue
                            except (json.JSONDecodeError, ValueError):
                                continue
            except Exception as e:
                print(f"[ResourceService] Load hash cache failed: {e}")

    def save_hash_cache(self) -> None:
        with self._hash_lock:
            try:
                with open(self._hash_cache_filename, "wt", encoding="utf-8") as fp:
                    for entry in sorted(self._hash_cache.values(), key=lambda x: x.filepath):
                        json.dump({entry.filepath: entry.hash}, fp)
                        fp.write("\n")
            except Exception as e:
                print(f"[ResourceService] Save hash cache failed: {e}")

    def get_hash(
        self,
        filepath: str,
        force_recompute: bool = False,
        use_addnet_hash: bool = False,
    ) -> Optional[str]:
        if not os.path.isfile(filepath):
            return None

        filepath = os.path.abspath(filepath)

        with self._hash_lock:
            if not force_recompute and filepath in self._hash_cache:
                entry = self._hash_cache[filepath]
                try:
                    stat = os.stat(filepath)
                    if stat.st_mtime == entry.last_modified and stat.st_size == entry.size:
                        return entry.hash
                except OSError:
                    pass

            print(f"[ResourceService] Calculating hash for {filepath}")
            hash_value = sha256(filepath, use_addnet_hash=use_addnet_hash)
            print(f"[ResourceService] Hash for {filepath}: {hash_value}")

            try:
                stat = os.stat(filepath)
                self._hash_cache[filepath] = HashCacheEntry(
                    filepath=filepath,
                    hash=hash_value,
                    last_modified=stat.st_mtime,
                    size=stat.st_size,
                )
                self._save_hash_cache_entry(filepath, hash_value)
            except OSError:
                pass

            return hash_value

    def _save_hash_cache_entry(self, filepath: str, hash_value: str) -> None:
        try:
            with open(self._hash_cache_filename, "at", encoding="utf-8") as fp:
                json.dump({filepath: hash_value}, fp)
                fp.write("\n")
        except Exception as e:
            print(f"[ResourceService] Save hash entry failed: {e}")

    def rebuild_hash_cache(
        self,
        resource_types: Optional[List[ResourceType]] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        if resource_types is None:
            resource_types = [ResourceType.CHECKPOINT, ResourceType.LORA]

        if max_workers is None or max_workers <= 0:
            max_workers = cpu_count()

        all_files: List[Tuple[str, bool]] = []
        for resource_type in resource_types:
            instances = self.get_instances_by_type(resource_type)
            use_addnet = resource_type == ResourceType.LORA
            for instance in instances:
                all_files.append((instance.path, use_addnet))

        print(f"[ResourceService] Rebuilding hash cache for {len(all_files)} files")

        def calc_hash(filepath: str, use_addnet: bool) -> Tuple[str, Optional[str]]:
            return filepath, self.get_hash(filepath, force_recompute=True, use_addnet_hash=use_addnet)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(calc_hash, fp, ua) for fp, ua in all_files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ResourceService] Hash calculation error: {e}")

        self.save_hash_cache()
        print("[ResourceService] Hash cache rebuild complete")

    def verify_resource_hash(self, resource_id: str) -> bool:
        definition = get_resource_definition(resource_id)
        if not definition or not definition.expected_hash:
            return True

        filepath = self.find_filepath_by_name(definition.resource_type, definition.name)
        if not filepath or not os.path.isfile(filepath):
            return False

        actual_hash = self.get_hash(filepath)
        return actual_hash == definition.expected_hash

    def _get_download_lock(self, resource_id: str) -> threading.Lock:
        with self._download_lock_lock:
            if resource_id not in self._download_locks:
                self._download_locks[resource_id] = threading.Lock()
            return self._download_locks[resource_id]

    def get_download_state(self, resource_id: str) -> DownloadState:
        if resource_id not in self._download_states:
            self._download_states[resource_id] = DownloadState(resource_id=resource_id)
        return self._download_states[resource_id]

    def is_downloaded(self, resource_id: str) -> bool:
        definition = get_resource_definition(resource_id)
        if not definition:
            return False
        filepath = self.find_filepath_by_name(definition.resource_type, definition.name)
        return filepath is not None and os.path.isfile(filepath)

    def download(
        self,
        resource_id: str,
        on_progress: Optional[Callable[[DownloadState], None]] = None,
    ) -> bool:
        definition = get_resource_definition(resource_id)
        if not definition:
            print(f"[ResourceService] Resource not found: {resource_id}")
            return False

        if self.is_downloaded(resource_id):
            state = self.get_download_state(resource_id)
            state.status = "completed"
            state.progress = 100.0
            if on_progress:
                on_progress(state)
            return True

        lock = self._get_download_lock(resource_id)
        if not lock.acquire(blocking=False):
            print(f"[ResourceService] Download already in progress: {resource_id}")
            return False

        try:
            state = self.get_download_state(resource_id)
            state.status = "downloading"
            state.progress = 0.0
            state.started_at = time.time()
            state.error = None

            if on_progress:
                on_progress(state)

            model_dir = self._get_target_download_directory(definition.resource_type)
            if model_dir is None:
                state.status = "failed"
                state.error = f"No target directory for resource type {definition.resource_type.value}"
                if on_progress:
                    on_progress(state)
                return False

            for url in definition.urls:
                try:
                    state.current_url = url
                    if on_progress:
                        on_progress(state)

                    print(f"[ResourceService] Downloading {definition.name} from {url}")
                    downloaded_path = load_file_from_url(
                        url=url,
                        model_dir=model_dir,
                        file_name=definition.name,
                        progress=True,
                    )

                    if os.path.isfile(downloaded_path):
                        self.scan_type(definition.resource_type, force=True)
                        self.get_hash(downloaded_path, force_recompute=True)

                        state.status = "completed"
                        state.progress = 100.0
                        state.completed_at = time.time()
                        if on_progress:
                            on_progress(state)

                        self._notify_observers()
                        return True

                except Exception as e:
                    print(f"[ResourceService] Download failed from {url}: {e}")
                    state.error = str(e)
                    continue

            state.status = "failed"
            if on_progress:
                on_progress(state)
            return False

        finally:
            lock.release()

    def download_by_definition(
        self,
        definition: ResourceDefinition,
        on_progress: Optional[Callable[[DownloadState], None]] = None,
    ) -> bool:
        if definition.resource_id in [r.resource_id for r in get_resources_by_type(definition.resource_type)]:
            return self.download(definition.resource_id, on_progress)

        resource_id = f"custom_{definition.resource_type.value}_{definition.name}"
        return self._download_custom(definition, resource_id, on_progress)

    def _download_custom(
        self,
        definition: ResourceDefinition,
        resource_id: str,
        on_progress: Optional[Callable[[DownloadState], None]] = None,
    ) -> bool:
        lock = self._get_download_lock(resource_id)
        if not lock.acquire(blocking=False):
            return False

        try:
            state = self.get_download_state(resource_id)
            state.status = "downloading"
            state.progress = 0.0
            state.started_at = time.time()
            state.error = None

            if on_progress:
                on_progress(state)

            existing_path = self.find_filepath_by_name(definition.resource_type, definition.name)
            if existing_path and os.path.isfile(existing_path):
                self.scan_type(definition.resource_type, force=True)
                state.status = "completed"
                state.progress = 100.0
                state.completed_at = time.time()
                if on_progress:
                    on_progress(state)
                self._notify_observers()
                return True

            model_dir = self._get_target_download_directory(definition.resource_type)
            if model_dir is None:
                state.status = "failed"
                state.error = f"No target directory for resource type {definition.resource_type.value}"
                if on_progress:
                    on_progress(state)
                return False

            for url in definition.urls:
                try:
                    state.current_url = url
                    if on_progress:
                        on_progress(state)

                    print(f"[ResourceService] Downloading {definition.name} from {url}")
                    downloaded_path = load_file_from_url(
                        url=url,
                        model_dir=model_dir,
                        file_name=definition.name,
                        progress=True,
                    )

                    if os.path.isfile(downloaded_path):
                        self.scan_type(definition.resource_type, force=True)

                        state.status = "completed"
                        state.progress = 100.0
                        state.completed_at = time.time()
                        if on_progress:
                            on_progress(state)

                        self._notify_observers()
                        return True

                except Exception as e:
                    print(f"[ResourceService] Download failed from {url}: {e}")
                    state.error = str(e)
                    continue

            state.status = "failed"
            if on_progress:
                on_progress(state)
            return False

        finally:
            lock.release()

    def download_batch(
        self,
        resource_ids: List[str],
        on_progress: Optional[Callable[[str, DownloadState], None]] = None,
    ) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for resource_id in resource_ids:
            def progress_callback(state: DownloadState, rid: str = resource_id) -> None:
                if on_progress:
                    on_progress(rid, state)

            results[resource_id] = self.download(resource_id, progress_callback)
        return results

    def download_from_config(
        self,
        resource_type: ResourceType,
        downloads_dict: Dict[str, str],
        on_progress: Optional[Callable[[str, DownloadState], None]] = None,
    ) -> None:
        for filename, url in downloads_dict.items():
            definition = ResourceDefinition(
                resource_id=f"config_{resource_type.value}_{filename}",
                resource_type=resource_type,
                name=filename,
                urls=[url],
            )
            self.download_by_definition(
                definition,
                on_progress=lambda s, f=filename: on_progress(f, s) if on_progress else None,
            )

    def ensure_core_resources(self) -> None:
        from modules.resource_registry import (
            VAE_APPROX_RESOURCES,
            FOOOCUS_EXPANSION_RESOURCES,
        )

        core_resources = VAE_APPROX_RESOURCES + FOOOCUS_EXPANSION_RESOURCES
        for resource in core_resources:
            if not self.is_downloaded(resource.resource_id):
                self.download(resource.resource_id)

    def download_by_name(
        self,
        resource_type: ResourceType,
        filename: str,
        url: Optional[str] = None,
        on_progress: Optional[Callable[[DownloadState], None]] = None,
    ) -> Optional[bool]:
        definitions = get_resources_by_type(resource_type)
        for d in definitions:
            if d.name == filename:
                return self.download(d.resource_id, on_progress)

        if url:
            definition = ResourceDefinition(
                resource_id=f"manual_{resource_type.value}_{filename}",
                resource_type=resource_type,
                name=filename,
                urls=[url],
            )
            return self.download_by_definition(definition, on_progress)

        print(f"[ResourceService] No registered resource and no URL provided: {filename}")
        return None

    def rehash_by_name(
        self,
        resource_type: ResourceType,
        filename: str,
    ) -> Optional[str]:
        filepath = self.find_filepath_by_name(resource_type, filename)
        if not filepath or not os.path.isfile(filepath):
            return None
        return self.get_hash(resource_type, filename, force_recompute=True)

    def get_all_download_states(self) -> Dict[str, DownloadState]:
        return dict(self._download_states)

    def get_resource_status(
        self, resource_type: Optional[ResourceType] = None
    ) -> Dict[str, Dict[str, Any]]:
        if resource_type is None:
            result: Dict[str, Dict[str, Any]] = {}
            for rt in ResourceType:
                result[rt.value] = self.get_resource_status(rt)
            return result

        instances = self.get_instances_by_type(resource_type)
        definitions = get_resources_by_type(resource_type)

        registered_names = {d.name: d for d in definitions}
        scanned_names = {i.name: i for i in instances}

        all_names = set(registered_names.keys()) | set(scanned_names.keys())

        status_dict: Dict[str, Dict[str, Any]] = {}
        for name in sorted(all_names):
            definition = registered_names.get(name)
            instance = scanned_names.get(name)

            status = {
                "name": name,
                "is_registered": definition is not None,
                "is_scanned": instance is not None,
                "is_downloaded": instance is not None,
                "path": instance.path if instance else None,
                "directory_index": instance.directory_index if instance else None,
                "directory_path": instance.directory_path if instance else None,
                "is_primary_path": instance.is_primary if instance else None,
                "hash": instance.hash if instance else None,
                "size": instance.size if instance else None,
                "description": definition.description if definition else None,
                "urls": definition.urls if definition else [],
            }

            if instance:
                all_hits_info = []
                for match in instance.all_matches:
                    all_hits_info.append({
                        "path": match.path,
                        "directory_index": match.directory_index,
                        "directory_path": match.directory_path,
                        "size": match.size,
                        "last_modified": match.last_modified,
                        "is_primary": match.directory_index == instance.directory_index,
                    })
                status["all_matches"] = all_hits_info

            if definition:
                download_state = self.get_download_state(definition.resource_id)
                status["download_status"] = download_state.status
                status["download_progress"] = download_state.progress
                status["download_error"] = download_state.error

            status_dict[name] = status

        return status_dict

    def get_resource_summary(self) -> Dict[str, Any]:
        summary = {
            "by_type": {},
        }

        for resource_type in ResourceType:
            instances = self.get_instances_by_type(resource_type)
            definitions = get_resources_by_type(resource_type)
            downloaded = [r for r in definitions if self.is_downloaded(r.resource_id)]

            directories = self._get_directories_for_type(resource_type)

            summary["by_type"][resource_type.value] = {
                "registered": len(definitions),
                "scanned": len(instances),
                "downloaded": len(downloaded),
                "directories": directories,
                "primary_directory": directories[0] if directories else None,
            }

        return summary


def get_resource_service() -> ResourceService:
    return ResourceService()
