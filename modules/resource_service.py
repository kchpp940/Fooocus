import os
import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Dict, List, Optional

from modules.resource_registry import (
    ResourceCategory, ResourceType, RESOURCE_TYPE_MAP,
    HASHED_CATEGORIES, INPAINT_VERSION_PATCH_URLS,
    PERFORMANCE_LORA_URLS, IP_ADAPTER_URLS,
    CHECKPOINT, LORA, VAE, VAE_APPROX, EMBEDDING, INPAINT,
    CONTROLNET, CLIP_VISION, UPSCALE_MODEL, FOOOCUS_EXPANSION,
    SAFETY_CHECKER, SAM, WILDCARD,
)
from modules.model_loader import load_file_from_url


_hash_cache: Dict[str, str] = {}
_hash_cache_filename = "hash_cache.txt"
_path_registry: Dict[ResourceCategory, List[str]] = {}
_file_registry: Dict[ResourceCategory, List[str]] = {}
_download_registry: Dict[str, str] = {}
_initialized = False


def initialize(paths_map: Dict[ResourceCategory, List[str]]) -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True
    for category, paths in paths_map.items():
        _path_registry[category] = paths
    _load_hash_cache_from_file()
    refresh_all_files()
    from modules.model_loader import register_download_callback
    register_download_callback(_on_file_downloaded)


def register_path(category: ResourceCategory, paths: List[str]) -> None:
    _path_registry[category] = paths


def _on_file_downloaded(filepath: str, url: str) -> None:
    category = _infer_category_from_path(filepath)
    if category is not None and category in RESOURCE_TYPE_MAP:
        _file_registry[category] = scan_files(RESOURCE_TYPE_MAP[category])


def _infer_category_from_path(filepath: str) -> Optional[ResourceCategory]:
    for category, paths in _path_registry.items():
        for p in paths:
            if filepath.startswith(p):
                return category
    return None


def get_paths(category: ResourceCategory) -> List[str]:
    return _path_registry.get(category, [])


def get_path(category: ResourceCategory) -> str:
    paths = _path_registry.get(category, [])
    return paths[0] if paths else ""


def set_paths(category: ResourceCategory, paths: List[str]) -> None:
    _path_registry[category] = paths


def refresh_all_files() -> None:
    global _file_registry
    for rt in [CHECKPOINT, LORA, VAE, WILDCARD]:
        _file_registry[rt.category] = scan_files(rt)


def scan_files(rt: ResourceType) -> List[str]:
    paths = _path_registry.get(rt.category, [])
    if not paths:
        return []
    files = []
    for folder in paths:
        if os.path.isdir(folder):
            for root, _, fs in os.walk(folder, topdown=False):
                relative_path = os.path.relpath(root, folder)
                if relative_path == ".":
                    relative_path = ""
                for filename in sorted(fs, key=lambda s: s.casefold()):
                    _, file_extension = os.path.splitext(filename)
                    if rt.extensions is None or file_extension.lower() in rt.extensions:
                        if rt.name_filter is None or rt.name_filter in filename:
                            path = os.path.join(relative_path, filename)
                            files.append(path)
    return files


def get_filenames(category: ResourceCategory) -> List[str]:
    return _file_registry.get(category, [])


def get_file_path(filename: str, category: ResourceCategory) -> str:
    paths = _path_registry.get(category, [])
    if not isinstance(paths, list):
        paths = [paths]
    for folder in paths:
        full = os.path.abspath(os.path.realpath(os.path.join(folder, filename)))
        if os.path.isfile(full):
            return full
    if paths:
        return os.path.abspath(os.path.realpath(os.path.join(paths[0], filename)))
    return filename


def download_known_file(category: ResourceCategory, filename: str, progress: bool = True) -> str:
    rt = RESOURCE_TYPE_MAP.get(category)
    if rt is None:
        raise ValueError(f"Unknown resource category: {category}")
    url = rt.known_urls.get(filename) or _download_registry.get(filename)
    if url is None:
        raise ValueError(f"No known URL for {filename} in {category.value}")
    model_dir = get_path(category)
    return load_file_from_url(url=url, model_dir=model_dir, file_name=filename, progress=progress)


def download_file(url: str, category: ResourceCategory, filename: Optional[str] = None, progress: bool = True) -> str:
    model_dir = get_path(category)
    return load_file_from_url(url=url, model_dir=model_dir, file_name=filename, progress=progress)


def download_inpaint_models(v: str) -> tuple:
    assert v in ["v1", "v2.5", "v2.6"]
    download_known_file(ResourceCategory.INPAINT, "fooocus_inpaint_head.pth")
    head_file = os.path.join(get_path(ResourceCategory.INPAINT), "fooocus_inpaint_head.pth")
    patch_file = None
    patch_filename = INPAINT_VERSION_PATCH_URLS.get(v)
    if patch_filename:
        download_known_file(ResourceCategory.INPAINT, patch_filename)
        patch_file = os.path.join(get_path(ResourceCategory.INPAINT), patch_filename)
    return head_file, patch_file


def download_controlnet_canny() -> str:
    download_known_file(ResourceCategory.CONTROLNET, "control-lora-canny-rank128.safetensors")
    return os.path.join(get_path(ResourceCategory.CONTROLNET), "control-lora-canny-rank128.safetensors")


def download_controlnet_cpds() -> str:
    download_known_file(ResourceCategory.CONTROLNET, "fooocus_xl_cpds_128.safetensors")
    return os.path.join(get_path(ResourceCategory.CONTROLNET), "fooocus_xl_cpds_128.safetensors")


def download_ip_adapters(v: str) -> list:
    assert v in ["ip", "face"]
    results = []
    entries = IP_ADAPTER_URLS.get(v, [])
    for filename, url, category in entries:
        download_known_file(category, filename)
        results.append(os.path.join(get_path(category), filename))
    return results


def download_upscale_model() -> str:
    download_known_file(ResourceCategory.UPSCALE_MODEL, "fooocus_upscaler_s409985e5.bin")
    return os.path.join(get_path(ResourceCategory.UPSCALE_MODEL), "fooocus_upscaler_s409985e5.bin")


def download_safety_checker_model() -> str:
    download_known_file(ResourceCategory.SAFETY_CHECKER, "stable-diffusion-safety-checker.bin")
    return os.path.join(get_path(ResourceCategory.SAFETY_CHECKER), "stable-diffusion-safety-checker.bin")


def download_sam_model(sam_model: str) -> str:
    filename_map = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }
    if sam_model not in filename_map:
        raise ValueError(f"sam model {sam_model} does not exist.")
    filename = filename_map[sam_model]
    download_known_file(ResourceCategory.SAM, filename)
    return os.path.join(get_path(ResourceCategory.SAM), filename)


def download_performance_lora(performance_key: str) -> str:
    if performance_key not in PERFORMANCE_LORA_URLS:
        raise ValueError(f"Unknown performance key: {performance_key}")
    filename, url = PERFORMANCE_LORA_URLS[performance_key]
    download_known_file(ResourceCategory.LORA, filename)
    return filename


def download_vae_approx_files() -> None:
    for filename in VAE_APPROX.known_urls:
        download_known_file(ResourceCategory.VAE_APPROX, filename)


def download_expansion_model() -> None:
    download_known_file(ResourceCategory.FOOOCUS_EXPANSION, "pytorch_model.bin")


def download_preset_models(
    default_model: str,
    previous_default_models: list,
    checkpoint_downloads: dict,
    embeddings_downloads: dict,
    lora_downloads: dict,
    vae_downloads: dict,
    disable_preset_download: bool = False,
    always_download_new_model: bool = False,
) -> tuple:
    download_vae_approx_files()
    download_expansion_model()

    if disable_preset_download:
        print("Skipped model download.")
        return default_model, checkpoint_downloads

    if not always_download_new_model:
        if not _file_exists_in_category(default_model, ResourceCategory.CHECKPOINT):
            for alternative_model_name in previous_default_models:
                if _file_exists_in_category(alternative_model_name, ResourceCategory.CHECKPOINT):
                    print(f"You do not have [{default_model}] but you have [{alternative_model_name}].")
                    print(f"Fooocus will use [{alternative_model_name}] to avoid downloading new models, "
                          f"but you are not using the latest models.")
                    print("Use --always-download-new-model to avoid fallback and always get new models.")
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    for file_name, url in checkpoint_downloads.items():
        download_file(url, ResourceCategory.CHECKPOINT, file_name)
    for file_name, url in embeddings_downloads.items():
        download_file(url, ResourceCategory.EMBEDDING, file_name)
    for file_name, url in lora_downloads.items():
        download_file(url, ResourceCategory.LORA, file_name)
    for file_name, url in vae_downloads.items():
        download_file(url, ResourceCategory.VAE, file_name)

    return default_model, checkpoint_downloads


def _file_exists_in_category(filename: str, category: ResourceCategory) -> bool:
    filepath = get_file_path(filename, category)
    return os.path.isfile(filepath)


def register_download_url(filename: str, url: str) -> None:
    _download_registry[filename] = url


def sha256_from_cache(filepath: str) -> str:
    global _hash_cache
    if filepath not in _hash_cache:
        from modules.util import sha256, HASH_SHA256_LENGTH
        print(f"[Cache] Calculating sha256 for {filepath}")
        hash_value = sha256(filepath)
        print(f"[Cache] sha256 for {filepath}: {hash_value}")
        _hash_cache[filepath] = hash_value
        _save_hash_cache_entry(filepath, hash_value)
    return _hash_cache[filepath]


def init_hash_cache(rebuild: int = 0) -> None:
    if rebuild:
        max_workers = rebuild if rebuild > 0 else cpu_count()
        rebuild_hash_cache(max_workers)
    _save_hash_cache_full()


def rebuild_hash_cache(max_workers: int = 0) -> None:
    if max_workers <= 0:
        max_workers = cpu_count()

    def _compute_hash(filename, paths):
        filepath = get_file_path(filename, ResourceCategory.CHECKPOINT)
        sha256_from_cache(filepath)

    print("[Cache] Rebuilding hash cache")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filename in get_filenames(ResourceCategory.CHECKPOINT):
            executor.submit(_compute_hash, filename, get_paths(ResourceCategory.CHECKPOINT))
        for filename in get_filenames(ResourceCategory.LORA):
            executor.submit(_compute_hash, filename, get_paths(ResourceCategory.LORA))
    print("[Cache] Done")


def _load_hash_cache_from_file() -> None:
    global _hash_cache
    from modules.util import HASH_SHA256_LENGTH
    try:
        if os.path.exists(_hash_cache_filename):
            with open(_hash_cache_filename, "rt", encoding="utf-8") as fp:
                for line in fp:
                    entry = json.loads(line)
                    for filepath, hash_value in entry.items():
                        if not os.path.exists(filepath) or not isinstance(hash_value, str) or len(hash_value) != HASH_SHA256_LENGTH:
                            print(f"[Cache] Skipping invalid cache entry: {filepath}")
                            continue
                        _hash_cache[filepath] = hash_value
    except Exception as e:
        print(f"[Cache] Loading failed: {e}")


def _save_hash_cache_entry(filepath: str, hash_value: str) -> None:
    try:
        with open(_hash_cache_filename, "at", encoding="utf-8") as fp:
            json.dump({filepath: hash_value}, fp)
            fp.write("\n")
    except Exception as e:
        print(f"[Cache] Saving failed: {e}")


def _save_hash_cache_full() -> None:
    global _hash_cache
    try:
        items = sorted(_hash_cache.items())
        with open(_hash_cache_filename, "wt", encoding="utf-8") as fp:
            for filepath, hash_value in items:
                json.dump({filepath: hash_value}, fp)
                fp.write("\n")
    except Exception as e:
        print(f"[Cache] Saving failed: {e}")
