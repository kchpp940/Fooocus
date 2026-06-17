import args_manager
from multiprocessing import cpu_count

from modules.resource_service import (
    sha256_from_cache,
    init_hash_cache,
    rebuild_hash_cache,
)

hash_cache_filename = 'hash_cache.txt'
hash_cache = {}


def _sync_cache():
    global hash_cache
    from modules.resource_service import _hash_cache
    hash_cache = _hash_cache


def save_cache_to_file(filename=None, hash_value=None):
    if filename is not None and hash_value is not None:
        from modules.resource_service import _save_hash_cache_entry
        _save_hash_cache_entry(filename, hash_value)
    else:
        from modules.resource_service import _save_hash_cache_full
        _save_hash_cache_full()
    _sync_cache()


def load_cache_from_file():
    from modules.resource_service import _load_hash_cache_from_file
    _load_hash_cache_from_file()
    _sync_cache()


def init_cache(model_filenames=None, paths_checkpoints=None, lora_filenames=None, paths_loras=None):
    rebuild = 0
    if args_manager.args.rebuild_hash_cache:
        rebuild = args_manager.args.rebuild_hash_cache if args_manager.args.rebuild_hash_cache > 0 else cpu_count()
    init_hash_cache(rebuild=rebuild)
    _sync_cache()


def rebuild_cache(lora_filenames=None, model_filenames=None, paths_checkpoints=None, paths_loras=None, max_workers=cpu_count()):
    rebuild_hash_cache(max_workers=max_workers)
    _sync_cache()
