import json
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import args_manager
from modules.util import sha256, HASH_SHA256_LENGTH, get_file_from_folder_list
from modules.resource_service import get_resource_service, ResourceType

hash_cache_filename = 'hash_cache.txt'
hash_cache = {}

_resource_service = get_resource_service()


def sha256_from_cache(filepath):
    return _resource_service.get_hash(filepath)


def load_cache_from_file():
    _resource_service.load_hash_cache()
    global hash_cache
    hash_cache = {entry.filepath: entry.hash for entry in _resource_service._hash_cache.values()}


def save_cache_to_file(filename=None, hash_value=None):
    if filename is not None and hash_value is not None:
        _resource_service._save_hash_cache_entry(filename, hash_value)
    else:
        _resource_service.save_hash_cache()


def init_cache(model_filenames, paths_checkpoints, lora_filenames, paths_loras):
    load_cache_from_file()

    if args_manager.args.rebuild_hash_cache:
        max_workers = args_manager.args.rebuild_hash_cache if args_manager.args.rebuild_hash_cache > 0 else cpu_count()
        rebuild_cache(lora_filenames, model_filenames, paths_checkpoints, paths_loras, max_workers)

    save_cache_to_file()


def rebuild_cache(lora_filenames, model_filenames, paths_checkpoints, paths_loras, max_workers=cpu_count()):
    def thread(filename, paths):
        filepath = get_file_from_folder_list(filename, paths)
        sha256_from_cache(filepath)

    print('[Cache] Rebuilding hash cache')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model_filename in model_filenames:
            executor.submit(thread, model_filename, paths_checkpoints)
        for lora_filename in lora_filenames:
            executor.submit(thread, lora_filename, paths_loras)
    print('[Cache] Done')
