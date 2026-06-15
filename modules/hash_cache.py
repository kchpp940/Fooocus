import json
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import args_manager
from modules.util import sha256, HASH_SHA256_LENGTH, get_file_from_folder_list

hash_cache_filename = 'hash_cache.txt'
hash_cache = {}
_hash_cache_lock = threading.Lock()
_hash_valid_pattern = re.compile(r'^[0-9a-fA-F]+$')


def _is_valid_hash(hash_value):
    if not isinstance(hash_value, str):
        return False
    if len(hash_value) != HASH_SHA256_LENGTH:
        return False
    if not _hash_valid_pattern.match(hash_value):
        return False
    return True


def _validate_cache_entry(filepath, hash_value):
    if not isinstance(filepath, str) or not filepath:
        return False
    if not os.path.isfile(filepath):
        return False
    if not _is_valid_hash(hash_value):
        return False
    return True


def sha256_from_cache(filepath):
    global hash_cache

    if filepath in hash_cache:
        cached_hash = hash_cache[filepath]
        if _validate_cache_entry(filepath, cached_hash):
            return cached_hash
        print(f'[Cache] Invalidating stale cache entry for {filepath}')
        del hash_cache[filepath]

    print(f"[Cache] Calculating sha256 for {filepath}")
    hash_value = sha256(filepath)
    print(f"[Cache] sha256 for {filepath}: {hash_value}")

    with _hash_cache_lock:
        hash_cache[filepath] = hash_value
        save_cache_to_file(filepath, hash_value)

    return hash_value


def load_cache_from_file():
    global hash_cache

    loaded_entries = {}

    try:
        if os.path.exists(hash_cache_filename):
            with open(hash_cache_filename, 'rt', encoding='utf-8') as fp:
                line_no = 0
                for line in fp:
                    line_no += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f'[Cache] Skipping invalid JSON on line {line_no}: {e}')
                        continue

                    if not isinstance(entry, dict):
                        print(f'[Cache] Skipping non-dict entry on line {line_no}')
                        continue

                    for filepath, hash_value in entry.items():
                        if not _validate_cache_entry(filepath, hash_value):
                            print(f'[Cache] Skipping invalid cache entry on line {line_no}: {filepath}')
                            continue
                        loaded_entries[filepath] = hash_value
    except Exception as e:
        print(f'[Cache] Loading failed: {e}')
        loaded_entries = {}

    with _hash_cache_lock:
        hash_cache = loaded_entries
    print(f'[Cache] Loaded {len(hash_cache)} valid entries from {hash_cache_filename}')


def save_cache_to_file(filename=None, hash_value=None):
    global hash_cache

    with _hash_cache_lock:
        if filename is not None and hash_value is not None:
            items = [(filename, hash_value)]
            mode = 'at'
        else:
            items = sorted(hash_cache.items())
            mode = 'wt'

        try:
            if mode == 'wt':
                dir_name = os.path.dirname(os.path.abspath(hash_cache_filename)) or '.'
                fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix='.hash_cache_', suffix='.tmp')
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as fp:
                        for filepath, hv in items:
                            json.dump({filepath: hv}, fp)
                            fp.write('\n')
                    os.replace(temp_path, hash_cache_filename)
                except Exception:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
                    raise
            else:
                with open(hash_cache_filename, mode, encoding='utf-8') as fp:
                    for filepath, hv in items:
                        json.dump({filepath: hv}, fp)
                        fp.write('\n')
        except Exception as e:
            print(f'[Cache] Saving failed: {e}')


def init_cache(model_filenames, paths_checkpoints, lora_filenames, paths_loras,
               vae_filenames=None, path_vae=None,
               embedding_filenames=None, path_embeddings=None):
    load_cache_from_file()

    if args_manager.args.rebuild_hash_cache:
        max_workers = args_manager.args.rebuild_hash_cache if args_manager.args.rebuild_hash_cache > 0 else cpu_count()
        rebuild_cache(
            lora_filenames, model_filenames, paths_checkpoints, paths_loras,
            vae_filenames, path_vae, embedding_filenames, path_embeddings, max_workers
        )

    with _hash_cache_lock:
        valid_entries = {}
        for filepath, hv in hash_cache.items():
            if _validate_cache_entry(filepath, hv):
                valid_entries[filepath] = hv
        hash_cache = valid_entries

    save_cache_to_file()


def rebuild_cache(lora_filenames, model_filenames, paths_checkpoints, paths_loras,
                  vae_filenames=None, path_vae=None,
                  embedding_filenames=None, path_embeddings=None,
                  max_workers=cpu_count()):
    global hash_cache

    print('[Cache] Rebuilding hash cache')

    def thread(filename, paths):
        filepath = get_file_from_folder_list(filename, paths)
        if os.path.isfile(filepath):
            sha256_from_cache(filepath)
        else:
            print(f'[Cache] Skipping missing file: {filename}')

    with _hash_cache_lock:
        hash_cache = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for model_filename in model_filenames:
            futures.append(executor.submit(thread, model_filename, paths_checkpoints))
        for lora_filename in lora_filenames:
            futures.append(executor.submit(thread, lora_filename, paths_loras))
        if vae_filenames and path_vae:
            for vae_filename in vae_filenames:
                futures.append(executor.submit(thread, vae_filename, path_vae))
        if embedding_filenames and path_embeddings:
            for embedding_filename in embedding_filenames:
                futures.append(executor.submit(thread, embedding_filename, path_embeddings))
        for future in futures:
            future.result()

    print('[Cache] Done rebuilding hash cache')
