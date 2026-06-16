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
_hash_cache_lock = threading.RLock()
_file_write_lock = threading.Lock()
_hash_valid_pattern = re.compile(r'^[0-9a-fA-F]+$')


def is_valid_hash(hash_value):
    """Public: check if a hash string is well-formed.

    Returns True iff hash_value is a string of correct length and hex chars only.
    """
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
    if not is_valid_hash(hash_value):
        return False
    return True


def get_cached_hash(filepath):
    """Public: get the cached hash for a single file.

    Returns the cached hash string if present and valid, otherwise None.
    Does NOT compute a new hash or modify the cache.
    """
    with _hash_cache_lock:
        if filepath in hash_cache:
            cached = hash_cache[filepath]
            if _validate_cache_entry(filepath, cached):
                return cached
    return None


def _update_memory_entry(filepath, new_hash):
    """Update the in-memory cache dict only (no disk I/O).

    Caller must hold _hash_cache_lock.

    Returns:
        (old_hash, is_new) tuple. old_hash is None if no previous entry.
        is_new is True if this is a new entry or the hash changed.
    """
    old_hash = hash_cache.get(filepath)
    hash_cache[filepath] = new_hash

    if old_hash is None:
        return None, True
    if old_hash != new_hash:
        return old_hash, True
    return old_hash, False


def _append_single_entry_to_disk(filepath, hash_value):
    """Append a single {filepath: hash_value} line to hash_cache.txt.

    Uses _file_write_lock to prevent line interleaving in multi-threaded
    appends. Does NOT touch _hash_cache_lock.
    """
    with _file_write_lock:
        try:
            with open(hash_cache_filename, 'at', encoding='utf-8') as fp:
                json.dump({filepath: hash_value}, fp)
                fp.write('\n')
        except Exception as e:
            print(f'[Cache] Append to {hash_cache_filename} failed: {e}')


def _rewrite_entire_cache_to_disk(items):
    """Atomically rewrite the entire hash_cache.txt from a sorted list of items.

    Uses tempfile + os.replace for atomicity. Uses _file_write_lock to
    exclude concurrent appenders. Does NOT touch _hash_cache_lock
    (caller should snapshot memory first).
    """
    dir_name = os.path.dirname(os.path.abspath(hash_cache_filename)) or '.'

    with _file_write_lock:
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


def refresh_cache_entry(filepath, force=False, defer_write=False):
    """Public: recompute and update the cache entry for a single file.

    This is the unified entry point for refreshing a file's hash.

    Lock discipline:
    - Hash computation happens OUTSIDE any lock (I/O heavy, does not need
      the memory dict to be stable).
    - Memory update happens INSIDE _hash_cache_lock (fast, in-memory only).
    - Disk write happens OUTSIDE _hash_cache_lock (protected by
      _file_write_lock only, so we never hold both locks simultaneously).

    Args:
        filepath: Absolute path to the model file.
        force: If True, recompute even if a valid cached entry exists.
        defer_write: If True, update memory but skip disk write. Used by
            rebuild_cache() to batch I/O: all threads update memory, then
            the main thread does one full atomic rewrite.

    Returns:
        The new (or existing valid) hash string, or None on error.
    """
    global hash_cache

    filepath = os.path.abspath(filepath)

    if not force:
        existing = get_cached_hash(filepath)
        if existing is not None:
            return existing

    if not os.path.isfile(filepath):
        print(f'[Cache] Cannot refresh hash: file does not exist {filepath}')
        return None

    try:
        new_hash = sha256(filepath)
    except Exception as e:
        print(f'[Cache] Failed to compute sha256 for {filepath}: {e}')
        return None

    if not is_valid_hash(new_hash):
        print(f'[Cache] Computed hash is invalid for {filepath}: {new_hash}')
        return None

    with _hash_cache_lock:
        old_hash, changed = _update_memory_entry(filepath, new_hash)

    if changed and old_hash is not None:
        print(f'[Cache] Hash updated for {filepath}: {old_hash} -> {new_hash}')
    elif changed:
        print(f'[Cache] Hash cached for {filepath}: {new_hash}')

    if changed and not defer_write:
        _append_single_entry_to_disk(filepath, new_hash)

    return new_hash


def sha256_from_cache(filepath):
    """Public: get the sha256 for a file, using cache when available.

    Falls back to recomputing via refresh_cache_entry.
    """
    cached = get_cached_hash(filepath)
    if cached is not None:
        return cached
    return refresh_cache_entry(filepath, force=True)


def load_cache_from_file():
    """Load hash_cache.txt into memory, skipping invalid entries."""
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
    """Save cache to disk.

    Two modes:
    1. Single-entry append (filename and hash_value provided):
       - Uses _file_write_lock only
       - Appends one line
       - Does not take _hash_cache_lock
    2. Full rewrite (no arguments):
       - Takes _hash_cache_lock to snapshot memory
       - Releases memory lock before writing
       - Uses _file_write_lock + tempfile + os.replace for atomic rewrite

    This function is safe to call from outside callers and is the
    canonical way to persist the cache.
    """
    global hash_cache

    if filename is not None and hash_value is not None:
        _append_single_entry_to_disk(filename, hash_value)
        return

    with _hash_cache_lock:
        items = sorted(hash_cache.items())

    try:
        _rewrite_entire_cache_to_disk(items)
    except Exception as e:
        print(f'[Cache] Full rewrite failed: {e}')


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
    """Rebuild the entire cache from scratch.

    I/O optimization: each worker thread only updates the in-memory dict
    (defer_write=True). When all threads are done, the main thread does
    ONE atomic full rewrite of hash_cache.txt. This reduces O(N) I/O
    operations to O(1), which matters for large model collections.
    """
    global hash_cache

    print('[Cache] Rebuilding hash cache')

    def thread(filename, paths):
        filepath = get_file_from_folder_list(filename, paths)
        if os.path.isfile(filepath):
            refresh_cache_entry(filepath, force=True, defer_write=True)
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

    print('[Cache] All hashes computed, writing full cache to disk ...')
    save_cache_to_file()
    print('[Cache] Done rebuilding hash cache')
