import os
import tempfile
from urllib.parse import urlparse
from typing import Optional


def is_file_readable(filepath: str) -> tuple[bool, str]:
    """Basic sanity check: file exists, is a regular file, and is readable.

    Returns (is_ok, reason) tuple.
    """
    try:
        stat = os.stat(filepath)
    except OSError as e:
        return False, f'stat failed: {e}'

    if stat.st_size == 0:
        return False, 'file is zero bytes'

    try:
        with open(filepath, 'rb') as f:
            f.read(1)
    except OSError as e:
        return False, f'file is not readable: {e}'

    return True, 'ok'


def _check_expected_size(filepath: str, expected_size: int) -> tuple[bool, str]:
    """Check if file size matches expected size.

    Returns (matches, reason) tuple.
    """
    try:
        actual_size = os.path.getsize(filepath)
    except OSError as e:
        return False, f'stat failed: {e}'

    if actual_size != expected_size:
        return False, f'size mismatch (expected {expected_size}, got {actual_size})'

    return True, 'ok'


def _check_expected_hash(filepath: str, expected_hash: str) -> tuple[bool, str]:
    """Check if file hash matches expected hash.

    Returns (matches, reason) tuple.
    """
    from modules.util import sha256
    from modules.hash_cache import _is_valid_hash

    if not _is_valid_hash(expected_hash):
        return False, f'expected hash is invalid: {expected_hash}'

    try:
        actual_hash = sha256(filepath)
    except Exception as e:
        return False, f'cannot compute hash: {e}'

    if actual_hash != expected_hash:
        return False, f'hash mismatch (expected {expected_hash}, got {actual_hash})'

    return True, 'ok'


def _refresh_hash_cache(filepath: str) -> None:
    """Recalculate and update hash cache for a file.

    Does not delete the file; just refreshes the cache entry.
    """
    try:
        from modules.hash_cache import hash_cache, save_cache_to_file, _hash_cache_lock
        from modules.util import sha256
    except ImportError:
        return

    try:
        current_hash = sha256(filepath)
    except Exception as e:
        print(f'[Cache] Failed to compute hash for {filepath}: {e}')
        return

    with _hash_cache_lock:
        old_hash = hash_cache.get(filepath)
        hash_cache[filepath] = current_hash
        save_cache_to_file(filepath, current_hash)

    if old_hash and old_hash != current_hash:
        print(f'[Cache] Hash updated for {filepath}: {old_hash} -> {current_hash}')
    else:
        print(f'[Cache] Hash cached for {filepath}: {current_hash}')


def _verify_existing_file(
        filepath: str,
        *,
        expected_size: Optional[int] = None,
        expected_hash: Optional[str] = None,
) -> tuple[bool, str]:
    """Verify an existing file is usable.

    Checks (in order):
    1. Basic readability (non-zero, readable)
    2. Expected size (if provided)
    3. Expected hash (if provided)

    Hash cache mismatch does NOT trigger redownload; we just refresh the cache.

    Returns (is_valid, reason) tuple.
    """
    ok, reason = is_file_readable(filepath)
    if not ok:
        return False, reason

    if expected_size is not None:
        ok, reason = _check_expected_size(filepath, expected_size)
        if not ok:
            return False, reason

    if expected_hash is not None:
        ok, reason = _check_expected_hash(filepath, expected_hash)
        if not ok:
            return False, reason

    try:
        from modules.hash_cache import hash_cache, _validate_cache_entry
    except ImportError:
        return True, 'file is valid (hash cache not available)'

    if filepath in hash_cache:
        cached_hash = hash_cache[filepath]
        if not _validate_cache_entry(filepath, cached_hash):
            print(f'[Cache] Invalid cache entry for {filepath}, refreshing ...')
            _refresh_hash_cache(filepath)
        else:
            from modules.util import sha256
            try:
                current_hash = sha256(filepath)
            except Exception:
                current_hash = None
            if current_hash and cached_hash != current_hash:
                print(f'[Cache] Hash diverged for {filepath}, refreshing cache ...')
                _refresh_hash_cache(filepath)

    return True, 'file is valid'


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
        expected_size: Optional[int] = None,
        expected_hash: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if valid.

    Uses atomic download pattern: downloads to a temporary .part file first,
    then atomically renames to the final filename. This prevents half-downloaded
    files from being treated as complete models.

    Existing file validation (layered):
    - Always: basic readability check (non-zero bytes, readable)
    - If expected_size provided: exact size match
    - If expected_hash provided: exact hash match
    - Hash cache mismatch does NOT trigger redownload; cache is simply refreshed.

    Args:
        url: URL to download from.
        model_dir: Directory to save the file in.
        progress: Whether to show a progress bar.
        file_name: Override the filename from the URL.
        expected_size: Expected file size in bytes. Triggers redownload on mismatch.
        expected_hash: Expected SHA256 hash prefix. Triggers redownload on mismatch.

    Returns:
        Path to the downloaded (or existing valid) file.
    """
    import args_manager

    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    always_download = getattr(args_manager.args, 'always_download_new_model', False)

    if os.path.exists(cached_file):
        if always_download:
            print(f'[Download] Forced redownload of {cached_file} (always_download_new_model)')
            try:
                os.remove(cached_file)
            except OSError as e:
                print(f'[Download] Warning: could not remove file {cached_file}: {e}')
        else:
            valid, reason = _verify_existing_file(
                cached_file,
                expected_size=expected_size,
                expected_hash=expected_hash,
            )
            if valid:
                return cached_file
            print(f'[Download] Re-downloading {cached_file}: {reason}')
            try:
                os.remove(cached_file)
            except OSError as e:
                print(f'[Download] Warning: could not remove invalid file {cached_file}: {e}')

    temp_file = cached_file + '.part'

    if os.path.exists(temp_file):
        print(f'[Download] Removing stale temp file {temp_file}')
        try:
            os.remove(temp_file)
        except OSError:
            pass

    print(f'Downloading: "{url}" to {cached_file}\n')
    from torch.hub import download_url_to_file

    try:
        download_url_to_file(url, temp_file, progress=progress)
    except Exception:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass
        raise

    if not os.path.exists(temp_file):
        raise RuntimeError(f'Download failed: temp file {temp_file} was not created')

    ok, reason = is_file_readable(temp_file)
    if not ok:
        try:
            os.remove(temp_file)
        except OSError:
            pass
        raise RuntimeError(f'Downloaded file failed basic integrity check: {reason}')

    if expected_size is not None:
        ok, reason = _check_expected_size(temp_file, expected_size)
        if not ok:
            try:
                os.remove(temp_file)
            except OSError:
                pass
            raise RuntimeError(f'Downloaded file failed size check: {reason}')

    if expected_hash is not None:
        ok, reason = _check_expected_hash(temp_file, expected_hash)
        if not ok:
            try:
                os.remove(temp_file)
            except OSError:
                pass
            raise RuntimeError(f'Downloaded file failed hash check: {reason}')

    try:
        os.replace(temp_file, cached_file)
    except OSError as e:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except OSError:
            pass
        raise RuntimeError(f'Failed to finalize download: {e}')

    _refresh_hash_cache(cached_file)

    return cached_file
