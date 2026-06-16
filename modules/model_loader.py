import os
import tempfile
from urllib.parse import urlparse
from typing import Optional


MIN_SANE_FILE_SIZES = {
    '.safetensors': 100 * 1024 * 1024,
    '.ckpt': 100 * 1024 * 1024,
    '.bin': 1 * 1024 * 1024,
    '.pth': 1 * 1024 * 1024,
    '.pt': 100 * 1024,
    '.fooocus.patch': 10 * 1024,
}
DEFAULT_MIN_SANE_SIZE = 10 * 1024


def _get_min_sane_size(filepath: str) -> int:
    ext = os.path.splitext(filepath)[1].lower()
    return MIN_SANE_FILE_SIZES.get(ext, DEFAULT_MIN_SANE_SIZE)


def is_file_sane(filepath: str) -> tuple[bool, str]:
    """Check if a file is sane (not zero bytes, not too small).

    Returns (is_sane, reason) tuple.
    """
    try:
        stat = os.stat(filepath)
    except OSError as e:
        return False, f'stat failed: {e}'

    if stat.st_size == 0:
        return False, 'file is zero bytes'

    min_size = _get_min_sane_size(filepath)
    if stat.st_size < min_size:
        return False, f'file too small ({stat.st_size} bytes < {min_size} bytes)'

    return True, 'ok'


def _is_file_sane(filepath: str) -> tuple[bool, str]:
    """Deprecated alias for is_file_sane."""
    return is_file_sane(filepath)


def _should_redownload(filepath: str, always_download: bool = False) -> tuple[bool, str]:
    if always_download:
        return True, 'forced redownload by preset flag'

    sane, reason = _is_file_sane(filepath)
    if not sane:
        return True, reason

    try:
        from modules.util import sha256
        from modules.hash_cache import hash_cache, _is_valid_hash, _validate_cache_entry
    except ImportError:
        return False, 'hash cache not available'

    try:
        current_hash = sha256(filepath)
    except Exception as e:
        return True, f'cannot compute hash: {e}'

    if not _is_valid_hash(current_hash):
        return True, f'computed hash is invalid: {current_hash}'

    if filepath in hash_cache:
        cached_hash = hash_cache[filepath]
        if not _validate_cache_entry(filepath, cached_hash):
            return True, 'cached hash is invalid'
        if cached_hash != current_hash:
            return True, f'hash mismatch (cached={cached_hash}, current={current_hash})'

    return False, 'file is valid'


def _refresh_hash_cache(filepath: str) -> None:
    try:
        from modules.hash_cache import hash_cache, save_cache_to_file, _hash_cache_lock, _validate_cache_entry
        from modules.util import sha256
    except ImportError:
        return

    try:
        current_hash = sha256(filepath)
    except Exception as e:
        print(f'[Download] Failed to compute hash for {filepath}: {e}')
        return

    with _hash_cache_lock:
        hash_cache[filepath] = current_hash
        save_cache_to_file(filepath, current_hash)
    print(f'[Download] Updated hash cache for {filepath}: {current_hash}')


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Uses atomic download pattern: downloads to a temporary .part file first,
    then atomically renames to the final filename. This prevents half-downloaded
    files from being treated as complete models.

    Performs integrity checks on existing files: zero-byte, too small, hash mismatch,
    or forced redownload will trigger a fresh download.

    Returns the path to the downloaded file.
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
        need_redownload, reason = _should_redownload(cached_file, always_download=always_download)
        if not need_redownload:
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

    sane, reason = _is_file_sane(temp_file)
    if not sane:
        try:
            os.remove(temp_file)
        except OSError:
            pass
        raise RuntimeError(f'Downloaded file failed integrity check: {reason}')

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
