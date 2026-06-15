import os
import tempfile
from urllib.parse import urlparse
from typing import Optional


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

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    if os.path.exists(cached_file):
        return cached_file

    temp_file = cached_file + '.part'

    if os.path.exists(temp_file):
        print(f'[Download] Resuming interrupted download: removing stale temp file {temp_file}')
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

    try:
        os.replace(temp_file, cached_file)
    except OSError as e:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except OSError:
            pass
        raise RuntimeError(f'Failed to finalize download: {e}')

    return cached_file
