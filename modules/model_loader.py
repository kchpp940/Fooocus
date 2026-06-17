import os
import ssl
import time
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from typing import Optional, Callable


def _download_with_progress(url: str, dst: str, progress_callback: Optional[Callable[[int, int, float], None]] = None):
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    except Exception:
        ctx = None

    headers = {"User-Agent": "Fooocus/1.0"}
    req = Request(url, headers=headers)

    if ctx is not None:
        response = urlopen(req, context=ctx)
    else:
        response = urlopen(req)

    total_size = response.headers.get("Content-Length")
    total_size = int(total_size) if total_size else 0

    downloaded = 0
    chunk_size = 8192
    last_update_time = time.time()
    last_downloaded = 0

    tmp_dst = dst + ".downloading"

    with open(tmp_dst, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if progress_callback and (now - last_update_time >= 0.2 or downloaded == total_size):
                elapsed = now - last_update_time if now > last_update_time else 0.001
                speed = (downloaded - last_downloaded) / elapsed if elapsed > 0 else 0
                progress = downloaded / total_size if total_size > 0 else 0
                try:
                    progress_callback(downloaded, total_size, speed)
                except Exception:
                    pass
                last_update_time = now
                last_downloaded = downloaded

    if os.path.exists(tmp_dst):
        os.replace(tmp_dst, dst)

    return dst


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        if progress_callback is not None:
            _download_with_progress(url, cached_file, progress_callback)
        else:
            from torch.hub import download_url_to_file
            download_url_to_file(url, cached_file, progress=progress)
    return cached_file
