import cv2
import os
import os.path as osp
import torch
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def is_file_readable(filepath):
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


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def load_file_from_url(url, model_dir=None, progress=True, file_name=None, save_dir=None,
                       expected_size=None, expected_hash=None):
    """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Uses atomic download pattern: downloads to a temporary .part file first,
    then atomically renames to the final filename.

    Performs integrity checks on existing files:
    - Always: basic readability (non-zero bytes, readable)
    - If expected_size provided: exact size match
    - If expected_hash provided: exact hash match

    Args:
        url: URL to download from.
        model_dir: Model directory name (relative to torch hub dir).
        progress: Whether to show progress.
        file_name: Override filename.
        save_dir: Override save directory (absolute).
        expected_size: Expected file size in bytes. Triggers redownload on mismatch.
        expected_hash: Expected SHA256 hash prefix. Triggers redownload on mismatch.
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, model_dir)
    os.makedirs(save_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(save_dir, filename))

    if os.path.exists(cached_file):
        ok, reason = is_file_readable(cached_file)
        if ok and expected_size is not None:
            try:
                actual = os.path.getsize(cached_file)
                if actual != expected_size:
                    ok = False
                    reason = f'size mismatch (expected {expected_size}, got {actual})'
            except OSError as e:
                ok = False
                reason = f'stat failed: {e}'
        if ok and expected_hash is not None:
            import hashlib
            try:
                h = hashlib.sha256()
                with open(cached_file, 'rb') as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(chunk)
                actual_hash = h.hexdigest()[:len(expected_hash)]
                if actual_hash != expected_hash:
                    ok = False
                    reason = f'hash mismatch (expected {expected_hash}, got {actual_hash})'
            except Exception as e:
                ok = False
                reason = f'cannot compute hash: {e}'

        if ok:
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

    try:
        download_url_to_file(url, temp_file, hash_prefix=None, progress=progress)
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
        try:
            actual = os.path.getsize(temp_file)
            if actual != expected_size:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
                raise RuntimeError(
                    f'Downloaded file size mismatch: expected {expected_size}, got {actual}')
        except OSError as e:
            try:
                os.remove(temp_file)
            except OSError:
                pass
            raise RuntimeError(f'Downloaded file stat failed: {e}')

    if expected_hash is not None:
        import hashlib
        try:
            h = hashlib.sha256()
            with open(temp_file, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            actual_hash = h.hexdigest()[:len(expected_hash)]
            if actual_hash != expected_hash:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
                raise RuntimeError(
                    f'Downloaded file hash mismatch: expected {expected_hash}, got {actual_hash}')
        except Exception as e:
            try:
                os.remove(temp_file)
            except OSError:
                pass
            raise RuntimeError(f'Downloaded file hash check failed: {e}')

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


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
