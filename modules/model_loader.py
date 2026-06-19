import os
from urllib.parse import urlparse
from typing import Optional

import modules.diagnostics as diagnostics
from modules.diagnostics import (
    DiagnosticStage, DiagnosticErrorCategory, get_current_context,
    log_info, log_error,
)


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    This function is the central download entry point. It consults the
    ManifestResolver (if initialized) to:
    1. Redirect the download to the manifest-declared target directory.
    2. Verify file hashes when manifest declares them.
    3. Block downloads in offline/strict mode when the file is missing.

    Returns the path to the downloaded (or cached) file.
    """
    from modules.manifest import ManifestResolver, ManifestResolutionError

    resolver = ManifestResolver.get()
    item = None

    if resolver:
        resolved_dir, resolved_name, item = resolver.resolve_download(url, model_dir, file_name)
        model_dir = resolved_dir
        file_name = resolved_name

    ctx = get_current_context()
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url_resolved = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url_resolved)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    if not os.path.exists(cached_file):
        if resolver and resolver.strict:
            block_msg = resolver.check_before_download(url, cached_file, item)
            if block_msg:
                print(block_msg)
                raise ManifestResolutionError(block_msg, item=item, url=url)

        log_info(
            DiagnosticStage.MODEL_DOWNLOAD,
            f"开始下载模型: {file_name}",
            ctx=ctx,
            extra_data={
                "url": url_resolved,
                "target_dir": model_dir,
                "target_file": file_name,
            },
        )
        print(f'Downloading: "{url_resolved}" to {cached_file}\n')
        try:
            if ctx:
                ctx.start_stage(DiagnosticStage.MODEL_DOWNLOAD)
            from torch.hub import download_url_to_file
            download_url_to_file(url_resolved, cached_file, progress=progress)
            if ctx:
                ctx.add_model_loaded(cached_file)
                log_info(
                    DiagnosticStage.MODEL_DOWNLOAD,
                    f"模型下载完成: {file_name}",
                    ctx=ctx,
                    extra_data={"file_size": os.path.getsize(cached_file) if os.path.exists(cached_file) else None},
                )
                ctx.end_stage(DiagnosticStage.MODEL_DOWNLOAD, "completed")
        except Exception as e:
            log_error(
                DiagnosticStage.MODEL_DOWNLOAD,
                f"模型下载失败: {file_name}",
                exception=e,
                category=DiagnosticErrorCategory.DOWNLOAD_FAILED,
                ctx=ctx,
                extra_data={"url": url_resolved, "target_file": cached_file},
            )
            if os.path.exists(cached_file):
                try:
                    os.remove(cached_file)
                except OSError:
                    pass
            raise diagnostics.DiagnosticsError(
                human_message=f"模型下载失败: {file_name}",
                category=DiagnosticErrorCategory.DOWNLOAD_FAILED,
                stage=DiagnosticStage.MODEL_DOWNLOAD,
                ctx=ctx,
                extra_data={"file_name": file_name, "url": url_resolved},
                cause=e,
            )
    else:
        log_info(
            DiagnosticStage.RESOURCE_SCAN,
            f"使用已缓存的模型文件: {file_name}",
            ctx=ctx,
            extra_data={"file_path": cached_file},
        )

    if resolver and resolver.check_hash:
        if item is None:
            item = resolver.lookup_by_url(url) or resolver.lookup_by_path(cached_file)
        if item and not resolver.verify_hash(cached_file, item):
            print(f"[Manifest] WARNING: Hash verification failed for {file_name}")

    return cached_file
