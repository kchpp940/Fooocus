import os
from urllib.parse import urlparse
from typing import Optional

from modules.diagnostics import (
    DiagnosticJob, DiagnosticJobError, DiagnosticStage,
    DiagnosticErrorCategory, LogLevel,
)


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
        job: DiagnosticJob,
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
        job.record_event(
            LogLevel.INFO,
            DiagnosticStage.MODEL_DOWNLOAD,
            f"开始下载模型: {file_name}",
            extra_data={
                "url": url,
                "target_dir": model_dir,
                "target_file": file_name,
            },
        )
        print(f'Downloading: "{url}" to {cached_file}\n')
        try:
            _scope = job.scope(DiagnosticStage.MODEL_DOWNLOAD, DiagnosticErrorCategory.DOWNLOAD_FAILED)
            with _scope:
                from torch.hub import download_url_to_file
                download_url_to_file(url, cached_file, progress=progress)
                job.add_model_loaded(cached_file)
                job.record_event(
                    LogLevel.INFO,
                    DiagnosticStage.MODEL_DOWNLOAD,
                    f"模型下载完成: {file_name}",
                    extra_data={"file_size": os.path.getsize(cached_file) if os.path.exists(cached_file) else None},
                )
        except Exception as e:
            job.record_error(
                DiagnosticStage.MODEL_DOWNLOAD,
                f"模型下载失败: {file_name}",
                category=DiagnosticErrorCategory.DOWNLOAD_FAILED,
                exception=e,
                extra_data={"url": url, "target_file": cached_file},
            )
            if os.path.exists(cached_file):
                try:
                    os.remove(cached_file)
                except OSError:
                    pass
            raise DiagnosticJobError(
                human_message=f"模型下载失败: {file_name}",
                category=DiagnosticErrorCategory.DOWNLOAD_FAILED,
                stage=DiagnosticStage.MODEL_DOWNLOAD,
                extra_data={"file_name": file_name, "url": url},
                cause=e,
            )
    else:
        job.record_event(
            LogLevel.INFO,
            DiagnosticStage.RESOURCE_SCAN,
            f"使用已缓存的模型文件: {file_name}",
            extra_data={"file_path": cached_file},
        )
    return cached_file


def load_file_from_url_bootstrap(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    return load_file_from_url(url, model_dir=model_dir, progress=progress, file_name=file_name, job=DiagnosticJob())
