import os
import sys
import json
import uuid
import time
import traceback
import threading
import logging
from typing import Optional, Any, Dict, List
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DiagnosticStage(Enum):
    REQUEST_INIT = "request_init"
    RESOURCE_SCAN = "resource_scan"
    MODEL_LOAD = "model_load"
    MODEL_DOWNLOAD = "model_download"
    LORA_LOAD = "lora_load"
    PROMPT_PROCESS = "prompt_process"
    CLIP_ENCODE = "clip_encode"
    CONTROLNET_PREPROCESS = "controlnet_preprocess"
    IP_ADAPTER_PREPROCESS = "ip_adapter_preprocess"
    INPAINT_PREPROCESS = "inpaint_preprocess"
    VAE_ENCODE = "vae_encode"
    DIFFUSION = "diffusion"
    UPSCALE = "upscale"
    METADATA_PARSE = "metadata_parse"
    IMAGE_SAVE = "image_save"
    NSFW_CHECK = "nsfw_check"
    WEBCALLBACK = "webcallback"
    ENHANCE = "enhance"
    CLEANUP = "cleanup"
    UNKNOWN = "unknown"


class DiagnosticErrorCategory(Enum):
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_CORRUPTED = "model_corrupted"
    MODEL_LOAD_FAILED = "model_load_failed"
    LORA_LOAD_FAILED = "lora_load_failed"
    DOWNLOAD_FAILED = "download_failed"
    RESOURCE_SCAN_FAILED = "resource_scan_failed"
    METADATA_PARSE_FAILED = "metadata_parse_failed"
    CONTROLNET_PREPROCESS_FAILED = "controlnet_preprocess_failed"
    IP_ADAPTER_PREPROCESS_FAILED = "ip_adapter_preprocess_failed"
    INPAINT_PREPROCESS_FAILED = "inpaint_preprocess_failed"
    PROMPT_TOO_LONG = "prompt_too_long"
    GPU_OUT_OF_MEMORY = "gpu_out_of_memory"
    IMAGE_SAVE_FAILED = "image_save_failed"
    DIFFUSION_FAILED = "diffusion_failed"
    VAE_ENCODE_FAILED = "vae_encode_failed"
    CLIP_ENCODE_FAILED = "clip_encode_failed"
    UPSCALE_FAILED = "upscale_failed"
    NSFW_CHECK_FAILED = "nsfw_check_failed"
    ENHANCE_FAILED = "enhance_failed"
    USER_INTERRUPTED = "user_interrupted"
    INVALID_INPUT = "invalid_input"
    UNKNOWN_ERROR = "unknown_error"


_DIAGNOSTICS_ENABLED = True
_SENSITIVE_DATA_REDACTED = True
_PROMPT_MAX_CHARS_FOR_LOG = 120
_OUTPUT_FORMAT = "json"
_MAX_RETAINED_TASKS = 500
_LOG_TO_FILE = False
_LOG_FILE_PATH: Optional[str] = None


def set_diagnostics_enabled(enabled: bool):
    global _DIAGNOSTICS_ENABLED
    _DIAGNOSTICS_ENABLED = enabled


def is_diagnostics_enabled() -> bool:
    return _DIAGNOSTICS_ENABLED


def set_sensitive_data_redacted(redacted: bool):
    global _SENSITIVE_DATA_REDACTED
    _SENSITIVE_DATA_REDACTED = redacted


def is_sensitive_data_redacted() -> bool:
    return _SENSITIVE_DATA_REDACTED


def set_prompt_max_chars(chars: int):
    global _PROMPT_MAX_CHARS_FOR_LOG
    _PROMPT_MAX_CHARS_FOR_LOG = chars


def set_log_to_file(enabled: bool, file_path: Optional[str] = None):
    global _LOG_TO_FILE, _LOG_FILE_PATH
    _LOG_TO_FILE = enabled
    if file_path:
        _LOG_FILE_PATH = file_path
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)


_HOME_PATH = os.path.expanduser("~")


def truncate_for_log(text: str, max_chars: Optional[int] = None) -> str:
    max_chars = max_chars or _PROMPT_MAX_CHARS_FOR_LOG
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated, total {len(text)} chars]"


def redact_path(path: str) -> str:
    if not _SENSITIVE_DATA_REDACTED:
        return path
    try:
        p = str(path)
        if _HOME_PATH and p.startswith(_HOME_PATH):
            p = "~" + p[len(_HOME_PATH):]
        return os.path.join(os.path.dirname(p), os.path.basename(p))
    except Exception:
        return os.path.basename(str(path))


def _redact_params(params: Dict[str, Any]) -> Dict[str, Any]:
    if not _SENSITIVE_DATA_REDACTED:
        return params
    result = {}
    for key, value in params.items():
        key_lower = key.lower()
        if "prompt" in key_lower or "negative" in key_lower:
            if isinstance(value, str):
                result[key] = truncate_for_log(value)
            elif isinstance(value, list):
                result[key] = [truncate_for_log(str(v)) for v in value[:5]]
                if len(value) > 5:
                    result[key].append(f"... and {len(value) - 5} more")
            else:
                result[key] = value
        elif "path" in key_lower or "file" in key_lower or "dir" in key_lower:
            if isinstance(value, str):
                result[key] = redact_path(value)
            elif isinstance(value, list):
                result[key] = [redact_path(str(v)) for v in value]
            else:
                result[key] = value
        else:
            result[key] = value
    return result


_logger_lock = threading.Lock()


def _get_logger():
    logger = logging.getLogger("fooocus_diagnostics")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        if _LOG_TO_FILE and _LOG_FILE_PATH:
            try:
                file_handler = logging.FileHandler(_LOG_FILE_PATH, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter('%(message)s'))
                logger.addHandler(file_handler)
            except Exception:
                pass
    return logger


def _write_log_entry(entry: Dict[str, Any]):
    if not _DIAGNOSTICS_ENABLED:
        return
    try:
        with _logger_lock:
            logger = _get_logger()
            if _OUTPUT_FORMAT == "json":
                logger.info(json.dumps(entry, ensure_ascii=False, default=str))
            else:
                ts = entry.get('timestamp', '')
                lvl = entry.get('level', 'info').upper()
                tid = entry.get('trace_id', 'N/A')
                stg = entry.get('stage', 'N/A')
                msg = entry.get('human_message', '')
                logger.info(f"[{ts}] [{lvl}] [trace_id={tid}] [{stg}] {msg}")
    except Exception as e:
        print(f"[Diagnostic Logging Failed] {str(e)}", file=sys.stderr)


_task_registry_lock = threading.Lock()
_task_registry: Dict[str, 'DiagnosticJob'] = {}


def _register_job(job: 'DiagnosticJob'):
    with _task_registry_lock:
        _task_registry[job.trace_id] = job
        if len(_task_registry) > _MAX_RETAINED_TASKS:
            oldest_ids = sorted(_task_registry.keys(),
                                key=lambda tid: _task_registry[tid].created_at)
            for old_id in oldest_ids[:len(_task_registry) - _MAX_RETAINED_TASKS]:
                del _task_registry[old_id]


def get_job(trace_id: str) -> Optional['DiagnosticJob']:
    with _task_registry_lock:
        return _task_registry.get(trace_id)


_CATEGORY_MESSAGES = {
    "model_not_found": "找不到请求的模型文件。请检查模型路径或重新下载。",
    "model_corrupted": "模型文件可能已损坏。请验证文件完整性或重新下载。",
    "model_load_failed": "模型加载失败。请检查显存是否充足，或尝试使用较小的模型。",
    "lora_load_failed": "LoRA 模型加载失败。请检查 LoRA 文件是否完整。",
    "download_failed": "模型下载失败。请检查网络连接后重试。",
    "resource_scan_failed": "资源扫描失败。请检查模型目录的读取权限。",
    "metadata_parse_failed": "图像元数据解析失败。该图像可能不包含完整的参数信息。",
    "controlnet_preprocess_failed": "ControlNet 预处理失败。请检查输入图像格式。",
    "ip_adapter_preprocess_failed": "IP-Adapter 预处理失败。请检查输入图像格式。",
    "inpaint_preprocess_failed": "图像修复预处理失败。请检查遮罩和输入图像。",
    "gpu_out_of_memory": "显存不足。请尝试降低分辨率、减少图像数量或关闭其他占用显存的程序。",
    "image_save_failed": "图像保存失败。请检查输出目录写入权限和磁盘空间。",
    "diffusion_failed": "图像生成过程中发生错误。请尝试调整参数或更换模型。",
    "vae_encode_failed": "VAE 编码失败。请检查输入图像是否过大。",
    "clip_encode_failed": "CLIP 编码失败。请检查 prompt 是否过长或包含特殊字符。",
    "upscale_failed": "图像放大失败。请检查输入图像和放大模型。",
    "nsfw_check_failed": "内容检查失败。请检查 NSFW 检测模型。",
    "enhance_failed": "图像增强失败。请检查增强参数和输入图像。",
    "user_interrupted": "用户已中断生成。",
    "invalid_input": "输入参数无效。请检查所有输入值是否正确。",
    "unknown_error": "发生未知错误。请查看详细诊断信息了解更多。",
}


class DiagnosticScope:
    def __init__(self, job: 'DiagnosticJob', stage: DiagnosticStage,
                 category: Optional[DiagnosticErrorCategory] = None):
        self._job = job
        self._stage = stage
        self._category = category
        self._entered = False

    def __enter__(self):
        self._entered = True
        self._job._begin_stage(self._stage)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._entered:
            return False
        if exc_val is not None:
            cat = self._category or DiagnosticErrorCategory.UNKNOWN_ERROR
            if isinstance(exc_val, DiagnosticJobError):
                cat = exc_val.category
            self._job._finish_stage(self._stage, "failed")
            if not isinstance(exc_val, DiagnosticJobError):
                self._job.record_error(
                    stage=self._stage,
                    human_message=str(exc_val),
                    category=cat,
                    exception=exc_val,
                )
        else:
            self._job._finish_stage(self._stage, "completed")
        return False


class DiagnosticJob:
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id: str = trace_id or str(uuid.uuid4())
        self.created_at: float = time.time()
        self.status: str = "running"

        self._stages: Dict[str, Dict[str, Any]] = {}
        self._current_stage: Optional[str] = None
        self._current_stage_start: Optional[float] = None

        self._events: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []
        self._warnings: List[Dict[str, Any]] = []

        self.params: Dict[str, Any] = {}
        self.output_files: List[str] = []
        self.models_loaded: List[str] = []

        _register_job(self)

    # ---- scope context manager ----

    def scope(self, stage: DiagnosticStage,
              category: Optional[DiagnosticErrorCategory] = None) -> DiagnosticScope:
        return DiagnosticScope(self, stage, category)

    # ---- low-level stage bookkeeping ----

    def _begin_stage(self, stage: DiagnosticStage):
        stage_name = stage.value
        if self._current_stage and self._current_stage_start is not None:
            elapsed = time.time() - self._current_stage_start
            self._stages[self._current_stage] = {
                "duration_sec": elapsed,
                "status": "completed",
            }
        self._current_stage = stage_name
        self._current_stage_start = time.time()
        self._stages[stage_name] = {
            "status": "in_progress",
            "started_at": self._current_stage_start,
        }

    def _finish_stage(self, stage: DiagnosticStage, status: str = "completed"):
        stage_name = stage.value
        if self._current_stage_start is not None:
            elapsed = time.time() - self._current_stage_start
        else:
            elapsed = 0.0
        self._stages[stage_name] = {
            "duration_sec": elapsed,
            "status": status,
        }
        if self._current_stage == stage_name:
            self._current_stage = None
            self._current_stage_start = None

    # ---- record helpers ----

    def record_event(self, level: LogLevel, stage: DiagnosticStage,
                     human_message: str,
                     extra_data: Optional[Dict[str, Any]] = None):
        ts = datetime.now().isoformat()
        entry: Dict[str, Any] = {
            "timestamp": ts,
            "level": level.value,
            "trace_id": self.trace_id,
            "stage": stage.value,
            "human_message": human_message,
        }
        if self._current_stage:
            entry["current_stage"] = self._current_stage
            entry["task_status"] = self.status
        if extra_data:
            entry["data"] = _redact_params(extra_data) if _SENSITIVE_DATA_REDACTED else extra_data

        _write_log_entry(entry)
        self._events.append(entry)

        if level in (LogLevel.ERROR, LogLevel.CRITICAL):
            pass
        elif level == LogLevel.WARNING:
            self._warnings.append({
                "stage": stage.value,
                "human_message": human_message,
                "timestamp": ts,
                "data": _redact_params(extra_data) if extra_data and _SENSITIVE_DATA_REDACTED else extra_data,
            })

    def record_error(self, stage: DiagnosticStage,
                     human_message: str,
                     category: DiagnosticErrorCategory = DiagnosticErrorCategory.UNKNOWN_ERROR,
                     exception: Optional[BaseException] = None,
                     extra_data: Optional[Dict[str, Any]] = None):
        error_type = type(exception).__name__ if exception else None
        error_message = truncate_for_log(str(exception), 500) if exception else None
        traceback_str = None
        if exception is not None:
            try:
                traceback_str = traceback.format_exc()
            except Exception:
                traceback_str = None

        ts = datetime.now().isoformat()
        entry: Dict[str, Any] = {
            "timestamp": ts,
            "level": LogLevel.ERROR.value,
            "trace_id": self.trace_id,
            "stage": stage.value,
            "human_message": human_message,
            "category": category.value,
            "error_type": error_type,
            "error_message": error_message,
        }
        if traceback_str and not _SENSITIVE_DATA_REDACTED:
            entry["traceback"] = truncate_for_log(traceback_str, 2000)
        if extra_data:
            entry["data"] = _redact_params(extra_data) if _SENSITIVE_DATA_REDACTED else extra_data

        _write_log_entry(entry)

        self._errors.append({
            "stage": stage.value,
            "human_message": human_message,
            "timestamp": ts,
            "category": category.value,
            "error_type": error_type,
            "data": _redact_params(extra_data) if extra_data and _SENSITIVE_DATA_REDACTED else extra_data,
        })
        self.status = "failed"

    # ---- param / output helpers ----

    def set_param(self, key: str, value: Any):
        self.params[key] = value

    def add_output_file(self, filepath: str):
        if _SENSITIVE_DATA_REDACTED:
            self.output_files.append(os.path.basename(filepath))
        else:
            self.output_files.append(filepath)

    def add_model_loaded(self, model_name: str):
        self.models_loaded.append(model_name)

    # ---- lifecycle ----

    def mark_success(self):
        self.status = "success"
        if self._current_stage and self._current_stage_start is not None:
            self._stages[self._current_stage] = {
                "duration_sec": time.time() - self._current_stage_start,
                "status": "completed",
            }
            self._current_stage = None
            self._current_stage_start = None

    def mark_failed(self):
        self.status = "failed"
        if self._current_stage and self._current_stage_start is not None:
            self._stages[self._current_stage] = {
                "duration_sec": time.time() - self._current_stage_start,
                "status": "failed",
            }
            self._current_stage = None
            self._current_stage_start = None

    # ---- output methods ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "status": self.status,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "total_duration_sec": round(time.time() - self.created_at, 3),
            "current_stage": self._current_stage,
            "stages": self._stages,
            "params": _redact_params(self.params),
            "output_files": self.output_files,
            "models_loaded": self.models_loaded,
            "errors_count": len(self._errors),
            "warnings_count": len(self._warnings),
            "errors": self._errors,
            "warnings": self._warnings,
        }

    def summary(self) -> str:
        lines = [
            "=== Fooocus Diagnostic Summary ===",
            f"Trace ID: {self.trace_id}",
            f"Status: {self.status.upper()}",
            f"Created: {datetime.fromtimestamp(self.created_at).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {round(time.time() - self.created_at, 2)}s",
        ]
        if self._current_stage:
            lines.append(f"Failed at stage: {self._current_stage}")
        if self._errors:
            lines.append("")
            lines.append(f"Errors ({len(self._errors)}):")
            for i, err in enumerate(self._errors[:5], 1):
                lines.append(f"  [{i}] {err.get('category', 'unknown')}: {err.get('human_message', 'No description')}")
                if err.get("error_type"):
                    lines.append(f"       Type: {err['error_type']}")
        if self._warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self._warnings)}):")
            for i, w in enumerate(self._warnings[:3], 1):
                lines.append(f"  [{i}] {w.get('human_message', 'No description')}")
        if self.params:
            lines.append("")
            lines.append("Key Parameters:")
            safe_params = _redact_params(self.params)
            for k, v in list(safe_params.items())[:10]:
                lines.append(f"  {k}: {truncate_for_log(str(v))}")
        if self.models_loaded:
            lines.append("")
            lines.append(f"Models loaded ({len(self.models_loaded)}):")
            for m in self.models_loaded:
                lines.append(f"  - {os.path.basename(m) if _SENSITIVE_DATA_REDACTED else m}")
        if self._stages:
            lines.append("")
            lines.append("Stage Timings:")
            for stage, info in self._stages.items():
                dur = info.get("duration_sec", "?")
                st = info.get("status", "?")
                dur_str = f"{dur:.2f}s" if isinstance(dur, float) else str(dur)
                lines.append(f"  {stage}: {dur_str} [{st}]")
        lines.append("==================================")
        return "\n".join(lines)

    def to_public_error(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "title": "生成遇到问题",
            "message": "处理您的请求时发生错误。",
            "trace_id": self.trace_id,
            "show_diagnostics_button": True,
            "diagnostics_summary": self.summary(),
        }
        if self._errors:
            latest = self._errors[-1]
            cat = latest.get("category", "unknown_error")
            result["message"] = _CATEGORY_MESSAGES.get(cat, result["message"])
            result["category"] = cat
        if self._current_stage:
            result["title"] = f"生成在「{self._current_stage}」阶段失败"
        return result


class DiagnosticJobError(Exception):
    def __init__(
        self,
        human_message: str,
        category: DiagnosticErrorCategory = DiagnosticErrorCategory.UNKNOWN_ERROR,
        stage: Optional[DiagnosticStage] = None,
        job: Optional[DiagnosticJob] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        self.human_message = human_message
        self.category = category
        self.stage = stage
        self.job = job
        self.extra_data = extra_data
        self.trace_id = job.trace_id if job else None
        self.cause = cause

        super().__init__(human_message)
        if cause:
            self.__cause__ = cause

        if job and stage:
            job.record_error(
                stage=stage,
                human_message=human_message,
                category=category,
                exception=cause or self,
                extra_data=extra_data,
            )

    def public_error_message(self) -> str:
        return f"{self.human_message} (Trace ID: {self.trace_id})"
