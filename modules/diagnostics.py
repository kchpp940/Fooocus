import os
import sys
import json
import uuid
import time
import traceback
import threading
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List, Callable
from datetime import datetime
from enum import Enum
from collections import defaultdict
import re


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
    PROMPT_PROCESS = "prompt_process"
    CONTROLNET_PREPROCESS = "controlnet_preprocess"
    IP_ADAPTER_PREPROCESS = "ip_adapter_preprocess"
    INPAINT_PREPROCESS = "inpaint_preprocess"
    VAE_ENCODE = "vae_encode"
    DIFFUSION = "diffusion"
    METADATA_PARSE = "metadata_parse"
    IMAGE_SAVE = "image_save"
    WEBCALLBACK = "webcallback"
    ENHANCE = "enhance"
    CLEANUP = "cleanup"
    UNKNOWN = "unknown"


class DiagnosticErrorCategory(Enum):
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_CORRUPTED = "model_corrupted"
    MODEL_LOAD_FAILED = "model_load_failed"
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


def generate_trace_id() -> str:
    return str(uuid.uuid4())


class DiagnosticContext:
    _thread_local = threading.local()

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or generate_trace_id()
        self.created_at = time.time()
        self.stages_timings: Dict[str, Dict[str, float]] = {}
        self.current_stage: Optional[str] = None
        self.current_stage_start: Optional[float] = None
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.params_summary: Dict[str, Any] = {}
        self.output_files: List[str] = []
        self.models_loaded: List[str] = []
        self.status: str = "running"

    def start_stage(self, stage: DiagnosticStage):
        stage_name = stage.value
        if self.current_stage and self.current_stage_start:
            elapsed = time.time() - self.current_stage_start
            self.stages_timings[self.current_stage] = {
                "duration_sec": elapsed,
                "status": "completed"
            }
        self.current_stage = stage_name
        self.current_stage_start = time.time()
        self.stages_timings[stage_name] = {
            "status": "in_progress",
            "started_at": self.current_stage_start
        }

    def end_stage(self, stage: Optional[DiagnosticStage] = None, status: str = "completed"):
        if stage:
            stage_name = stage.value
        elif self.current_stage:
            stage_name = self.current_stage
        else:
            return
        if stage_name in self.stages_timings and self.current_stage_start:
            elapsed = time.time() - self.current_stage_start
            self.stages_timings[stage_name] = {
                "duration_sec": elapsed,
                "status": status
            }
        self.current_stage = None
        self.current_stage_start = None

    def mark_failed(self):
        self.status = "failed"
        if self.current_stage and self.current_stage_start:
            self.stages_timings[self.current_stage] = {
                "duration_sec": time.time() - self.current_stage_start,
                "status": "failed"
            }

    def mark_success(self):
        self.status = "success"
        if self.current_stage and self.current_stage_start:
            self.stages_timings[self.current_stage] = {
                "duration_sec": time.time() - self.current_stage_start,
                "status": "completed"
            }

    def add_param_summary(self, key: str, value: Any):
        self.params_summary[key] = value

    def add_output_file(self, filepath: str):
        if _SENSITIVE_DATA_REDACTED:
            self.output_files.append(os.path.basename(filepath))
        else:
            self.output_files.append(filepath)

    def add_model_loaded(self, model_name: str):
        self.models_loaded.append(model_name)

    def to_dict(self) -> Dict[str, Any]:
        total_duration = time.time() - self.created_at
        return {
            "trace_id": self.trace_id,
            "status": self.status,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "total_duration_sec": round(total_duration, 3),
            "current_stage": self.current_stage,
            "stages_timings": self.stages_timings,
            "params_summary": redact_sensitive_params(self.params_summary),
            "output_files": self.output_files,
            "models_loaded": self.models_loaded,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings
        }

    def to_summary_string(self) -> str:
        lines = [
            f"=== Fooocus Diagnostic Summary ===",
            f"Trace ID: {self.trace_id}",
            f"Status: {self.status.upper()}",
            f"Created: {datetime.fromtimestamp(self.created_at).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {round(time.time() - self.created_at, 2)}s",
        ]
        if self.current_stage:
            lines.append(f"Failed at stage: {self.current_stage}")
        if self.errors:
            lines.append("")
            lines.append(f"Errors ({len(self.errors)}):")
            for i, err in enumerate(self.errors[:5], 1):
                lines.append(f"  [{i}] {err.get('category', 'unknown')}: {err.get('human_message', 'No description')}")
                if err.get("error_type"):
                    lines.append(f"       Type: {err['error_type']}")
        if self.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self.warnings)}):")
            for i, w in enumerate(self.warnings[:3], 1):
                lines.append(f"  [{i}] {w.get('human_message', 'No description')}")
        if self.params_summary:
            lines.append("")
            lines.append("Key Parameters:")
            safe_params = redact_sensitive_params(self.params_summary)
            for k, v in list(safe_params.items())[:10]:
                lines.append(f"  {k}: {truncate_for_log(str(v))}")
        if self.models_loaded:
            lines.append("")
            lines.append(f"Models loaded ({len(self.models_loaded)}):")
            for m in self.models_loaded:
                lines.append(f"  - {os.path.basename(m) if _SENSITIVE_DATA_REDACTED else m}")
        if self.stages_timings:
            lines.append("")
            lines.append("Stage Timings:")
            for stage, info in self.stages_timings.items():
                dur = info.get("duration_sec", info.get("elapsed", "?"))
                st = info.get("status", "?")
                dur_str = f"{dur:.2f}s" if isinstance(dur, float) else str(dur)
                lines.append(f"  {stage}: {dur_str} [{st}]")
        lines.append("==================================")
        return "\n".join(lines)

    @classmethod
    def get_current(cls) -> Optional['DiagnosticContext']:
        return getattr(cls._thread_local, 'current_context', None)

    @classmethod
    def set_current(cls, ctx: Optional['DiagnosticContext']):
        cls._thread_local.current_context = ctx

    def __enter__(self):
        self._previous = DiagnosticContext.get_current()
        DiagnosticContext.set_current(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.mark_failed()
        else:
            self.mark_success()
        if self._previous:
            DiagnosticContext.set_current(self._previous)
        else:
            DiagnosticContext.set_current(None)
        return False


def get_current_context() -> Optional[DiagnosticContext]:
    return DiagnosticContext.get_current()


def truncate_for_log(text: str, max_chars: Optional[int] = None) -> str:
    max_chars = max_chars or _PROMPT_MAX_CHARS_FOR_LOG
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated, total {len(text)} chars]"


_HOME_PATH = os.path.expanduser("~")


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


def redact_sensitive_params(params: Dict[str, Any]) -> Dict[str, Any]:
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


_task_registry_lock = threading.Lock()
_task_registry: Dict[str, DiagnosticContext] = {}


def register_task(ctx: DiagnosticContext):
    with _task_registry_lock:
        _task_registry[ctx.trace_id] = ctx
        if len(_task_registry) > _MAX_RETAINED_TASKS:
            oldest_ids = sorted(_task_registry.keys(),
                                key=lambda tid: _task_registry[tid].created_at)
            for old_id in oldest_ids[:len(_task_registry) - _MAX_RETAINED_TASKS]:
                del _task_registry[old_id]


def unregister_task(trace_id: str):
    with _task_registry_lock:
        _task_registry.pop(trace_id, None)


def get_task(trace_id: str) -> Optional[DiagnosticContext]:
    with _task_registry_lock:
        return _task_registry.get(trace_id)


def get_all_tasks(status_filter: Optional[str] = None, limit: int = 50) -> List[DiagnosticContext]:
    with _task_registry_lock:
        tasks = list(_task_registry.values())
    tasks.sort(key=lambda t: t.created_at, reverse=True)
    if status_filter:
        tasks = [t for t in tasks if t.status == status_filter]
    return tasks[:limit]


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
                msg = f"[{entry.get('timestamp')}] [{entry.get('level', 'info').upper()}] [trace_id={entry.get('trace_id', 'N/A')}] [{entry.get('stage', 'N/A')}] {entry.get('human_message', '')}"
                logger.info(msg)
    except Exception as e:
        print(f"[Diagnostic Logging Failed] {str(e)}", file=sys.stderr)


def log_event(
    level: LogLevel,
    stage: DiagnosticStage,
    human_message: str,
    ctx: Optional[DiagnosticContext] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    **kwargs
):
    ctx = ctx or get_current_context()
    if ctx is None and level in (LogLevel.ERROR, LogLevel.CRITICAL, LogLevel.WARNING):
        ctx = DiagnosticContext()
        register_task(ctx)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level.value,
        "trace_id": ctx.trace_id if ctx else None,
        "stage": stage.value,
        "human_message": human_message,
    }

    if ctx:
        entry["current_stage"] = ctx.current_stage
        entry["task_status"] = ctx.status

    if extra_data:
        if _SENSITIVE_DATA_REDACTED:
            entry["data"] = redact_sensitive_params(extra_data)
        else:
            entry["data"] = extra_data

    for k, v in kwargs.items():
        if k not in entry:
            entry[k] = v

    _write_log_entry(entry)

    if ctx and level in (LogLevel.ERROR, LogLevel.CRITICAL):
        ctx.errors.append({
            "stage": stage.value,
            "human_message": human_message,
            "timestamp": entry["timestamp"],
            "category": kwargs.get("category"),
            "error_type": kwargs.get("error_type"),
            "data": redact_sensitive_params(extra_data) if extra_data and _SENSITIVE_DATA_REDACTED else extra_data
        })
        ctx.mark_failed()

    if ctx and level == LogLevel.WARNING:
        ctx.warnings.append({
            "stage": stage.value,
            "human_message": human_message,
            "timestamp": entry["timestamp"],
            "data": redact_sensitive_params(extra_data) if extra_data and _SENSITIVE_DATA_REDACTED else extra_data
        })

    return ctx


def log_error(
    stage: DiagnosticStage,
    human_message: str,
    exception: Optional[BaseException] = None,
    category: DiagnosticErrorCategory = DiagnosticErrorCategory.UNKNOWN_ERROR,
    ctx: Optional[DiagnosticContext] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    should_include_traceback: bool = True,
) -> DiagnosticContext:
    error_type = None
    error_message = None
    traceback_str = None

    if exception is not None:
        error_type = type(exception).__name__
        error_message = str(exception)
        if should_include_traceback:
            try:
                traceback_str = traceback.format_exc()
            except Exception:
                traceback_str = None

    ctx = log_event(
        level=LogLevel.ERROR,
        stage=stage,
        human_message=human_message,
        ctx=ctx,
        extra_data=extra_data,
        category=category.value,
        error_type=error_type,
        error_message=truncate_for_log(error_message, 500) if error_message else None,
        traceback=truncate_for_log(traceback_str, 2000) if traceback_str and not _SENSITIVE_DATA_REDACTED else None,
    )
    return ctx


def log_warning(
    stage: DiagnosticStage,
    human_message: str,
    ctx: Optional[DiagnosticContext] = None,
    extra_data: Optional[Dict[str, Any]] = None,
):
    return log_event(
        level=LogLevel.WARNING,
        stage=stage,
        human_message=human_message,
        ctx=ctx,
        extra_data=extra_data,
    )


def log_info(
    stage: DiagnosticStage,
    human_message: str,
    ctx: Optional[DiagnosticContext] = None,
    extra_data: Optional[Dict[str, Any]] = None,
):
    return log_event(
        level=LogLevel.INFO,
        stage=stage,
        human_message=human_message,
        ctx=ctx,
        extra_data=extra_data,
    )


def log_debug(
    stage: DiagnosticStage,
    human_message: str,
    ctx: Optional[DiagnosticContext] = None,
    extra_data: Optional[Dict[str, Any]] = None,
):
    return log_event(
        level=LogLevel.DEBUG,
        stage=stage,
        human_message=human_message,
        ctx=ctx,
        extra_data=extra_data,
    )


class DiagnosticsError(Exception):
    def __init__(
        self,
        human_message: str,
        category: DiagnosticErrorCategory = DiagnosticErrorCategory.UNKNOWN_ERROR,
        stage: Optional[DiagnosticStage] = None,
        ctx: Optional[DiagnosticContext] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ):
        self.human_message = human_message
        self.category = category
        self.stage = stage
        self.ctx = ctx
        self.extra_data = extra_data
        self.trace_id = trace_id or (ctx.trace_id if ctx else None)
        self.cause = cause

        super().__init__(human_message)
        if cause:
            self.__cause__ = cause

        if stage:
            log_error(
                stage=stage,
                human_message=human_message,
                exception=cause or self,
                category=category,
                ctx=ctx,
                extra_data=extra_data,
            )

    def get_diagnostic_summary(self) -> str:
        if self.ctx:
            return self.ctx.to_summary_string()
        lines = [
            f"=== Fooocus Error Summary ===",
            f"Trace ID: {self.trace_id or 'N/A'}",
            f"Category: {self.category.value}",
            f"Message: {self.human_message}",
            f"Error Type: {type(self.cause).__name__ if self.cause else type(self).__name__}",
        ]
        if self.cause:
            lines.append(f"Details: {str(self.cause)}")
        lines.append("==============================")
        return "\n".join(lines)

    def public_error_message(self) -> str:
        return f"{self.human_message} (Trace ID: {self.trace_id})"


def safe_call_with_diagnostics(
    func: Callable,
    stage: DiagnosticStage,
    error_category: DiagnosticErrorCategory,
    error_human_message: str,
    ctx: Optional[DiagnosticContext] = None,
    extra_data_on_error: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs
):
    try:
        if ctx:
            ctx.start_stage(stage)
        result = func(*args, **kwargs)
        if ctx:
            ctx.end_stage(stage, "completed")
        return result
    except DiagnosticsError:
        if ctx:
            ctx.end_stage(stage, "failed")
        raise
    except Exception as e:
        if ctx:
            ctx.end_stage(stage, "failed")
        raise DiagnosticsError(
            human_message=error_human_message,
            category=error_category,
            stage=stage,
            ctx=ctx,
            extra_data=extra_data_on_error,
            cause=e,
        )


def get_user_friendly_error(trace_id: Optional[str] = None) -> Dict[str, Any]:
    result = {
        "title": "生成遇到问题",
        "message": "处理您的请求时发生错误。",
        "trace_id": trace_id,
        "show_diagnostics_button": False,
        "diagnostics_summary": None,
    }
    if trace_id:
        ctx = get_task(trace_id)
        if ctx:
            result["show_diagnostics_button"] = True
            result["diagnostics_summary"] = ctx.to_summary_string()
            if ctx.errors:
                latest = ctx.errors[-1]
                cat = latest.get("category", "unknown_error")
                messages = {
                    "model_not_found": "找不到请求的模型文件。请检查模型路径或重新下载。",
                    "model_corrupted": "模型文件可能已损坏。请验证文件完整性或重新下载。",
                    "model_load_failed": "模型加载失败。请检查显存是否充足，或尝试使用较小的模型。",
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
                    "unknown_error": "发生未知错误。请查看详细诊断信息了解更多。",
                }
                result["message"] = messages.get(cat, result["message"])
                if ctx.current_stage:
                    result["title"] = f"生成在「{ctx.current_stage}」阶段失败"
    return result
