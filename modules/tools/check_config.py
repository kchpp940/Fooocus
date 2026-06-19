from __future__ import annotations

import json
import numbers
import os
from typing import Any, Dict, List, Optional

from modules.tools.common import ToolResult

from modules.services.config_inspector import (
    load_and_validate_config,
    list_all_presets,
    load_preset_content,
)


def add_check_config_args(parser) -> None:
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config.txt (defaults to project root /config.txt)",
    )
    parser.add_argument(
        "--fix-paths",
        action="store_true",
        help="Auto-create missing path_* directories",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Validate a specific preset by name",
    )


def _auto_config_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "config.txt")


def _validate_config_file(config_path: str, result: ToolResult) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(config_path):
        result.add_warning(
            name="config_file",
            message=f"Config file does not exist: {config_path}",
            suggestion="Create it from config_modification_tutorial.txt",
            detail={"path": config_path},
        )
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            result.add_error(
                name="config_file",
                message="Config file root is not a valid JSON object",
                suggestion="Top-level value should be { ... }",
                detail={"path": config_path},
            )
            return {"__error__": "not_an_object"}
        result.add_info(
            name="config_file",
            message=f"Loaded {len(cfg)} keys from {os.path.basename(config_path)}",
            detail={"keys_count": len(cfg), "path": config_path},
        )
        return cfg
    except json.JSONDecodeError as e:
        result.add_error(
            name="config_file",
            message=f"Config file is not valid JSON: {e.msg}",
            suggestion="Fix JSON syntax; check for trailing commas and backslashes.",
            detail={"path": config_path, "line": getattr(e, "lineno", None)},
        )
        return {"__error__": f"json_decode: {e.msg}"}


def _validate_path_value(
    key: str,
    value: Any,
    result: ToolResult,
    fix_paths: bool,
) -> None:
    folders: List[str] = []
    if isinstance(value, list):
        folders = [v for v in value if isinstance(v, str)]
    elif isinstance(value, str):
        folders = [value]

    for folder in folders:
        abs_folder = os.path.abspath(folder)
        if os.path.isdir(abs_folder):
            result.add_info(
                name=f"config_{key}",
                message=f"'{key}' directory exists: {abs_folder}",
                detail={"path": abs_folder},
            )
        else:
            if fix_paths:
                try:
                    os.makedirs(abs_folder, exist_ok=True)
                    result.add_info(
                        name=f"config_{key}",
                        message=f"Created missing '{key}' directory: {abs_folder}",
                        detail={"path": abs_folder, "created": True},
                    )
                except OSError as e:
                    result.add_error(
                        name=f"config_{key}",
                        message=f"Could not create '{key}' directory {abs_folder}: {e}",
                        suggestion="Fix permissions or choose a different path.",
                        detail={"path": abs_folder},
                    )
            else:
                result.add_warning(
                    name=f"config_{key}",
                    message=f"'{key}' directory does not exist: {abs_folder}",
                    suggestion="Create it manually or re-run with --fix-paths",
                    detail={"path": abs_folder},
                )


def _validate_flags_values(cfg: Dict[str, Any], result: ToolResult) -> None:
    try:
        import modules.flags as flags
        if "default_performance" in cfg:
            if cfg["default_performance"] not in flags.Performance.values():
                result.add_error(
                    name="config_default_performance",
                    message=f"default_performance='{cfg['default_performance']}' is not valid",
                    suggestion=f"Valid values: {', '.join(flags.Performance.values())}",
                )
        if "default_output_format" in cfg:
            if cfg["default_output_format"] not in flags.OutputFormat.list():
                result.add_error(
                    name="config_default_output_format",
                    message=f"default_output_format='{cfg['default_output_format']}' invalid",
                    suggestion=f"Valid: {', '.join(flags.OutputFormat.list())}",
                )
        if "default_sampler" in cfg and cfg["default_sampler"] not in flags.sampler_list:
            result.add_error(
                name="config_default_sampler",
                message=f"default_sampler='{cfg['default_sampler']}' not in sampler_list",
            )
        if "default_scheduler" in cfg and cfg["default_scheduler"] not in flags.scheduler_list:
            result.add_error(
                name="config_default_scheduler",
                message=f"default_scheduler='{cfg['default_scheduler']}' not in scheduler_list",
            )
    except Exception as e:
        result.add_warning(
            name="flags_import",
            message=f"Could not validate flag-based values: {e}",
        )


def _validate_numeric_range(
    key: str, value: Any, result: ToolResult,
    min_val: Optional[float] = None, max_val: Optional[float] = None,
) -> None:
    if not isinstance(value, numbers.Number):
        result.add_error(
            name=f"config_{key}",
            message=f"'{key}' = {value!r} is not a number",
            suggestion="Set to a numeric value.",
        )
        return
    if min_val is not None and value < min_val:
        result.add_error(
            name=f"config_{key}",
            message=f"'{key}' = {value} is below minimum {min_val}",
        )
    if max_val is not None and value > max_val:
        result.add_error(
            name=f"config_{key}",
            message=f"'{key}' = {value} is above maximum {max_val}",
        )


def _emit_schema_results(schema_info: Dict[str, Any], result: ToolResult, cfg_from_file: Optional[Dict[str, Any]]) -> None:
    if not schema_info.get("available"):
        result.add_warning(
            name="schema_unavailable",
            message="config_schema module could not be loaded for schema validation",
        )
        return
    issues = schema_info.get("issues", []) or []
    if not issues:
        count = len([k for k in (cfg_from_file or {}) if k != "__error__"])
        result.add_info(
            name="schema_validation",
            message=f"All {count} configured keys pass schema validation",
        )
    else:
        for issue in issues:
            severity = issue.get("severity", "warning")
            ikey = issue.get("key", "?")
            msg = issue.get("message", "")
            if severity == "error":
                result.add_error(
                    name=f"schema_{ikey}",
                    message=f"[VALIDATION_ERROR] {ikey}: {msg}",
                    suggestion="Check the value type/format against the documentation.",
                    detail=issue.get("detail"),
                )
            else:
                result.add_warning(
                    name=f"schema_{ikey}",
                    message=f"[{severity.upper()}] {ikey}: {msg}",
                    suggestion="Check spelling or remove if obsolete.",
                )
    result.data["schema_issues_count"] = len(issues)


def _validate_preset(preset_name: str, result: ToolResult) -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        presets = list_all_presets(root)
        if preset_name in presets:
            result.add_info(
                name=f"preset_{preset_name}",
                message=f"Preset '{preset_name}' exists",
            )
            content = load_preset_content(preset_name, root)
            if not isinstance(content, dict):
                result.add_warning(
                    name=f"preset_{preset_name}",
                    message=f"Preset '{preset_name}' content is empty or not a dict",
                )
        else:
            result.add_error(
                name=f"preset_{preset_name}",
                message=f"Preset '{preset_name}' not found",
                suggestion=f"Available: {', '.join(presets[:15])}{' ...' if len(presets) > 15 else ''}",
                detail={"available_count": len(presets)},
            )
    except Exception as e:
        result.add_warning(
            name=f"preset_{preset_name}",
            message=f"Preset validation error: {e}",
        )


def check_configuration(args: Any) -> ToolResult:
    result = ToolResult(tool_name="check-config")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = getattr(args, "config_path", None) or _auto_config_path()
    fix_paths = bool(getattr(args, "fix_paths", False))

    # 先用 service 层做完整的无副作用解析 / schema 校验
    inspection = load_and_validate_config(
        config_path=config_path,
        project_root=project_root,
        fix_missing_dirs=fix_paths,
    )

    # 同时保留原始 JSON 解析检查，用于向用户展示更友好的错误信息
    cfg = _validate_config_file(config_path, result)

    # 如果 service 层创建了目录，记录一下
    created = inspection.get("created_dirs") or []
    for path in created:
        result.add_info(
            name="config_created_dir",
            message=f"Created missing directory: {path}",
            detail={"path": path},
        )

    # 把 service 层对 path_* 的有效解析灌入结果
    paths_info = inspection.get("paths", {}).get("by_key", {}) or {}
    for key, info in paths_info.items():
        for folder in info.get("effective", []):
            if os.path.isdir(folder):
                result.add_info(
                    name=f"config_{key}",
                    message=f"'{key}' directory exists: {folder}",
                    detail={"path": folder, "source": info.get("source", "default")},
                )
            elif not fix_paths:
                result.add_warning(
                    name=f"config_{key}",
                    message=f"'{key}' directory does not exist: {folder}",
                    suggestion="Create it manually or re-run with --fix-paths",
                    detail={"path": folder, "source": info.get("source", "default")},
                )

    # 特殊：temp_path 和 path_user_data
    for special_key in ("temp_path", "path_user_data"):
        sp = inspection.get("paths", {}).get(special_key)
        if sp is None:
            continue
        folder = sp.get("effective")
        if folder is None:
            continue
        if os.path.isdir(folder):
            result.add_info(
                name=f"config_{special_key}",
                message=f"'{special_key}' directory exists: {folder}",
                detail={"path": folder},
            )
        elif not fix_paths:
            result.add_warning(
                name=f"config_{special_key}",
                message=f"'{special_key}' directory does not exist: {folder}",
                suggestion="Create it manually or re-run with --fix-paths",
                detail={"path": folder},
            )

    # flag / 数值范围校验（仍然走友好的本地逻辑，因为这些是 CLI 输出层面的友好化）
    if cfg is not None and "__error__" not in cfg:
        _validate_flags_values(cfg, result)
        for k, lo, hi in [
            ("default_cfg_scale", None, None),
            ("default_sample_sharpness", None, None),
            ("default_cfg_tsnr", None, None),
            ("default_refiner_switch", 0, 1),
        ]:
            if k in cfg:
                _validate_numeric_range(k, cfg[k], result, min_val=lo, max_val=hi)

    # schema 校验结果从 service 层拿来直接 emit
    _emit_schema_results(
        inspection.get("schema", {}) or {},
        result,
        cfg if cfg and "__error__" not in cfg else None,
    )

    preset = getattr(args, "preset", None)
    if preset:
        _validate_preset(preset, result)

    result.data["project_root"] = project_root
    result.data["config_path"] = config_path
    result.data.update({
        k: v for k, v in (inspection.get("config") or {}).items()
        if k != "data"
    })

    result.compute_summary()
    return result
