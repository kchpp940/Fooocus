import os
import sys
import json
from typing import Any, Dict, List, Optional

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from modules.tools.common import ToolResult


def add_check_config_args(parser) -> None:
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config.txt (auto-detect otherwise)",
    )
    parser.add_argument(
        "--fix-paths",
        action="store_true",
        help="Create missing directories that should exist",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Also validate this preset name",
    )


def _auto_config_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "config.txt")


def _load_config_dict(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        return {"__error__": f"JSON decode error: {e}"}
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}


def _validate_config_file(path: str, result: ToolResult) -> Optional[Dict[str, Any]]:
    result.data["config_path"] = path
    if not os.path.exists(path):
        result.add_warning(
            name="config_file_missing",
            message=f"Config file does not exist: {path}",
            suggestion="This is normal on first launch; config will be auto-created.",
        )
        return None

    cfg = _load_config_dict(path)
    if cfg is None:
        result.add_error(
            name="config_file_unreadable",
            message=f"Config file exists but could not be read: {path}",
        )
        return None

    if "__error__" in cfg:
        result.add_error(
            name="config_file_invalid_json",
            message=f"Config file is not valid JSON: {cfg['__error__']}",
            suggestion="Fix syntax errors; common issue: trailing comma before closing }.",
            detail={"path": path},
        )
        return None

    result.add_info(
        name="config_file",
        message=f"Loaded {len(cfg)} keys from {os.path.basename(path)}",
        detail={"keys_count": len(cfg), "path": path},
    )
    return cfg


def _validate_path_value(key: str, value: Any, result: ToolResult, fix_paths: bool) -> None:
    if isinstance(value, list):
        paths_to_check = value
    else:
        paths_to_check = [value]
    for idx, p in enumerate(paths_to_check):
        if not isinstance(p, str):
            result.add_error(
                name=f"config_{key}",
                message=f"Config '{key}'[{idx}] is not a string path",
                suggestion="Use a string (or list of strings) representing a filesystem path.",
                detail={"value_type": type(p).__name__},
            )
            continue
        label = f"config_{key}" if len(paths_to_check) == 1 else f"config_{key}[{idx}]"
        if os.path.exists(p):
            if os.path.isdir(p):
                result.add_info(
                    name=label,
                    message=f"'{key}' directory exists: {p}",
                    detail={"path": p},
                )
            else:
                result.add_warning(
                    name=label,
                    message=f"'{key}' points to an existing file, not a directory: {p}",
                    detail={"path": p},
                )
        else:
            if fix_paths:
                try:
                    os.makedirs(p, exist_ok=True)
                    result.add_info(
                        name=label,
                        message=f"'{key}' created missing directory: {p}",
                        detail={"path": p, "created": True},
                    )
                except OSError as e:
                    result.add_error(
                        name=label,
                        message=f"'{key}' directory missing and cannot create '{p}': {e}",
                        detail={"path": p},
                    )
            else:
                result.add_warning(
                    name=label,
                    message=f"'{key}' directory does not exist: {p}",
                    suggestion="Pass --fix-paths to auto-create, or create the directory manually.",
                    detail={"path": p},
                )


def _validate_numeric_range(
    key: str, value: Any, result: ToolResult,
    min_val: Optional[float] = None, max_val: Optional[float] = None,
) -> None:
    if not isinstance(value, (int, float)):
        result.add_error(
            name=f"config_{key}",
            message=f"'{key}' should be numeric, got {type(value).__name__}",
            suggestion="Use an int or float number.",
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


def _load_schema_definition():
    try:
        from modules.config_schema import create_default_schema
        return create_default_schema()
    except Exception:
        return None


def _validate_with_config_schema(result: ToolResult, cfg_from_file: Optional[Dict[str, Any]]) -> None:
    schema = _load_schema_definition()
    if schema is None:
        result.add_warning(
            name="schema_unavailable",
            message="config_schema module could not be loaded for schema validation",
        )
        return

    issues: List[str] = []
    cfg = cfg_from_file or {}

    for key, item in schema.items.items():
        if item.validator is None or key not in cfg:
            continue
        val = cfg[key]
        try:
            valid = item.validator(val)
        except Exception:
            valid = False
        if not valid:
            issues.append(f"VALIDATION_ERROR::{key}::Value failed validator")

    for key in cfg:
        if not schema.has(key) and schema.resolve_alias(key) is None:
            issues.append(f"UNKNOWN_KEY::{key}::Not defined in schema")

    if not issues:
        count = len([k for k in cfg if k != "__error__"])
        result.add_info(
            name="schema_validation",
            message=f"All {count} configured keys pass schema validation",
        )
    else:
        for issue in issues:
            parts = issue.split("::", 2)
            severity, ikey, msg = parts if len(parts) == 3 else ("WARNING", "?", issue)
            if severity == "VALIDATION_ERROR":
                result.add_error(
                    name=f"schema_{ikey}",
                    message=f"[{severity}] {ikey}: {msg}",
                    suggestion="Check the value type/format against the documentation.",
                    detail={"key": ikey, "value_in_file": str(cfg.get(ikey))[:200] if cfg.get(ikey) is not None else None},
                )
            else:
                result.add_warning(
                    name=f"schema_{ikey}",
                    message=f"[{severity}] {ikey}: {msg}",
                    suggestion="Check spelling or remove if obsolete.",
                )
    result.data["schema_issues_count"] = len(issues)


def _list_presets(root: str) -> List[str]:
    presets = ["initial"]
    builtin_dir = os.path.join(root, "presets")
    if os.path.isdir(builtin_dir):
        for fname in os.listdir(builtin_dir):
            if fname.endswith(".json"):
                presets.append(fname[:-5])
    user_dir = os.path.join(
        os.environ.get("FOOOCUS_USER_DATA_DIR") or
        os.path.join(os.path.expanduser("~"), ".fooocus"),
        "user_presets",
    )
    if os.path.isdir(user_dir):
        for fname in os.listdir(user_dir):
            if fname.endswith(".json"):
                presets.append(f"[User] {fname[:-5]}")
    return presets


def _load_preset_content(preset_name: str, root: str) -> Dict[str, Any]:
    if preset_name == "initial":
        return {}
    user_prefix = "[User] "
    if preset_name.startswith(user_prefix):
        short = preset_name[len(user_prefix):]
        user_dir = os.path.join(
            os.environ.get("FOOOCUS_USER_DATA_DIR") or
            os.path.join(os.path.expanduser("~"), ".fooocus"),
            "user_presets",
        )
        path = os.path.join(user_dir, f"{short}.json")
    else:
        path = os.path.join(root, "presets", f"{preset_name}.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _validate_preset(preset_name: str, result: ToolResult) -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        presets = _list_presets(root)
        if preset_name in presets:
            result.add_info(
                name=f"preset_{preset_name}",
                message=f"Preset '{preset_name}' exists",
            )
            content = _load_preset_content(preset_name, root)
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


def check_configuration(args: Any) -> ToolResult:
    result = ToolResult(tool_name="check-config")

    config_path = getattr(args, "config_path", None) or _auto_config_path()
    cfg = _validate_config_file(config_path, result)

    fix_paths = getattr(args, "fix_paths", False)

    if cfg is not None and "__error__" not in cfg:
        path_keys_prefix = "path_"
        for key, value in cfg.items():
            if key.startswith(path_keys_prefix):
                _validate_path_value(key, value, result, fix_paths)
        _validate_flags_values(cfg, result)

        for k, lo, hi in [
            ("default_cfg_scale", None, None),
            ("default_sample_sharpness", None, None),
            ("default_cfg_tsnr", None, None),
            ("default_refiner_switch", 0, 1),
        ]:
            if k in cfg:
                _validate_numeric_range(k, cfg[k], result, min_val=lo, max_val=hi)

    _validate_with_config_schema(result, cfg if cfg and "__error__" not in cfg else None)

    preset = getattr(args, "preset", None)
    if preset:
        _validate_preset(preset, result)

    result.compute_summary()
    return result
