import os
import sys
import json
from typing import Any, Dict, List, Optional

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from modules.tools.common import ToolResult


DEFAULT_MODEL_EXTS = [".pth", ".ckpt", ".bin", ".safetensors", ".fooocus.patch"]
WILDCARD_EXTS = [".txt"]


def add_scan_resources_args(parser) -> None:
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only scan model directories (skip presets and wildcards)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force a refresh by calling modules.config.update_files()",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="Override the list of model file extensions to look for",
    )


def _scan_folder(folder: str, exts: List[str], result: ToolResult, label: str) -> List[str]:
    if not os.path.isdir(folder):
        result.add_warning(
            name=f"scan_{label}",
            message=f"Directory does not exist: {folder}",
            detail={"folder": folder},
        )
        return []

    found: List[str] = []
    for root, _, files in os.walk(folder, topdown=False):
        rel = os.path.relpath(root, folder)
        if rel == ".":
            rel = ""
        for fname in sorted(files, key=lambda s: s.casefold()):
            _, ext = os.path.splitext(fname)
            if ext.lower() in exts:
                found.append(os.path.join(rel, fname) if rel else fname)
    result.add_info(
        name=f"scan_{label}",
        message=f"Scanned {folder}: {len(found)} file(s)",
        detail={"folder": folder, "count": len(found), "extensions": exts},
    )
    return found


def _read_safe_config_paths(root: str, result: ToolResult) -> Dict[str, List[str]]:
    config_path = os.path.join(root, "config.txt")
    override: Dict[str, Any] = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                override = json.load(f)
        except Exception as e:
            result.add_warning(
                name="config_parse",
                message=f"Could not parse config.txt for custom paths: {e}",
            )

    def resolve_path(default_rel: str, key: Optional[str] = None,
                      is_array: bool = False) -> List[str]:
        if key and key in override:
            val = override[key]
            if isinstance(val, list):
                return [os.path.abspath(v) for v in val if isinstance(v, str)]
            if isinstance(val, str):
                return [os.path.abspath(val)]
        default = os.path.abspath(os.path.join(root, default_rel))
        return [default]

    return {
        "checkpoints": resolve_path("models/checkpoints", "path_checkpoints", is_array=True),
        "loras": resolve_path("models/loras", "path_loras", is_array=True),
        "vae": resolve_path("models/vae", "path_vae"),
        "vae_approx": resolve_path("models/vae_approx", "path_vae_approx"),
        "upscale_models": resolve_path("models/upscale_models", "path_upscale_models"),
        "inpaint": resolve_path("models/inpaint", "path_inpaint"),
        "controlnet": resolve_path("models/controlnet", "path_controlnet"),
        "clip_vision": resolve_path("models/clip_vision", "path_clip_vision"),
        "embeddings": resolve_path("models/embeddings", "path_embeddings"),
        "safety_checker": resolve_path("models/safety_checker", "path_safety_checker"),
        "sam": resolve_path("models/sam", "path_sam"),
    }


def scan_resources(args: Any) -> ToolResult:
    result = ToolResult(tool_name="scan-resources")

    exts = getattr(args, "extensions", None) or DEFAULT_MODEL_EXTS
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths = _read_safe_config_paths(root, result)

    all_models: Dict[str, List[str]] = {}
    total = 0
    for label, folder_list in paths.items():
        all_models[label] = []
        for idx, folder in enumerate(folder_list):
            sub_label = label if len(folder_list) == 1 else f"{label}_{idx}"
            files = _scan_folder(folder, exts, result, sub_label)
            all_models[label].extend(files)
            total += len(files)

    result.data["models_by_category"] = {k: len(v) for k, v in all_models.items()}
    result.data["total_model_files"] = total
    result.data["model_extensions"] = exts

    if not getattr(args, "models_only", False):
        preset_folder = os.path.join(root, "presets")
        user_preset_folder = os.path.join(
            os.environ.get("FOOOCUS_USER_DATA_DIR") or
            os.path.join(os.path.expanduser("~"), ".fooocus"),
            "user_presets",
        )

        builtin_presets = _scan_folder(preset_folder, [".json"], result, "presets_builtin")
        result.data["builtin_presets"] = sorted(builtin_presets)

        if user_preset_folder and os.path.isdir(user_preset_folder):
            user_presets = _scan_folder(user_preset_folder, [".json"], result, "presets_user")
            result.data["user_presets"] = sorted(user_presets)

        wildcard_folder = os.path.join(root, "wildcards")
        override = {}
        config_path = os.path.join(root, "config.txt")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    override = json.load(f)
            except Exception:
                pass
        if "path_wildcards" in override and isinstance(override["path_wildcards"], str):
            wildcard_folder = os.path.abspath(override["path_wildcards"])
        wildcards = _scan_folder(wildcard_folder, WILDCARD_EXTS, result, "wildcards")
        result.data["wildcards"] = sorted(wildcards)

    if getattr(args, "refresh", False):
        result.add_warning(
            name="refresh",
            message="--refresh flag requires the main modules.config state; "
                    "this maintenance CLI runs in an isolated context so it has no in-process state to refresh. "
                    "Use scan output directly instead.",
        )

    if total == 0:
        result.add_warning(
            name="no_models_found",
            message="No model files were found in any category",
            suggestion="Add model files to models/ subdirectories or configure custom paths.",
        )

    result.compute_summary()
    return result
