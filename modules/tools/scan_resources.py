from __future__ import annotations

import os
from typing import Any, Dict, List

from modules.tools.common import ToolResult

from modules.services.resource_scanner import (
    run_resource_scan,
    DEFAULT_MODEL_EXTS,
    WILDCARD_EXTS,
)


def add_scan_resources_args(parser) -> None:
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only scan model directories (skip presets and wildcards)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Placeholder (kept for CLI compatibility). In isolated maintenance CLI context this flag has no in-process state to refresh.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help=f"Model file extensions to match (default: {' '.join(DEFAULT_MODEL_EXTS)})",
    )


def scan_resources(args: Any) -> ToolResult:
    result = ToolResult(tool_name="scan-resources")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_only = bool(getattr(args, "models_only", False))
    exts: List[str] = list(getattr(args, "extensions", None) or DEFAULT_MODEL_EXTS)

    scan = run_resource_scan(
        scan_models=True,
        include_presets=not models_only,
        include_wildcards=not models_only,
        model_extensions=exts,
        wildcard_extensions=WILDCARD_EXTS,
        project_root=project_root,
    )

    models = scan.get("models") or {"by_category": {}, "total_count": 0}
    total = 0
    counts: Dict[str, int] = {}
    for category, info in models.get("by_category", {}).items():
        count = int(info.get("count", 0))
        counts[category] = count
        total += count
        for folder, files in (info.get("files_by_folder") or {}).items():
            result.add_info(
                name=f"scan_{category}",
                message=f"Scanned {folder}: {len(files)} file(s)",
                detail={
                    "folder": folder,
                    "count": len(files),
                    "extensions": exts,
                },
            )

    result.data["models_by_category"] = counts
    result.data["total_model_files"] = total
    result.data["model_extensions"] = exts

    if not models_only:
        presets = scan.get("presets") or {}
        builtin = list(presets.get("builtin", []))
        user = list(presets.get("user", []))
        result.add_info(
            name="scan_presets_builtin",
            message=f"Found {len(builtin)} built-in preset(s)",
            detail={"count": len(builtin), "presets": builtin[:30]},
        )
        result.data["builtin_presets"] = builtin
        if user:
            result.add_info(
                name="scan_presets_user",
                message=f"Found {len(user)} user preset(s)",
                detail={"count": len(user), "presets": user[:30]},
            )
            result.data["user_presets"] = user

        wildcards = scan.get("wildcards") or {}
        all_wildcard_files: List[str] = []
        for folder, files in (wildcards.get("files_by_folder") or {}).items():
            all_wildcard_files.extend(files)
            result.add_info(
                name="scan_wildcards",
                message=f"Scanned {folder}: {len(files)} wildcard file(s)",
                detail={"folder": folder, "count": len(files)},
            )
        result.data["wildcards"] = sorted(set(all_wildcard_files))

    if getattr(args, "refresh", False):
        result.add_warning(
            name="refresh",
            message="--refresh flag has no effect in the isolated maintenance CLI; "
                    "use scan output directly, or call modules.config.update_files() from the running app.",
        )

    if total == 0:
        result.add_warning(
            name="no_models_found",
            message="No model files were found in any category",
            suggestion="Add model files to models/ subdirectories or configure custom paths.",
        )

    result.compute_summary()
    return result
