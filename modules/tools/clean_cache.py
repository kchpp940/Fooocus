import os
import sys
import shutil
import tempfile
import json
from typing import Any

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from modules.tools.common import ToolResult
from modules.services.config_inspector import (
    resolve_effective_paths,
    load_config_overrides,
)


def add_clean_cache_args(parser) -> None:
    parser.add_argument(
        "--hash",
        action="store_true",
        dest="clean_hash",
        help="Remove the hash_cache.txt file",
    )
    parser.add_argument(
        "--temp",
        action="store_true",
        dest="clean_temp",
        help="Remove contents of the temp directory (fooocus temp folder)",
    )
    parser.add_argument(
        "--outputs-log",
        action="store_true",
        dest="clean_outputs_log",
        help="Remove log.html files in output directories (keep images)",
    )
    parser.add_argument(
        "--gradio-temp",
        action="store_true",
        dest="clean_gradio_temp",
        help="Clean Gradio temp directory (under the configured temp path)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="clean_all",
        help="Apply all cleaning options above (--hash --temp --outputs-log --gradio-temp)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually deleting anything",
    )
    parser.add_argument(
        "--force", "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt",
    )


def _delete_file(path: str, result: ToolResult, label: str, dry_run: bool) -> None:
    if not os.path.isfile(path):
        result.add_info(
            name=f"{label}_missing",
            message=f"Nothing to do: {path} is not a file or does not exist",
        )
        return
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    if dry_run:
        result.add_info(
            name=f"{label}_would_remove",
            message=f"Would remove file: {path} ({size / 1024:.1f} KB)",
            detail={"path": path, "size_bytes": size},
        )
        return
    try:
        os.remove(path)
        result.add_info(
            name=f"{label}_removed",
            message=f"Removed file: {path} ({size / 1024:.1f} KB)",
            detail={"path": path, "size_bytes": size},
        )
        result.data.setdefault("total_bytes_freed", 0)
        result.data["total_bytes_freed"] += size
        result.data.setdefault("removed_files", 0)
        result.data["removed_files"] += 1
    except Exception as e:
        result.add_error(
            name=f"{label}_failed",
            message=f"Failed to remove file {path}: {type(e).__name__}: {e}",
        )


def _clear_directory_contents(dir_path: str, result: ToolResult, label: str,
                               dry_run: bool, keep_files: list = None) -> None:
    keep_files = set(keep_files or [])
    if not os.path.isdir(dir_path):
        result.add_warning(
            name=f"{label}_missing",
            message=f"Directory does not exist: {dir_path}",
        )
        return
    removed = 0
    freed = 0
    errors = 0
    for entry in sorted(os.listdir(dir_path), key=lambda s: s.casefold()):
        if entry in keep_files:
            continue
        full = os.path.join(dir_path, entry)
        try:
            if os.path.islink(full):
                if dry_run:
                    result.add_info(
                        name=f"{label}_would_remove_link",
                        message=f"Would remove symlink: {full}",
                    )
                    continue
                os.unlink(full)
                removed += 1
                result.data.setdefault("removed_symlinks", 0)
                result.data["removed_symlinks"] += 1
            elif os.path.isfile(full):
                try:
                    sz = os.path.getsize(full)
                except OSError:
                    sz = 0
                if dry_run:
                    result.add_info(
                        name=f"{label}_would_remove_file",
                        message=f"Would remove file: {full} ({sz / 1024:.1f} KB)",
                    )
                    continue
                os.remove(full)
                removed += 1
                freed += sz
                result.data.setdefault("removed_files", 0)
                result.data["removed_files"] += 1
            elif os.path.isdir(full):
                if dry_run:
                    count = sum(1 for _ in os.walk(full) for __ in _[2])
                    result.add_info(
                        name=f"{label}_would_remove_tree",
                        message=f"Would remove directory tree: {full} (~{count} files)",
                    )
                    continue
                for r, _, files in os.walk(full):
                    for fn in files:
                        fp = os.path.join(r, fn)
                        try:
                            freed += os.path.getsize(fp)
                        except OSError:
                            pass
                shutil.rmtree(full, ignore_errors=False)
                removed += 1
                result.data.setdefault("removed_dirs", 0)
                result.data["removed_dirs"] += 1
        except Exception as e:
            errors += 1
            result.add_warning(
                name=f"{label}_item_failed",
                message=f"Failed to remove {full}: {type(e).__name__}: {e}",
            )
    result.data.setdefault("total_bytes_freed", 0)
    result.data["total_bytes_freed"] += freed
    result.add_info(
        name=f"{label}_done",
        message=(f"Processed directory '{dir_path}': removed {removed} item(s); "
                 f"freed ~{freed / (1024 * 1024):.2f} MB. Errors: {errors}"),
        detail={
            "items_removed": removed,
            "bytes_freed": freed,
            "errors": errors,
        },
    )


def clean_cache(args: Any) -> ToolResult:
    result = ToolResult(tool_name="clean-cache")

    clean_all = getattr(args, "clean_all", False)
    clean_hash = clean_all or getattr(args, "clean_hash", False)
    clean_temp = clean_all or getattr(args, "clean_temp", False)
    clean_outputs_log = clean_all or getattr(args, "clean_outputs_log", False)
    clean_gradio_temp = clean_all or getattr(args, "clean_gradio_temp", False)
    dry_run = getattr(args, "dry_run", False)
    force = getattr(args, "force", False)

    any_selected = clean_hash or clean_temp or clean_outputs_log or clean_gradio_temp
    if not any_selected:
        result.add_error(
            name="no_action",
            message="No cleaning option selected",
            suggestion="Pass at least one of: --hash, --temp, --outputs-log, --gradio-temp, or --all",
        )
        result.compute_summary()
        return result

    result.data["options"] = {
        "clean_hash": clean_hash,
        "clean_temp": clean_temp,
        "clean_outputs_log": clean_outputs_log,
        "clean_gradio_temp": clean_gradio_temp,
        "clean_all": clean_all,
        "dry_run": dry_run,
    }

    hash_cache_path = os.path.join(_root, "hash_cache.txt")

    cfg_override = load_config_overrides(project_root=_root)["data"]
    paths_info = resolve_effective_paths(cfg_override, project_root=_root)
    outputs_dir = (
        paths_info["by_key"]["path_outputs"]["effective"][0]
        if paths_info["by_key"].get("path_outputs", {}).get("effective")
        else os.path.join(_root, "outputs")
    )
    temp_dir = paths_info["temp_path"]["effective"]

    if not dry_run and not force:
        actions = []
        if clean_hash:
            actions.append(f"Remove hash cache file ({hash_cache_path})")
        if clean_temp:
            actions.append(f"Clear temp directory contents ({temp_dir})")
        if clean_outputs_log:
            actions.append("Remove log.html files in output directories")
        if clean_gradio_temp:
            actions.append(f"Clear Gradio temp directory (under {temp_dir})")
        print("=" * 60)
        print("About to perform the following actions:")
        for i, a in enumerate(actions, 1):
            print(f"  [{i}] {a}")
        print("=" * 60)
        try:
            resp = input("Continue? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            resp = "n"
        if resp not in ("y", "yes"):
            result.add_info(name="user_aborted", message="Aborted by user (no confirmation)")
            result.compute_summary()
            return result

    if clean_hash:
        _delete_file(hash_cache_path, result, "hash_cache", dry_run)

    if clean_temp:
        _clear_directory_contents(temp_dir, result, "temp_dir", dry_run)

    if clean_outputs_log and os.path.isdir(outputs_dir):
        logs_found = 0
        for root, _, files in os.walk(outputs_dir):
            for fname in files:
                if fname.lower() == "log.html":
                    full = os.path.join(root, fname)
                    if dry_run:
                        try:
                            sz = os.path.getsize(full)
                        except OSError:
                            sz = 0
                        result.add_info(
                            name="outputs_log_would_remove",
                            message=f"Would remove: {full} ({sz / 1024:.1f} KB)",
                        )
                        logs_found += 1
                        continue
                    try:
                        sz = os.path.getsize(full)
                        os.remove(full)
                        result.add_info(
                            name="outputs_log_removed",
                            message=f"Removed: {full} ({sz / 1024:.1f} KB)",
                        )
                        result.data.setdefault("total_bytes_freed", 0)
                        result.data["total_bytes_freed"] += sz
                        result.data.setdefault("removed_files", 0)
                        result.data["removed_files"] += 1
                        logs_found += 1
                    except Exception as e:
                        result.add_warning(
                            name="outputs_log_failed",
                            message=f"Failed to remove {full}: {e}",
                        )
        result.add_info(
            name="outputs_log_done",
            message=f"Processed output log files: {logs_found} file(s) affected",
        )

    if clean_gradio_temp:
        gt_dirs = [
            os.path.join(temp_dir, "gradio"),
            os.path.join(temp_dir, "blocks"),
        ]
        any_gradio = False
        for gt_dir in gt_dirs:
            if os.path.isdir(gt_dir):
                any_gradio = True
                _clear_directory_contents(gt_dir, result, f"gradio_temp[{os.path.basename(gt_dir)}]", dry_run)
        if not any_gradio:
            result.add_info(
                name="gradio_temp_missing",
                message="No Gradio temp subdirectory found under temp dir",
                detail={"temp_dir": temp_dir},
            )

    total = result.data.get("total_bytes_freed", 0)
    if total:
        result.add_info(
            name="total_freed",
            message=f"Total disk space freed: {total / (1024 * 1024):.2f} MB",
            detail={"bytes_freed": total},
        )

    result.compute_summary()
    return result
