import os
import sys
import tarfile
import tempfile
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from modules.tools.common import ToolResult
from modules.services.config_inspector import (
    resolve_effective_paths,
    load_config_overrides,
    get_user_data_dir,
    get_builtin_presets_dir,
    get_user_presets_dir,
)


def add_package_logs_args(parser) -> None:
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output tarball path (default: auto-named in cwd)",
    )
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="Include all generated image outputs (may be large!)",
    )
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Include model directory listings (not the actual model files)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Max number of most recent output images to include (default: 5)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["gztar", "tar", "zip"],
        default="gztar",
        help="Archive format (default: gztar)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be included without actually creating the archive",
    )


def _default_output_name(fmt: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = {
        "gztar": "tar.gz",
        "tar": "tar",
        "zip": "zip",
    }[fmt]
    return f"fooocus_logs_{ts}.{ext}"


def _collect_latest_images(folder: str, limit: int) -> List[str]:
    if not os.path.isdir(folder) or limit <= 0:
        return []
    images = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if Path(fname).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                full = os.path.join(root, fname)
                try:
                    st = os.stat(full)
                    images.append((st.st_mtime, full))
                except OSError:
                    continue
    images.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in images[:limit]]


def _add_dir_listing_to_manifest(dir_path: str, label: str,
                                 tmpdir: str, result: ToolResult) -> Optional[str]:
    if not os.path.isdir(dir_path):
        result.add_warning(
            name=f"listing_{label}",
            message=f"Directory not found for listing: {dir_path}",
        )
        return None
    entries = []
    for root, dirs, files in os.walk(dir_path):
        rel = os.path.relpath(root, dir_path)
        for d in sorted(dirs, key=lambda s: s.casefold()):
            entries.append(f"D {os.path.join(rel, d)}")
        for f in sorted(files, key=lambda s: s.casefold()):
            full = os.path.join(root, f)
            try:
                sz = os.path.getsize(full)
            except OSError:
                sz = -1
            entries.append(f"F {sz:>12d} {os.path.join(rel, f)}")
    manifest_path = os.path.join(tmpdir, f"manifest_{label}.txt")
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(f"# Manifest for {label}: {dir_path}\n")
            f.write(f"# Generated {datetime.now().isoformat()}\n")
            f.write("\n".join(entries))
            f.write("\n")
        result.add_info(
            name=f"listing_{label}",
            message=f"Included {label} manifest ({len(entries)} entries)",
            detail={"entries_count": len(entries)},
        )
        return manifest_path
    except Exception as e:
        result.add_warning(
            name=f"listing_{label}",
            message=f"Could not write listing for {label}: {e}",
        )
        return None


def package_logs(args: Any) -> ToolResult:
    result = ToolResult(tool_name="package-logs")

    fmt = getattr(args, "format", "gztar")
    dry_run = getattr(args, "dry_run", False)
    include_outputs = getattr(args, "include_outputs", False)
    include_models = getattr(args, "include_models", False)
    max_images = getattr(args, "max_images", 5)
    output = getattr(args, "output", None) or _default_output_name(fmt)

    output = os.path.abspath(output)
    result.data["output_path"] = output
    result.data["format"] = fmt
    result.data["dry_run"] = dry_run

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    source_paths: List[tuple] = []

    user_data_dir = get_user_data_dir(root)
    config_overrides = load_config_overrides(project_root=root)["data"]
    paths_info = resolve_effective_paths(config_overrides, project_root=root)

    config_path = load_config_overrides(project_root=root)["path"]
    outputs_dir = (
        paths_info["by_key"]["path_outputs"]["effective"][0]
        if paths_info["by_key"].get("path_outputs", {}).get("effective")
        else os.path.join(root, "outputs")
    )
    temp_dir = paths_info["temp_path"]["effective"]
    presets_dir = get_builtin_presets_dir(root)
    user_presets_dir = get_user_presets_dir(root)

    candidate_files = [
        (config_path, "config.txt"),
        (os.path.join(root, "config_modification_tutorial.txt"), "config_modification_tutorial.txt"),
        (os.path.join(root, "hash_cache.txt"), "hash_cache.txt"),
    ]
    for src, arcname in candidate_files:
        if src and os.path.isfile(src):
            source_paths.append((src, arcname))

    if os.path.isdir(user_data_dir):
        for fname in sorted(os.listdir(user_data_dir)):
            full = os.path.join(user_data_dir, fname)
            if os.path.isfile(full):
                source_paths.append((full, os.path.join("user_data", fname)))
            elif os.path.isdir(full) and fname == "user_presets":
                for pname in sorted(os.listdir(full)):
                    pfull = os.path.join(full, pname)
                    if os.path.isfile(pfull):
                        source_paths.append((pfull, os.path.join("user_data", "user_presets", pname)))

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="fooocus_pkg_")

        env_dump_path = os.path.join(tmpdir, "environment.txt")
        try:
            with open(env_dump_path, "w", encoding="utf-8") as f:
                f.write(f"Fooocus log package generated: {datetime.now().isoformat()}\n")
                f.write(f"Python: {sys.version}\n")
                f.write(f"Platform: {sys.platform}\n")
                try:
                    import platform
                    f.write(f"Platform details: {platform.platform()}\n")
                    f.write(f"Machine: {platform.machine()}\n")
                except Exception:
                    pass
                f.write(f"Working directory: {os.getcwd()}\n")
                f.write(f"Project root: {root}\n")
                f.write("\n--- ENVIRONMENT VARIABLES (redacted) ---\n")
                safe_env_prefixes = ("FOOOCUS_", "PYTHON", "TORCH", "CUDA", "HF_", "GRADIO")
                for k in sorted(os.environ.keys()):
                    if any(k.upper().startswith(p) for p in safe_env_prefixes):
                        v = os.environ[k]
                        if any(s in k.upper() for s in ["KEY", "SECRET", "TOKEN", "PASSWORD"]):
                            v = "<redacted>"
                        f.write(f"{k}={v}\n")
        except Exception as e:
            result.add_warning(name="env_dump", message=f"Could not write environment dump: {e}")
        else:
            source_paths.append((env_dump_path, "environment.txt"))

        if include_models:
            models_root = os.path.join(root, "models")
            listing = _add_dir_listing_to_manifest(models_root, "models", tmpdir, result)
            if listing:
                source_paths.append((listing, os.path.basename(listing)))

        if include_outputs and os.path.isdir(outputs_dir):
            count = 0
            for src, arc in source_paths:
                pass
            for r, _, files in os.walk(outputs_dir):
                rel = os.path.relpath(r, outputs_dir)
                for fname in files:
                    full = os.path.join(r, fname)
                    rel_arc = os.path.join("outputs", rel if rel != "." else "", fname)
                    source_paths.append((full, rel_arc))
                    count += 1
            result.add_info(
                name="outputs_included",
                message=f"Including all outputs: {count} file(s)",
                detail={"count": count},
            )
        else:
            latest_imgs = _collect_latest_images(outputs_dir, max_images)
            if latest_imgs:
                for i, full in enumerate(latest_imgs, 1):
                    rel = os.path.relpath(full, outputs_dir)
                    source_paths.append((full, os.path.join("outputs_sample", rel)))
                result.add_info(
                    name="outputs_sample",
                    message=f"Including {len(latest_imgs)} most recent output image(s)",
                    detail={"count": len(latest_imgs)},
                )
            logs_found = False
            if os.path.isdir(outputs_dir):
                for item in sorted(os.listdir(outputs_dir)):
                    sub = os.path.join(outputs_dir, item)
                    if os.path.isdir(sub):
                        for subfile in sorted(os.listdir(sub)):
                            if subfile.lower() == "log.html":
                                source_paths.append((
                                    os.path.join(sub, subfile),
                                    os.path.join("outputs", item, subfile),
                                ))
                                logs_found = True
            if logs_found:
                result.add_info(
                    name="logs_included",
                    message="Included log.html files from output directories",
                )

        listing = _add_dir_listing_to_manifest(presets_dir, "presets_builtin", tmpdir, result)
        if listing:
            source_paths.append((listing, os.path.basename(listing)))

    except Exception as e:
        result.add_warning(
            name="collection_error",
            message=f"Error during file collection: {type(e).__name__}: {e}",
        )

    source_paths = list({p[1]: p for p in source_paths}.values())

    result.data["files_to_package_count"] = len(source_paths)
    result.data["files_to_package"] = [arc for _, arc in source_paths]

    if dry_run:
        result.add_info(
            name="dry_run",
            message=f"Dry run: {len(source_paths)} file(s) would be packaged into {output}",
        )
        result.compute_summary()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return result

    try:
        if fmt == "zip":
            import zipfile
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
                for src, arc in source_paths:
                    if os.path.isfile(src):
                        zf.write(src, arcname=arc)
        else:
            mode = "w:gz" if fmt == "gztar" else "w"
            with tarfile.open(output, mode) as tf:
                for src, arc in source_paths:
                    if os.path.isfile(src):
                        tf.add(src, arcname=arc)
        size = os.path.getsize(output)
        result.add_info(
            name="archive_created",
            message=f"Created {output} ({size / (1024 * 1024):.2f} MB) with {len(source_paths)} entry(ies)",
            detail={"size_bytes": size, "entries_count": len(source_paths)},
        )
        result.data["archive_size_bytes"] = size
    except Exception as e:
        result.add_error(
            name="archive_failed",
            message=f"Failed to create archive: {type(e).__name__}: {e}",
            suggestion="Check that the output path is writable and has enough disk space.",
        )
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    result.compute_summary()
    return result
