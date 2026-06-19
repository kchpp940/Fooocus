from __future__ import annotations

import os
from typing import Any

from modules.tools.common import ToolResult

from modules.services.environment_inspector import (
    run_environment_inspection,
)


def add_check_env_args(parser) -> None:
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip PyTorch version check (useful in CI without GPU)",
    )
    parser.add_argument(
        "--requirements-file",
        type=str,
        default=None,
        help="Path to requirements_versions.txt (auto-detected if omitted)",
    )


def check_environment(args: Any) -> ToolResult:
    result = ToolResult(tool_name="check-env")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    req_file = getattr(args, "requirements_file", None)
    skip_torch = bool(getattr(args, "skip_torch", False))

    inspection = run_environment_inspection(
        project_root=project_root,
        skip_torch=skip_torch,
        requirements_file=req_file,
    )

    py = inspection["python"]
    if not py["compatible"]:
        result.add_error(
            name="python_version",
            message=f"Python version {py['version']} is below required {py['minimum_required']}",
            suggestion=f"Upgrade Python to {py['minimum_required']} or newer.",
            detail={"current": py["version"], "required": py["minimum_required"]},
        )
    elif not py["recommended"]:
        result.add_warning(
            name="python_version",
            message=f"Python version {py['version']} works but {py['minimum_recommended']}+ is recommended",
            detail={"current": py["version"], "recommended": py["minimum_recommended"]},
        )
    else:
        result.add_info(
            name="python_version",
            message=f"Python {py['version']} is compatible",
            detail={"version": py["version"]},
        )
    result.data["python_version"] = py["version"]

    plat = inspection["platform"]
    result.add_info(
        name="platform",
        message=f"Running on {plat['system']} ({plat['machine']})",
        detail={"system": plat["system"], "machine": plat["machine"]},
    )
    result.data["platform"] = {"system": plat["system"], "machine": plat["machine"]}

    deps_out: dict = {}
    for pkg_key, info in inspection["dependencies"].items():
        optional = info.get("optional", False)
        if info["installed"]:
            version = info.get("version", "unknown")
            if pkg_key == "torch":
                try:
                    import torch
                    cuda_ok = bool(torch.cuda.is_available())
                    detail = {
                        "version": version,
                        "cuda_available": cuda_ok,
                        "cuda_version": torch.version.cuda if cuda_ok else None,
                    }
                    result.add_info(
                        name="dep_torch",
                        message=f"PyTorch {version} installed (CUDA available: {cuda_ok})",
                        detail=detail,
                    )
                    deps_out["torch"] = {
                        "installed": True,
                        "version": version,
                        "cuda_available": cuda_ok,
                    }
                except Exception:
                    result.add_info(
                        name=f"dep_{pkg_key.lower()}",
                        message=f"{pkg_key} installed: {version}",
                        detail={"package": pkg_key, "version": version},
                    )
                    deps_out[pkg_key] = {"installed": True, "version": version}
            else:
                result.add_info(
                    name=f"dep_{pkg_key.lower()}",
                    message=f"{pkg_key} installed: {version}",
                    detail={"package": pkg_key, "version": version},
                )
                deps_out[pkg_key] = {"installed": True, "version": version}
        else:
            pip_name = info.get("pip_name", pkg_key)
            if optional:
                result.add_warning(
                    name=f"dep_{pkg_key.lower()}",
                    message=f"Optional dependency '{pkg_key}' is not installed",
                    suggestion=f"Install for full features: pip install {pip_name}",
                )
                deps_out[pkg_key] = {"installed": False, "optional": True}
            else:
                result.add_error(
                    name=f"dep_{pkg_key.lower()}",
                    message=f"Required dependency '{pkg_key}' is not installed",
                    suggestion=f"Run: pip install {pip_name}",
                )
                deps_out[pkg_key] = {"installed": False}
    result.data["dependencies"] = deps_out

    req = inspection["requirements_file"]
    if not req["found"]:
        result.add_warning(
            name="requirements_file",
            message="Could not locate requirements_versions.txt to cross-check",
        )
    else:
        result.add_info(
            name="requirements_file",
            message=f"Found {req['count']} requirements in {os.path.basename(req['path'])}",
            detail={"path": req["path"], "count": req["count"]},
        )
        result.data["requirements_file"] = req["path"]
        if req["missing"]:
            result.add_warning(
                name="requirements_missing",
                message=f"{req['total_missing']} package(s) from requirements_versions.txt are missing",
                suggestion=f"Run: pip install -r {req['path']}",
                detail={"missing": req["missing"][:20], "total_missing": req["total_missing"]},
            )

    result.compute_summary()
    return result
