import os
import sys
import platform
from typing import Any, Dict, List, Optional

from modules.tools.common import ToolResult, Severity


REQUIRED_PYTHON_VERSION = (3, 9)
RECOMMENDED_PYTHON_VERSION = (3, 10)


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


def _parse_requirements(path: str) -> Dict[str, str]:
    reqs: Dict[str, str] = {}
    if not os.path.exists(path):
        return reqs
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            for sep in ["==", ">=", "<=", "~=", "!="]:
                if sep in line:
                    name, _, _ = line.partition(sep)
                    reqs[name.strip().lower()] = line.strip()
                    break
            else:
                reqs[line.strip().lower()] = line.strip()
    return reqs


def _check_python_version(result: ToolResult) -> None:
    py_ver = sys.version_info[:3]
    py_ver_str = ".".join(str(v) for v in py_ver)
    if py_ver < REQUIRED_PYTHON_VERSION:
        required = ".".join(str(v) for v in REQUIRED_PYTHON_VERSION)
        result.add_error(
            name="python_version",
            message=f"Python version {py_ver_str} is below required {required}",
            suggestion=f"Upgrade Python to {required} or newer.",
            detail={"current": py_ver_str, "required": required},
        )
    elif py_ver < RECOMMENDED_PYTHON_VERSION:
        recommended = ".".join(str(v) for v in RECOMMENDED_PYTHON_VERSION)
        result.add_warning(
            name="python_version",
            message=f"Python version {py_ver_str} works but {recommended}+ is recommended",
            detail={"current": py_ver_str, "recommended": recommended},
        )
    else:
        result.add_info(
            name="python_version",
            message=f"Python {py_ver_str} is compatible",
            detail={"version": py_ver_str},
        )
    result.data["python_version"] = py_ver_str


def _check_platform(result: ToolResult) -> None:
    system = platform.system()
    machine = platform.machine()
    result.data["platform"] = {"system": system, "machine": machine}
    result.add_info(
        name="platform",
        message=f"Running on {system} ({machine})",
        detail={"system": system, "machine": machine},
    )


def _check_dependency(name: str, result: ToolResult, required: bool = True) -> None:
    try:
        module = __import__(name.lower() if name not in ("PIL", "cv2") else name)
        version = getattr(module, "__version__", "unknown")
        result.add_info(
            name=f"dep_{name.lower()}",
            message=f"{name} installed: {version}",
            detail={"package": name, "version": version},
        )
        result.data.setdefault("dependencies", {})[name] = {"installed": True, "version": version}
    except ImportError:
        msg = f"Required dependency '{name}' is not installed"
        if required:
            result.add_error(
                name=f"dep_{name.lower()}",
                message=msg,
                suggestion=f"Run: pip install {name}",
            )
            result.data.setdefault("dependencies", {})[name] = {"installed": False}
        else:
            result.add_warning(
                name=f"dep_{name.lower()}",
                message=f"Optional dependency '{name}' is not installed",
                suggestion=f"Install for full features: pip install {name}",
            )
            result.data.setdefault("dependencies", {})[name] = {"installed": False, "optional": True}


def _check_torch(result: ToolResult) -> None:
    try:
        import torch
        version = getattr(torch, "__version__", "unknown")
        cuda_available = torch.cuda.is_available()
        result.add_info(
            name="dep_torch",
            message=f"PyTorch {version} installed (CUDA available: {cuda_available})",
            detail={
                "version": version,
                "cuda_available": cuda_available,
                "cuda_version": torch.version.cuda if cuda_available else None,
            },
        )
        result.data.setdefault("dependencies", {})["torch"] = {
            "installed": True,
            "version": version,
            "cuda_available": cuda_available,
        }
    except ImportError:
        result.add_error(
            name="dep_torch",
            message="PyTorch is not installed or could not be imported",
            suggestion="Install PyTorch per README or run: python launch.py",
        )
        result.data.setdefault("dependencies", {})["torch"] = {"installed": False}


def _check_requirements_coverage(result: ToolResult, req_path: Optional[str]) -> None:
    if req_path is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(root, "requirements_versions.txt")
        if os.path.exists(candidate):
            req_path = candidate
    if not req_path or not os.path.exists(req_path):
        result.add_warning(
            name="requirements_file",
            message="Could not locate requirements_versions.txt to cross-check",
        )
        return

    reqs = _parse_requirements(req_path)
    result.add_info(
        name="requirements_file",
        message=f"Found {len(reqs)} requirements in {os.path.basename(req_path)}",
        detail={"path": req_path, "count": len(reqs)},
    )
    result.data["requirements_file"] = req_path

    missing: List[str] = []
    for pkg_name in reqs:
        import_name = {
            "pillow": "PIL",
            "opencv-python": "cv2",
            "opencv-python-headless": "cv2",
            "scikit-image": "skimage",
            "accelerate": "accelerate",
        }.get(pkg_name, pkg_name.replace("-", "_"))
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    if missing:
        result.add_warning(
            name="requirements_missing",
            message=f"{len(missing)} package(s) from requirements_versions.txt are missing",
            suggestion=f"Run: pip install -r {req_path}",
            detail={"missing": missing[:20], "total_missing": len(missing)},
        )


def check_environment(args: Any) -> ToolResult:
    result = ToolResult(tool_name="check-env")
    _check_python_version(result)
    _check_platform(result)

    core_deps = ["numpy", "PIL", "gradio", "cv2"]
    for dep in core_deps:
        _check_dependency(dep, result, required=True)

    optional_deps = ["trimesh", "rembg"]
    for dep in optional_deps:
        _check_dependency(dep, result, required=False)

    if not getattr(args, "skip_torch", False):
        _check_torch(result)

    req_file = getattr(args, "requirements_file", None)
    _check_requirements_coverage(result, req_file)

    result.compute_summary()
    return result
