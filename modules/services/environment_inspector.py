"""
无副作用的环境检查 service 层。

本模块不依赖 modules.config / args_manager，不触发任何顶层副作用。
启动流程和 CLI 工具都应调用这里的函数，而不是在各自模块里重复实现。

只返回原始数据结构（dict/list），不做格式化输出，不操作 sys.exit。
"""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


MIN_PYTHON_VERSION = (3, 9)
RECOMMENDED_PYTHON_VERSION = (3, 10)

CORE_REQUIRED: List[str] = [
    "numpy",
    "PIL",
    "gradio",
    "cv2",
]

CORE_OPTIONAL: List[str] = [
    "trimesh",
    "rembg",
]

IMPORT_NAMES: Dict[str, str] = {
    "PIL": "PIL",
    "cv2": "cv2",
    "numpy": "numpy",
    "gradio": "gradio",
    "torch": "torch",
    "trimesh": "trimesh",
    "rembg": "rembg",
}

PIP_NAMES: Dict[str, str] = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "gradio": "gradio",
    "torch": "torch",
    "trimesh": "trimesh",
    "rembg": "rembg",
}


def get_python_version_tuple() -> Tuple[int, ...]:
    return sys.version_info[:3]


def get_python_version_str() -> str:
    return "{}.{}.{}".format(*sys.version_info[:3])


def get_platform_info() -> Dict[str, str]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
    }


def check_python_compatibility() -> Dict[str, Any]:
    """返回 Python 版本兼容性检查结果。"""
    version = get_python_version_tuple()
    compatible = version >= MIN_PYTHON_VERSION
    recommended = version >= RECOMMENDED_PYTHON_VERSION
    return {
        "version": get_python_version_str(),
        "version_tuple": list(version),
        "compatible": compatible,
        "recommended": recommended,
        "minimum_required": ".".join(str(x) for x in MIN_PYTHON_VERSION),
        "minimum_recommended": ".".join(str(x) for x in RECOMMENDED_PYTHON_VERSION),
    }


def try_import_module(import_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """尝试 import 一个模块，返回 (是否安装, 版本或None, 错误信息或None)。"""
    try:
        mod = __import__(import_name)
    except Exception as exc:  # ImportError / ModuleNotFoundError / 其它
        return False, None, f"{type(exc).__name__}: {exc}"
    version = getattr(mod, "__version__", None)
    if not isinstance(version, str) or not version:
        version = None
    return True, version, None


def check_package(pkg_key: str) -> Dict[str, Any]:
    """检查单个包的状态。pkg_key 是 IMPORT_NAMES 的键。"""
    import_name = IMPORT_NAMES.get(pkg_key, pkg_key)
    installed, version, error = try_import_module(import_name)
    result: Dict[str, Any] = {
        "package": pkg_key,
        "import_name": import_name,
        "pip_name": PIP_NAMES.get(pkg_key, pkg_key),
        "installed": installed,
    }
    if installed:
        if version is not None:
            result["version"] = version
    else:
        if error:
            result["error"] = error
    return result


def check_required_and_optional(
    required: Optional[List[str]] = None,
    optional: Optional[List[str]] = None,
    skip_torch: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """批量检查核心依赖和可选依赖。"""
    result: Dict[str, Dict[str, Any]] = {}
    check_list = list(required or CORE_REQUIRED)
    if not skip_torch:
        check_list.append("torch")
    for pkg in check_list:
        result[pkg] = check_package(pkg)
        result[pkg]["optional"] = False
    for pkg in (optional or CORE_OPTIONAL):
        result[pkg] = check_package(pkg)
        result[pkg]["optional"] = True
    return result


def _parse_requirements_file(path: str) -> List[Dict[str, str]]:
    """
    解析 requirements_versions.txt，返回 [{name, version, raw}] 列表。
    仅处理最常见的 'pkg==version' 形式，忽略注释和空行。
    """
    entries: List[Dict[str, str]] = []
    if not os.path.isfile(path):
        return entries
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:==\s*([^#\s]+))?", line)
                if not m:
                    continue
                name = m.group(1)
                version = m.group(2) or ""
                entries.append({"name": name, "version": version, "raw": line})
    except Exception:
        return entries
    return entries


def _requirement_to_import_name(req_name: str) -> Optional[str]:
    """
    根据 requirements 中的 pip 包名推断 Python import 名。
    常见映射不完整时返回 None，调用方应视为无法自动检测。
    """
    mapping = {
        "numpy": "numpy",
        "pillow": "PIL",
        "Pillow": "PIL",
        "gradio": "gradio",
        "opencv-python": "cv2",
        "opencv-contrib-python": "cv2",
        "opencv-contrib-python-headless": "cv2",
        "torch": "torch",
        "trimesh": "trimesh",
        "rembg": "rembg",
        "pyyaml": "yaml",
        "transformers": "transformers",
        "safetensors": "safetensors",
        "accelerate": "accelerate",
        "einops": "einops",
        "timm": "timm",
    }
    if req_name in mapping:
        return mapping[req_name]
    simple = req_name.replace("-", "_").replace(".", "_").lower()
    return simple


def check_requirements_file(
    requirements_file: str,
) -> Dict[str, Any]:
    """
    检查 requirements 文件中列出来的包是否都可 import。
    返回 {'found': bool, 'path': str, 'entries': [...], 'missing': [...], 'total_missing': int}。
    """
    if not requirements_file or not os.path.isfile(requirements_file):
        return {
            "found": False,
            "path": requirements_file,
            "entries": [],
            "missing": [],
            "total_missing": 0,
        }
    entries = _parse_requirements_file(requirements_file)
    missing: List[str] = []
    for ent in entries:
        import_name = _requirement_to_import_name(ent["name"])
        if import_name is None:
            continue
        try:
            __import__(import_name)
        except Exception:
            missing.append(ent["name"])
    return {
        "found": True,
        "path": requirements_file,
        "count": len(entries),
        "entries": entries,
        "missing": sorted(set(missing)),
        "total_missing": len(set(missing)),
    }


def run_environment_inspection(
    project_root: Optional[str] = None,
    skip_torch: bool = False,
    requirements_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    一次性执行完整的环境检查。返回纯字典，适合后续 CLI 格式化或 launch 前置流程直接消费。

    参数：
      project_root:     项目根目录，用于自动定位 requirements_versions.txt
      skip_torch:       跳过 PyTorch import（在仅做静态检查时有用）
      requirements_file: 显式指定 requirements 文件路径
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    python_info = check_python_compatibility()
    platform_info = get_platform_info()
    deps = check_required_and_optional(skip_torch=skip_torch)

    if requirements_file is None:
        requirements_file = os.path.join(project_root, "requirements_versions.txt")
    req_info = check_requirements_file(requirements_file)

    return {
        "python_version": python_info["version"],
        "python": python_info,
        "platform": platform_info,
        "dependencies": deps,
        "requirements_file": req_info,
    }


# ---------------------------------------------------------------------------
# 与 launch_util 行为兼容的底层检查（find_spec 方式 + packaging 版本比较）
# ---------------------------------------------------------------------------

def is_package_installed(package_name: str) -> bool:
    """
    使用 importlib.util.find_spec 检测包是否可导入。
    与 modules.launch_util.is_installed 行为完全一致，速度快但不执行模块初始化。
    """
    try:
        spec = importlib.util.find_spec(package_name)
    except ModuleNotFoundError:
        return False
    return spec is not None


def _get_installed_version(package_name: str) -> Optional[str]:
    """
    通过 importlib.metadata 获取已安装包的版本号。
    返回 None 表示未安装或无法获取版本。
    """
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def check_requirements_strict(
    requirements_file: str,
) -> Dict[str, Any]:
    """
    使用 packaging 库严格检查 requirements 文件中的每个包是否安装且版本符合 specifier。
    行为与 modules.launch_util.requirements_met 一致。

    返回：
      {
        'found': bool,
        'path': str,
        'all_met': bool,
        'checked_count': int,
        'issues': [{'line': str, 'package': str, 'status': 'ok|missing|version_mismatch|error', 'detail': str}],
      }
    """
    if not requirements_file or not os.path.isfile(requirements_file):
        return {
            "found": False,
            "path": requirements_file,
            "all_met": False,
            "checked_count": 0,
            "issues": [],
        }

    # 惰性导入 packaging，避免在没有 packaging 的环境里报错
    try:
        from packaging.requirements import Requirement
        from packaging.version import parse as parse_version
    except Exception:
        # 如果 packaging 不可用，fallback 到简单检查
        simple = check_requirements_file(requirements_file)
        return {
            "found": True,
            "path": requirements_file,
            "all_met": simple["total_missing"] == 0,
            "checked_count": simple["count"],
            "issues": [
                {"line": m, "package": m, "status": "missing", "detail": "package not importable"}
                for m in simple["missing"]
            ],
            "fallback_reason": "packaging library not available; using simple import check",
        }

    issues: List[Dict[str, Any]] = []
    all_met = True
    checked = 0

    try:
        with open(requirements_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return {
            "found": True,
            "path": requirements_file,
            "all_met": False,
            "checked_count": 0,
            "issues": [{"line": "", "package": "", "status": "error", "detail": "could not read file"}],
        }

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except Exception as e:
            issues.append({"line": line, "package": "", "status": "error", "detail": f"could not parse: {e}"})
            all_met = False
            continue

        package = requirement.name
        checked += 1
        try:
            installed_version_str = _get_installed_version(package)
        except Exception as e:
            issues.append({"line": line, "package": package, "status": "error", "detail": str(e)})
            all_met = False
            continue

        if installed_version_str is None:
            issues.append({"line": line, "package": package, "status": "missing", "detail": "not installed"})
            all_met = False
            continue

        try:
            installed_version = parse_version(installed_version_str)
        except Exception as e:
            issues.append({"line": line, "package": package, "status": "error",
                           "detail": f"could not parse installed version {installed_version_str}: {e}"})
            all_met = False
            continue

        if installed_version not in requirement.specifier:
            issues.append({
                "line": line,
                "package": package,
                "status": "version_mismatch",
                "detail": f"installed {installed_version_str} does not satisfy {requirement.specifier}",
                "installed_version": installed_version_str,
                "required_spec": str(requirement.specifier),
            })
            all_met = False
        else:
            issues.append({
                "line": line,
                "package": package,
                "status": "ok",
                "detail": f"{installed_version_str} satisfies {requirement.specifier}",
                "installed_version": installed_version_str,
                "required_spec": str(requirement.specifier),
            })

    return {
        "found": True,
        "path": requirements_file,
        "all_met": all_met,
        "checked_count": checked,
        "issues": issues,
    }
