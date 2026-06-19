"""
无副作用的资源扫描 service 层。

和 modules/config_inspector 配合，使用 resolve_effective_paths 得到模型目录列表，
然后按照 modules/config.py + modules/extra_utils.py 中 get_files_from_folder 的真实规则扫描文件。

只返回纯数据结构，不修改 modules.config 的全局状态，不触发 parse_args。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from modules.services.config_inspector import (
    resolve_effective_paths,
    load_config_overrides,
    list_builtin_presets,
    list_user_presets,
    resolve_project_root,
)


DEFAULT_MODEL_EXTS = [".pth", ".ckpt", ".bin", ".safetensors", ".fooocus.patch"]
WILDCARD_EXTS = [".txt"]


# 和 modules/extra_utils.py:get_files_from_folder 保持一致的排序 + 去重逻辑
def scan_directory(
    folder: str,
    extensions: Optional[List[str]] = None,
    name_filter: Optional[str] = None,
) -> List[str]:
    """
    递归扫描目录，返回匹配扩展名的文件相对路径（相对于 folder）。
    与 modules/extra_utils.py:get_files_from_folder 行为完全一致：
      - 递归遍历子目录 (os.walk)
      - 按大小写不敏感排序文件名
      - name_filter 匹配文件名（不含扩展名）是否包含子串
    空目录或不存在返回 []。
    """
    if not folder or not os.path.isdir(folder):
        return []
    if extensions is None:
        extensions = DEFAULT_MODEL_EXTS
    lower_exts = {ext.lower() for ext in extensions}
    found: List[str] = []

    try:
        for root, _, files in os.walk(folder, topdown=False):
            relative_dir = os.path.relpath(root, folder)
            if relative_dir == ".":
                relative_dir = ""
            for filename in sorted(files, key=lambda s: s.casefold()):
                _, file_ext = os.path.splitext(filename)
                ext_ok = not extensions or file_ext.lower() in lower_exts
                name_ok = name_filter is None or name_filter in os.path.splitext(filename)[0]
                if ext_ok and name_ok:
                    rel_path = os.path.join(relative_dir, filename) if relative_dir else filename
                    found.append(rel_path)
    except Exception:
        return []

    return found


def scan_model_directories(
    extensions: Optional[List[str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    扫描所有模型类别目录。返回：
      {
        "by_category": {
            "checkpoints": {
                "folders": ["/abs/path", ...],
                "files_by_folder": {"/abs/path": ["a.safetensors", ...]},
                "count": 10,
            },
            ...
        },
        "total_count": 42,
        "extensions": [...],
      }
    """
    if extensions is None:
        extensions = DEFAULT_MODEL_EXTS
    if project_root is None:
        project_root = resolve_project_root()
    if config_overrides is None:
        config_overrides = load_config_overrides(project_root=project_root)["data"]

    paths = resolve_effective_paths(config_overrides, project_root)

    # 类别 key 对应到 config_inspector 中的 path_* key
    category_to_path_key = {
        "checkpoints": "path_checkpoints",
        "loras": "path_loras",
        "embeddings": "path_embeddings",
        "vae_approx": "path_vae_approx",
        "vae": "path_vae",
        "upscale_models": "path_upscale_models",
        "inpaint": "path_inpaint",
        "controlnet": "path_controlnet",
        "clip_vision": "path_clip_vision",
        "safety_checker": "path_safety_checker",
        "sam": "path_sam",
    }

    by_category: Dict[str, Any] = {}
    total = 0
    for category, path_key in category_to_path_key.items():
        info = paths["by_key"].get(path_key)
        folders: List[str] = []
        files_by_folder: Dict[str, List[str]] = {}
        count = 0
        if info is not None:
            folders = list(info["effective"])
            for folder in folders:
                files = scan_directory(folder, extensions)
                files_by_folder[folder] = files
                count += len(files)
        by_category[category] = {
            "folders": folders,
            "files_by_folder": files_by_folder,
            "count": count,
        }
        total += count

    return {
        "by_category": by_category,
        "total_count": total,
        "extensions": list(extensions),
        "project_root": project_root,
    }


def scan_wildcards(
    extensions: Optional[List[str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    if extensions is None:
        extensions = WILDCARD_EXTS
    if project_root is None:
        project_root = resolve_project_root()
    if config_overrides is None:
        config_overrides = load_config_overrides(project_root=project_root)["data"]

    paths = resolve_effective_paths(config_overrides, project_root)
    info = paths["by_key"].get("path_wildcards")
    folders = list(info["effective"]) if info is not None else []
    files_by_folder: Dict[str, List[str]] = {}
    count = 0
    for folder in folders:
        files = scan_directory(folder, extensions)
        files_by_folder[folder] = files
        count += len(files)
    return {
        "folders": folders,
        "files_by_folder": files_by_folder,
        "count": count,
        "extensions": list(extensions),
    }


def scan_presets(project_root: Optional[str] = None) -> Dict[str, Any]:
    builtin = list_builtin_presets(project_root)
    user = list_user_presets(project_root)
    return {
        "builtin": builtin,
        "user": user,
        "all": ["initial"] + builtin + user,
        "builtin_count": len(builtin),
        "user_count": len(user),
        "total_count": 1 + len(builtin) + len(user),
    }


def run_resource_scan(
    scan_models: bool = True,
    include_presets: bool = True,
    include_wildcards: bool = True,
    model_extensions: Optional[List[str]] = None,
    wildcard_extensions: Optional[List[str]] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """一站式执行资源扫描，返回合并 dict。"""
    if project_root is None:
        project_root = resolve_project_root()
    config_overrides = load_config_overrides(project_root=project_root)["data"]

    result: Dict[str, Any] = {"project_root": project_root}
    if scan_models:
        result["models"] = scan_model_directories(
            extensions=model_extensions,
            config_overrides=config_overrides,
            project_root=project_root,
        )
    if include_presets:
        result["presets"] = scan_presets(project_root)
    if include_wildcards:
        result["wildcards"] = scan_wildcards(
            extensions=wildcard_extensions,
            config_overrides=config_overrides,
            project_root=project_root,
        )
    return result
