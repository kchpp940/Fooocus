"""
无副作用的配置检查 / 路径解析 service 层。

忠实复现 modules/config.py 中的真实默认路径规则，但：
- 不 import args_manager（不触发全局 parse_args）
- 不调用 os.makedirs（只读检查）
- 不访问 modules.config 的全局 mutable 状态
- 不做 sys.exit / 格式化输出

所有返回值都是纯 dict/list，适合 launch 启动流程和 CLI 工具共用。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple


USER_PRESET_PREFIX = "[User] "

# 和 modules/config.py 中 get_dir_or_set_default 的默认值一一对应
# key -> (default_relative_path, is_array)
# default_relative_path 是相对于 modules/ 目录的，和 modules/config.py 保持一致
DEFAULT_PATH_SPEC: Dict[str, Tuple[str, bool]] = {
    "path_checkpoints": ("../models/checkpoints/", True),
    "path_loras": ("../models/loras/", True),
    "path_embeddings": ("../models/embeddings/", False),
    "path_vae_approx": ("../models/vae_approx/", False),
    "path_vae": ("../models/vae/", False),
    "path_upscale_models": ("../models/upscale_models/", False),
    "path_inpaint": ("../models/inpaint/", False),
    "path_controlnet": ("../models/controlnet/", False),
    "path_clip_vision": ("../models/clip_vision/", False),
    "path_fooocus_expansion": ("../models/prompt_expansion/fooocus_expansion", False),
    "path_wildcards": ("../wildcards/", False),
    "path_safety_checker": ("../models/safety_checker/", False),
    "path_sam": ("../models/sam/", False),
    "path_outputs": ("../outputs/", False),
}


def resolve_project_root() -> str:
    """返回项目根目录（Fooocus/），与 modules/config.py 的 _root_dir 保持一致。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_default_temp_path() -> str:
    return os.path.join(tempfile.gettempdir(), "fooocus")


def get_user_data_dir(project_root: Optional[str] = None) -> str:
    """
    与 modules/config.py:get_user_data_dir() 语义一致，但不 makedirs。
    优先级：FOOOCUS_USER_DATA_DIR -> config_path env 所在目录 -> ~/.fooocus
    """
    if project_root is None:
        project_root = resolve_project_root()

    env_dir = os.getenv("FOOOCUS_USER_DATA_DIR")
    if env_dir:
        return os.path.abspath(env_dir)

    config_path_val = os.getenv("config_path")
    if config_path_val:
        return os.path.dirname(os.path.abspath(config_path_val))

    home = os.path.expanduser("~")
    return os.path.join(home, ".fooocus")


def get_user_presets_dir(project_root: Optional[str] = None) -> str:
    return os.path.join(get_user_data_dir(project_root), "user_presets")


def get_builtin_presets_dir(project_root: Optional[str] = None) -> str:
    if project_root is None:
        project_root = resolve_project_root()
    return os.path.join(project_root, "presets")


def resolve_default_path(
    key: str,
    project_root: Optional[str] = None,
) -> List[str]:
    """
    仅根据默认规则计算 key 对应的绝对路径列表，不读取 config.txt / env。
    返回 list 与 modules/config 中 as_array=True 的行为保持一致。
    """
    if project_root is None:
        project_root = resolve_project_root()
    if key not in DEFAULT_PATH_SPEC:
        return []
    rel, is_array = DEFAULT_PATH_SPEC[key]
    # modules/config.py 里 default_value 的相对路径是相对 modules/ 的
    modules_dir = os.path.join(project_root, "modules")
    abs_path = os.path.abspath(os.path.join(modules_dir, rel))
    return [abs_path] if is_array else [abs_path]


def _normalize_path_value(raw: Any, project_root: str) -> List[str]:
    """把 config.txt 中某个 path_* key 的值标准化为绝对路径列表。"""
    if isinstance(raw, list):
        result = []
        for v in raw:
            if isinstance(v, str) and v:
                result.append(os.path.abspath(v) if os.path.isabs(v) else os.path.abspath(os.path.join(project_root, v)))
        return result
    if isinstance(raw, str) and raw:
        abs_path = os.path.abspath(raw) if os.path.isabs(raw) else os.path.abspath(os.path.join(project_root, raw))
        return [abs_path]
    return []


def load_config_overrides(
    config_path: Optional[str] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    读取 config.txt（若存在），返回纯 dict。
    与 modules/config.py 不同：不合并 default preset、不处理废弃的 user_path_config.txt。
    只返回用户在 config.txt 中显式写的值。

    返回 {'loaded': bool, 'path': str, 'data': dict, 'error': str or None, 'keys_count': int}
    """
    if project_root is None:
        project_root = resolve_project_root()
    if config_path is None:
        config_path = os.environ.get("config_path") or os.path.join(project_root, "config.txt")
    config_path = os.path.abspath(config_path)

    result: Dict[str, Any] = {
        "loaded": False,
        "path": config_path,
        "data": {},
        "error": None,
        "keys_count": 0,
    }
    if not os.path.isfile(config_path):
        result["error"] = "file_not_found"
        return result
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            result["error"] = "not_a_json_object"
            return result
        result["loaded"] = True
        result["data"] = data
        result["keys_count"] = len(data)
    except Exception as e:
        result["error"] = f"parse_error: {type(e).__name__}: {e}"
    return result


def resolve_effective_paths(
    config_overrides: Optional[Dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    综合：默认规则 -> config.txt 用户覆盖 -> env 覆盖，得到每个 path_* key 的有效路径列表。

    与 modules/config.py:get_dir_or_set_default 的求值顺序一致，但不做自动 fallback / makedirs。

    返回：
      {
        "by_key": { "path_checkpoints": {
            "effective": ["/abs/path1", "/abs/path2"],
            "source": "default|config|env",
            "exists_all": bool,
            "exists_any": bool,
        }, ... },
        "all_keys": [...],
      }
    """
    if project_root is None:
        project_root = resolve_project_root()
    if config_overrides is None:
        config_overrides = {}

    by_key: Dict[str, Any] = {}
    for key, (_, is_array) in DEFAULT_PATH_SPEC.items():
        # 1. 环境变量最高优先级（和 modules/config.py 的 get_dir_or_set_default 一致）
        from_env = os.getenv(key)
        if from_env:
            eff = _normalize_path_value(from_env, project_root)
            source = "env"
        elif key in config_overrides:
            eff = _normalize_path_value(config_overrides[key], project_root)
            source = "config"
        else:
            eff = resolve_default_path(key, project_root)
            source = "default"

        exists = [p for p in eff if os.path.isdir(p)]
        by_key[key] = {
            "effective": eff,
            "is_array": is_array,
            "source": source,
            "exists_all": bool(eff) and len(exists) == len(eff),
            "exists_any": len(exists) > 0,
            "missing": [p for p in eff if not os.path.isdir(p)],
        }

    # 特殊：temp_path 和 user_data / path_user_data
    temp_from_env = os.getenv("temp_path")
    if temp_from_env and isinstance(temp_from_env, str):
        temp_eff = os.path.abspath(temp_from_env) if os.path.isabs(temp_from_env) else os.path.abspath(os.path.join(project_root, temp_from_env))
        temp_src = "env"
    elif "temp_path" in config_overrides and isinstance(config_overrides["temp_path"], str):
        t = config_overrides["temp_path"]
        temp_eff = os.path.abspath(t) if os.path.isabs(t) else os.path.abspath(os.path.join(project_root, t))
        temp_src = "config"
    else:
        temp_eff = get_default_temp_path()
        temp_src = "default"

    user_data_eff = get_user_data_dir(project_root)

    return {
        "by_key": by_key,
        "temp_path": {
            "effective": temp_eff,
            "source": temp_src,
            "exists": os.path.isdir(temp_eff),
        },
        "path_user_data": {
            "effective": user_data_eff,
            "exists": os.path.isdir(user_data_eff),
        },
        "project_root": project_root,
    }


def list_builtin_presets(project_root: Optional[str] = None) -> List[str]:
    d = get_builtin_presets_dir(project_root)
    if not os.path.isdir(d):
        return []
    return sorted([f[:-5] for f in os.listdir(d) if f.endswith(".json")])


def list_user_presets(project_root: Optional[str] = None) -> List[str]:
    d = get_user_presets_dir(project_root)
    if not os.path.isdir(d):
        return []
    return sorted([USER_PRESET_PREFIX + f[:-5] for f in os.listdir(d) if f.endswith(".json")])


def list_all_presets(project_root: Optional[str] = None) -> List[str]:
    return ["initial"] + list_builtin_presets(project_root) + list_user_presets(project_root)


def is_user_preset(preset_name: str) -> bool:
    return isinstance(preset_name, str) and preset_name.startswith(USER_PRESET_PREFIX)


def strip_user_prefix(preset_name: str) -> str:
    return preset_name[len(USER_PRESET_PREFIX):] if is_user_preset(preset_name) else preset_name


def load_preset_content(preset_name: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(preset_name, str):
        return {}
    if preset_name == "initial":
        return {}
    if is_user_preset(preset_name):
        short = strip_user_prefix(preset_name)
        path = os.path.join(get_user_presets_dir(project_root), f"{short}.json")
    else:
        path = os.path.join(get_builtin_presets_dir(project_root), f"{preset_name}.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_config_schema():
    """惰性加载 config_schema，不 import modules.config。"""
    try:
        from modules.config_schema import create_default_schema
        return create_default_schema()
    except Exception:
        return None


def validate_config_against_schema(
    config_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    使用 modules/config_schema.py 中同样的 ConfigSchema 验证给定配置 dict。
    返回 {'available': bool, 'issues': [...], 'issues_count': int}。
    issues 中的每一项：{'key', 'severity': 'error|warning', 'message', 'detail'}
    """
    schema = load_config_schema()
    if schema is None:
        return {"available": False, "issues": [], "issues_count": 0}

    issues: List[Dict[str, Any]] = []
    for key, item in schema.items.items():
        if item.validator is None or key not in config_data:
            continue
        val = config_data[key]
        try:
            valid = bool(item.validator(val))
        except Exception:
            valid = False
        if not valid:
            issues.append({
                "key": key,
                "severity": "error",
                "message": f"Value for '{key}' failed schema validation",
                "detail": {"value": str(val)[:200] if val is not None else None},
            })
    for key in config_data:
        if not schema.has(key) and schema.resolve_alias(key) is None:
            issues.append({
                "key": key,
                "severity": "warning",
                "message": f"Unknown config key '{key}' (not in schema)",
                "detail": None,
            })
    return {"available": True, "issues": issues, "issues_count": len(issues)}


def load_and_validate_config(
    config_path: Optional[str] = None,
    project_root: Optional[str] = None,
    fix_missing_dirs: bool = False,
) -> Dict[str, Any]:
    """
    一站式：读取 config.txt -> 解析有效路径 -> 验证 schema。
    fix_missing_dirs=True 时会自动创建缺失的 path_* 目录（但不创建其它目录）。
    返回合并的大 dict，供启动流程或 CLI 分别消费。
    """
    if project_root is None:
        project_root = resolve_project_root()

    load_result = load_config_overrides(config_path, project_root)
    cfg_data = load_result["data"] if load_result["loaded"] else {}
    paths = resolve_effective_paths(cfg_data, project_root)

    created: List[str] = []
    if fix_missing_dirs:
        for key, info in paths["by_key"].items():
            for p in info.get("missing", []):
                try:
                    os.makedirs(p, exist_ok=True)
                    created.append(p)
                except Exception:
                    pass
        tp = paths["temp_path"]["effective"]
        if not paths["temp_path"]["exists"]:
            try:
                os.makedirs(tp, exist_ok=True)
                created.append(tp)
            except Exception:
                pass
        ud = paths["path_user_data"]["effective"]
        if not paths["path_user_data"]["exists"]:
            try:
                os.makedirs(ud, exist_ok=True)
                created.append(ud)
            except Exception:
                pass

    schema_result = validate_config_against_schema(cfg_data)

    return {
        "project_root": project_root,
        "config": load_result,
        "paths": paths,
        "schema": schema_result,
        "created_dirs": created,
    }
