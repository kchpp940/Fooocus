"""
统一的 preflight 检查编排 service。

负责组合调用 environment_inspector / config_inspector / resource_scanner
并对结果做统一的严重程度分级：

  - BLOCKING (fatal)：必须中断启动的严重问题
    • Python 版本不满足最低要求
    • 核心依赖（非 optional）缺失
    • 配置文件 JSON 语法错误 / 不是合法 JSON 对象
    • schema 验证错误（如类型不匹配、非法值）
    • —strict 模式下：所有 warning 升级为 blocking

  - WARNING：不致命但应该让用户知道的问题
    • Python 版本低于推荐值
    • 可选依赖缺失
    • 路径目录不存在（但 launch 通常会自动创建）
    • schema 未知 key（拼写错或废弃配置）
    • 模型文件为 0 个
    • requirements 文件缺失 / 部分包未满足

  - INFO：正常信息
    • 版本号、平台、已安装的依赖
    • 存在的目录、加载的预设数量

返回纯数据结构，由调用方（CLI / launch.py）自行决定如何格式化输出和处理退出码。
"""

from __future__ import annotations

import os
from enum import IntEnum
from typing import Any, Dict, List, Optional


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    BLOCKING = 2


class PreflightCheck:
    """单个 preflight 检查项。"""

    def __init__(
        self,
        name: str,
        severity: Severity,
        message: str,
        category: str = "general",
        detail: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        self.name = name
        self.severity = severity
        self.message = message
        self.category = category
        self.detail = detail or {}
        self.suggestion = suggestion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.name.lower(),
            "severity_level": int(self.severity),
            "message": self.message,
            "category": self.category,
            "detail": self.detail,
            "suggestion": self.suggestion,
        }


class PreflightResult:
    """preflight 检查总结果。"""

    def __init__(self):
        self.checks: List[PreflightCheck] = []
        self.data: Dict[str, Any] = {}

    def add(self, check: PreflightCheck) -> None:
        self.checks.append(check)

    @property
    def max_severity(self) -> Severity:
        if not self.checks:
            return Severity.INFO
        return max(c.severity for c in self.checks)

    @property
    def blocking_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.BLOCKING)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.INFO)

    def promote_warnings_to_blocking(self) -> None:
        """strict 模式下将所有 warning 升级为 blocking。"""
        for c in self.checks:
            if c.severity == Severity.WARNING:
                c.severity = Severity.BLOCKING

    def to_dict(self) -> Dict[str, Any]:
        max_sev = self.max_severity
        return {
            "checks": [c.to_dict() for c in self.checks],
            "summary": {
                "blocking": self.blocking_count,
                "warning": self.warning_count,
                "info": self.info_count,
                "max_severity": max_sev.name.lower(),
                "max_severity_level": int(max_sev),
            },
            "data": self.data,
            "blocking": self.blocking_count > 0,
            "exit_code": 2 if self.blocking_count > 0 else (1 if self.warning_count > 0 else 0),
        }


# ---------------------------------------------------------------------------
# 分类器：根据各 service 的输出生成 PreflightCheck 列表
# ---------------------------------------------------------------------------

def _classify_environment(env_info: Dict[str, Any], result: PreflightResult) -> None:
    py = env_info.get("python", {})
    if not py.get("compatible", False):
        result.add(PreflightCheck(
            name="python_version_incompatible",
            severity=Severity.BLOCKING,
            category="environment",
            message=f"Python version {py.get('version', '?')} is below minimum required {py.get('minimum_required', '?')}",
            detail={"version": py.get("version"), "required": py.get("minimum_required")},
            suggestion="Upgrade Python to meet the minimum required version.",
        ))
    elif not py.get("recommended", False):
        result.add(PreflightCheck(
            name="python_version_below_recommended",
            severity=Severity.WARNING,
            category="environment",
            message=f"Python version {py.get('version', '?')} works but {py.get('minimum_recommended', '?')}+ is recommended",
            detail={"version": py.get("version"), "recommended": py.get("minimum_recommended")},
        ))
    else:
        result.add(PreflightCheck(
            name="python_version_ok",
            severity=Severity.INFO,
            category="environment",
            message=f"Python {py.get('version', '?')} is compatible",
            detail={"version": py.get("version")},
        ))

    plat = env_info.get("platform", {})
    result.add(PreflightCheck(
        name="platform",
        severity=Severity.INFO,
        category="environment",
        message=f"Running on {plat.get('system', '?')} ({plat.get('machine', '?')})",
        detail={"system": plat.get("system"), "machine": plat.get("machine")},
    ))

    deps = env_info.get("dependencies", {}) or {}
    for pkg_key, info in deps.items():
        if info.get("installed", False):
            version = info.get("version", "unknown")
            result.add(PreflightCheck(
                name=f"dep_{pkg_key.lower()}",
                severity=Severity.INFO,
                category="environment",
                message=f"{pkg_key} installed: {version}",
                detail={"package": pkg_key, "version": version},
            ))
        else:
            is_optional = info.get("optional", False)
            pip_name = info.get("pip_name", pkg_key)
            if is_optional:
                result.add(PreflightCheck(
                    name=f"dep_{pkg_key.lower()}_missing",
                    severity=Severity.WARNING,
                    category="environment",
                    message=f"Optional dependency '{pkg_key}' is not installed",
                    detail={"package": pkg_key, "pip_name": pip_name},
                    suggestion=f"Install for full features: pip install {pip_name}",
                ))
            else:
                result.add(PreflightCheck(
                    name=f"dep_{pkg_key.lower()}_missing",
                    severity=Severity.BLOCKING,
                    category="environment",
                    message=f"Required dependency '{pkg_key}' is not installed",
                    detail={"package": pkg_key, "pip_name": pip_name},
                    suggestion=f"Run: pip install {pip_name}",
                ))

    req = env_info.get("requirements_file", {}) or {}
    if not req.get("found", False):
        result.add(PreflightCheck(
            name="requirements_file_missing",
            severity=Severity.WARNING,
            category="environment",
            message="Could not locate requirements_versions.txt to cross-check",
            detail={"path": req.get("path")},
        ))
    else:
        result.add(PreflightCheck(
            name="requirements_file_loaded",
            severity=Severity.INFO,
            category="environment",
            message=f"Found {req.get('count', 0)} requirements in {os.path.basename(req.get('path', ''))}",
            detail={"path": req.get("path"), "count": req.get("count")},
        ))
        missing = req.get("missing", []) or []
        if missing:
            result.add(PreflightCheck(
                name="requirements_missing_packages",
                severity=Severity.WARNING,
                category="environment",
                message=f"{len(missing)} package(s) from requirements_versions.txt could not be imported",
                detail={"missing": missing[:20], "total_missing": len(missing)},
                suggestion=f"Run: pip install -r {req.get('path', 'requirements_versions.txt')}",
            ))


def _classify_config(cfg_info: Dict[str, Any], result: PreflightResult) -> None:
    cfg_load = cfg_info.get("config", {}) or {}
    paths = cfg_info.get("paths", {}) or {}
    schema = cfg_info.get("schema", {}) or {}

    if not cfg_load.get("loaded", False):
        err = cfg_load.get("error", "")
        if "parse_error" in str(err):
            result.add(PreflightCheck(
                name="config_json_parse_error",
                severity=Severity.BLOCKING,
                category="config",
                message=f"Config file has JSON syntax error: {err}",
                detail={"path": cfg_load.get("path"), "error": err},
                suggestion="Fix JSON syntax (check for trailing commas and unescaped backslashes).",
            ))
        else:
            result.add(PreflightCheck(
                name="config_file_not_found",
                severity=Severity.WARNING,
                category="config",
                message=f"Config file not found at {cfg_load.get('path', '?')} (will use defaults)",
                detail={"path": cfg_load.get("path")},
                suggestion="Create config.txt if you need custom settings.",
            ))
    else:
        result.add(PreflightCheck(
            name="config_file_loaded",
            severity=Severity.INFO,
            category="config",
            message=f"Loaded {cfg_load.get('keys_count', 0)} keys from {os.path.basename(cfg_load.get('path', ''))}",
            detail={"path": cfg_load.get("path"), "keys_count": cfg_load.get("keys_count")},
        ))

    # 路径检查
    by_key = paths.get("by_key", {}) or {}
    missing_dir_categories = 0
    existing_dir_categories = 0
    for key, info in by_key.items():
        for folder in info.get("effective", []):
            if os.path.isdir(folder):
                result.add(PreflightCheck(
                    name=f"config_{key}_exists",
                    severity=Severity.INFO,
                    category="config.paths",
                    message=f"'{key}' directory exists: {folder}",
                    detail={"path": folder, "key": key, "source": info.get("source", "default")},
                ))
                existing_dir_categories += 1
            else:
                result.add(PreflightCheck(
                    name=f"config_{key}_missing",
                    severity=Severity.WARNING,
                    category="config.paths",
                    message=f"'{key}' directory does not exist: {folder}",
                    detail={"path": folder, "key": key, "source": info.get("source", "default")},
                    suggestion="Directory will be auto-created as needed, or create it manually.",
                ))
                missing_dir_categories += 1

    # temp_path 和 user_data
    for special in ("temp_path", "path_user_data"):
        sp = paths.get(special)
        if sp is None:
            continue
        folder = sp.get("effective")
        if folder is None:
            continue
        if os.path.isdir(folder):
            result.add(PreflightCheck(
                name=f"config_{special}_exists",
                severity=Severity.INFO,
                category="config.paths",
                message=f"'{special}' directory exists: {folder}",
                detail={"path": folder},
            ))
        else:
            result.add(PreflightCheck(
                name=f"config_{special}_missing",
                severity=Severity.WARNING,
                category="config.paths",
                message=f"'{special}' directory does not exist: {folder}",
                detail={"path": folder},
                suggestion="Will be created on first use.",
            ))

    # schema 检查
    if not schema.get("available", False):
        result.add(PreflightCheck(
            name="config_schema_unavailable",
            severity=Severity.WARNING,
            category="config.schema",
            message="config_schema module could not be loaded for schema validation",
        ))
    else:
        issues = schema.get("issues", []) or []
        errors = [i for i in issues if i.get("severity") == "error"]
        warnings = [i for i in issues if i.get("severity") == "warning"]
        if not issues:
            result.add(PreflightCheck(
                name="config_schema_ok",
                severity=Severity.INFO,
                category="config.schema",
                message="All configured keys pass schema validation",
            ))
        for issue in errors:
            result.add(PreflightCheck(
                name=f"config_schema_error_{issue.get('key', '?')}",
                severity=Severity.BLOCKING,
                category="config.schema",
                message=f"Schema validation error: {issue.get('key', '?')} — {issue.get('message', '')}",
                detail=issue.get("detail") or {"key": issue.get("key")},
                suggestion="Fix the value to match the expected type/range.",
            ))
        for issue in warnings:
            result.add(PreflightCheck(
                name=f"config_schema_warning_{issue.get('key', '?')}",
                severity=Severity.WARNING,
                category="config.schema",
                message=f"Schema warning: {issue.get('key', '?')} — {issue.get('message', '')}",
                detail=issue.get("detail") or {"key": issue.get("key")},
                suggestion="Check if the key is misspelled or obsolete.",
            ))


def _classify_resources(res_info: Dict[str, Any], result: PreflightResult) -> None:
    models = res_info.get("models", {}) or {}
    total = models.get("total_count", 0)
    by_cat = models.get("by_category", {}) or {}

    for category, info in by_cat.items():
        count = info.get("count", 0)
        for folder, files in (info.get("files_by_folder") or {}).items():
            result.add(PreflightCheck(
                name=f"resource_{category}",
                severity=Severity.INFO,
                category="resources",
                message=f"Scanned {folder}: {len(files)} file(s)",
                detail={"folder": folder, "count": len(files), "category": category},
            ))

    if total == 0:
        result.add(PreflightCheck(
            name="resources_no_models",
            severity=Severity.WARNING,
            category="resources",
            message="No model files were found in any category",
            suggestion="Add model files to models/ subdirectories or configure custom paths.",
        ))
    else:
        result.add(PreflightCheck(
            name="resources_models_summary",
            severity=Severity.INFO,
            category="resources",
            message=f"Found {total} model file(s) across {len(by_cat)} categories",
            detail={"total_count": total, "categories": len(by_cat)},
        ))

    presets = res_info.get("presets")
    if presets is not None:
        result.add(PreflightCheck(
            name="resources_presets",
            severity=Severity.INFO,
            category="resources",
            message=f"Found {presets.get('builtin_count', 0)} built-in + {presets.get('user_count', 0)} user preset(s)",
            detail={"builtin": presets.get("builtin_count"), "user": presets.get("user_count")},
        ))

    wildcards = res_info.get("wildcards")
    if wildcards is not None:
        result.add(PreflightCheck(
            name="resources_wildcards",
            severity=Severity.INFO,
            category="resources",
            message=f"Found {wildcards.get('count', 0)} wildcard file(s)",
            detail={"count": wildcards.get("count")},
        ))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_preflight(
    project_root: Optional[str] = None,
    check_env: bool = True,
    check_config: bool = True,
    check_resources: bool = False,
    strict: bool = False,
    skip_torch: bool = False,
    requirements_file: Optional[str] = None,
    config_path: Optional[str] = None,
    models_only: bool = False,
) -> PreflightResult:
    """
    运行 preflight 检查并返回分级结果。

    参数：
      project_root:       项目根目录（自动检测）
      check_env:          是否检查环境
      check_config:       是否检查配置
      check_resources:    是否扫描资源（默认关闭，因为较慢）
      strict:             严格模式，将 warning 升级为 blocking
      skip_torch:         环境检查中跳过 torch
      requirements_file:  显式指定 requirements 文件
      config_path:        显式指定 config.txt 路径
      models_only:        资源扫描只扫模型，不扫 preset/wildcard

    返回：PreflightResult 对象，包含所有检查项和严重等级。
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    result = PreflightResult()

    if check_env:
        from modules.services.environment_inspector import run_environment_inspection
        env_info = run_environment_inspection(
            project_root=project_root,
            skip_torch=skip_torch,
            requirements_file=requirements_file,
        )
        _classify_environment(env_info, result)
        result.data["environment"] = env_info

    if check_config:
        from modules.services.config_inspector import load_and_validate_config
        cfg_info = load_and_validate_config(
            config_path=config_path,
            project_root=project_root,
            fix_missing_dirs=False,
        )
        _classify_config(cfg_info, result)
        result.data["config"] = cfg_info

    if check_resources:
        from modules.services.resource_scanner import run_resource_scan
        res_info = run_resource_scan(
            scan_models=True,
            include_presets=not models_only,
            include_wildcards=not models_only,
            project_root=project_root,
        )
        _classify_resources(res_info, result)
        result.data["resources"] = res_info

    if strict:
        result.promote_warnings_to_blocking()

    return result
