import os
import sys
import shutil
import platform
import importlib
import importlib.util
import importlib.metadata
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from packaging.version import parse as parse_version
from packaging.specifiers import SpecifierSet


class CheckStatus:
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


class CheckCategory:
    PYTHON = "Python"
    PYTORCH = "PyTorch / CUDA"
    DEPENDENCIES = "Dependencies"
    DIRECTORIES = "Directories"
    PERMISSIONS = "Permissions"
    CONFIG = "Configuration"
    DISK = "Disk"


class CheckMode:
    REPORT = "report"
    HEALTHCHECK = "healthcheck"
    STRICT = "strict"
    _ALL = (REPORT, HEALTHCHECK, STRICT)


@dataclass
class CheckResult:
    name: str
    category: str
    status: str
    current: str = "N/A"
    expected: str = "N/A"
    message: str = ""
    fix_suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "current": self.current,
            "expected": self.expected,
            "message": self.message,
            "fix_suggestion": self.fix_suggestion,
        }


@dataclass
class PreflightReport:
    results: List[CheckResult] = field(default_factory=list)

    def add_result(self, result: CheckResult):
        self.results.append(result)

    @property
    def passed(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == CheckStatus.PASS]

    @property
    def failed(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == CheckStatus.FAIL]

    @property
    def warnings(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == CheckStatus.WARNING]

    @property
    def has_errors(self) -> bool:
        return len(self.failed) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": len(self.results),
            "passed": len(self.passed),
            "failed": len(self.failed),
            "warnings": len(self.warnings),
            "has_errors": self.has_errors,
            "results": [r.to_dict() for r in self.results],
        }


def _mode_exit_code(mode: str, report: PreflightReport) -> int:
    """根据模式和检查结果计算退出码（不含异常情况）。"""
    if mode == CheckMode.REPORT:
        return 0
    if mode in (CheckMode.HEALTHCHECK, CheckMode.STRICT):
        return 1 if report.has_errors else 0
    return 0


_PREFLIGHT_POLICIES: Dict[str, "PreflightPolicy"] = {}


@dataclass(frozen=True)
class PreflightPolicy:
    """
    单一入口的预检策略。集中定义 mode / stage / 缓存键 / 退出码规则。

    所有调用方只传 policy 名称，由 policy 决定内部行为，
    避免不同入口散传 mode/stage/exit_on_error 导致行为不一致。
    """
    name: str
    mode: str
    stage: str
    description: str = ""

    @classmethod
    def register(cls, name: str, mode: str, stage: str, description: str = "") -> "PreflightPolicy":
        policy = cls(name=name, mode=mode, stage=stage, description=description)
        _PREFLIGHT_POLICIES[name] = policy
        return policy

    @classmethod
    def get(cls, name: str) -> "PreflightPolicy":
        if name not in _PREFLIGHT_POLICIES:
            raise ValueError(
                f"Unknown preflight policy: {name!r}. "
                f"Available: {sorted(_PREFLIGHT_POLICIES.keys())}"
            )
        return _PREFLIGHT_POLICIES[name]

    @classmethod
    def all_policies(cls) -> Dict[str, "PreflightPolicy"]:
        return dict(_PREFLIGHT_POLICIES)


# ===== 标准策略注册 =====
PreflightPolicy.register(
    "launch_early",
    mode=CheckMode.REPORT,
    stage="early",
    description="启动早期内嵌检查（非阻塞，只出报告）",
)
PreflightPolicy.register(
    "launch_post_install",
    mode=CheckMode.REPORT,
    stage="post-install",
    description="依赖安装完成后的验证（非阻塞，只出报告）",
)
PreflightPolicy.register(
    "pre_update",
    mode=CheckMode.REPORT,
    stage="pre-update",
    description="git 更新前的预检（非阻塞，只出报告）",
)
PreflightPolicy.register(
    "cli_default",
    mode=CheckMode.STRICT,
    stage="preflight-only",
    description="独立 CLI 默认策略（严格模式，有 FAIL 就退出）",
)
PreflightPolicy.register(
    "healthcheck",
    mode=CheckMode.HEALTHCHECK,
    stage="healthcheck",
    description="Docker HEALTHCHECK 专用（FAIL 才不健康，WARNING 仍算健康）",
)
PreflightPolicy.register(
    "ci",
    mode=CheckMode.STRICT,
    stage="ci",
    description="CI pipeline 专用（严格模式 + JSON 输出）",
)
PreflightPolicy.register(
    "report",
    mode=CheckMode.REPORT,
    stage="report",
    description="通用报告模式（永远不阻塞）",
)
PreflightPolicy.register(
    "strict",
    mode=CheckMode.STRICT,
    stage="strict",
    description="通用严格模式（有 FAIL 就非零退出）",
)


def _resolve_policy(policy_name, mode, stage, exit_on_error_deprecated):
    """
    根据传入参数解析最终 policy。优先级：
    1. 显式 policy= 名称（最推荐）
    2. 显式 mode=（次选，stage 可选）
    3. exit_on_error=True → 等价于 strict 模式（废弃）
    4. 都没有 → 默认为 report 模式（内嵌安全默认值）
    """
    import warnings

    if policy_name is not None:
        return PreflightPolicy.get(policy_name)

    # 旧参数兼容：exit_on_error → strict
    if exit_on_error_deprecated:
        warnings.warn(
            "'exit_on_error' 参数已废弃，请使用 policy='strict' 或 mode='strict'。",
            DeprecationWarning,
            stacklevel=3
        )
        mode = mode or CheckMode.STRICT

    resolved_mode = mode if mode is not None else CheckMode.REPORT
    resolved_stage = stage if stage is not None else resolved_mode

    # 用 mode+stage 构造一个动态 policy（无需注册）
    return PreflightPolicy(
        name=f"dynamic:{resolved_mode}/{resolved_stage}",
        mode=resolved_mode,
        stage=resolved_stage,
        description="由 mode/stage 动态合成的策略",
    )


def _get_package_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_package_installed(package_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _check_write_permission(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".fooocus_write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except (OSError, PermissionError):
        return False


def _get_free_disk_space(path: str) -> int:
    try:
        disk_usage = shutil.disk_usage(path)
        return disk_usage.free
    except (OSError, PermissionError):
        return -1


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 0:
        return "Unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


class PreflightChecker:
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = root_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.report = PreflightReport()

    def run_all(self) -> PreflightReport:
        self.report = PreflightReport()
        self.check_python_version()
        self.check_pytorch_and_cuda()
        self.check_xformers()
        self.check_gradio()
        self.check_key_dependencies()
        self.check_model_directories()
        self.check_output_directory()
        self.check_temp_directory()
        self.check_config_file()
        self.check_disk_space()
        return self.report

    def check_python_version(self):
        current_version = sys.version.split()[0]
        current_info = sys.version_info

        min_version = (3, 10)
        max_version = (3, 12)
        expected_str = f">= {min_version[0]}.{min_version[1]}, < {max_version[0]}.{max_version[1]}"

        if min_version <= current_info < max_version:
            status = CheckStatus.PASS
            message = "Python version is compatible"
            fix_suggestion = ""
        elif current_info < min_version:
            status = CheckStatus.FAIL
            message = f"Python version is too old"
            fix_suggestion = (
                f"Upgrade Python to 3.10 or newer. "
                f"Visit https://www.python.org/downloads/ to download a compatible version."
            )
        else:
            status = CheckStatus.WARNING
            message = "Python version may not be fully tested"
            fix_suggestion = (
                f"Consider using Python 3.10 or 3.11 for best compatibility. "
                f"Some packages may not have pre-built wheels for Python {current_info.major}.{current_info.minor}."
            )

        self.report.add_result(CheckResult(
            name="Python Version",
            category=CheckCategory.PYTHON,
            status=status,
            current=current_version,
            expected=expected_str,
            message=message,
            fix_suggestion=fix_suggestion,
        ))

    def check_pytorch_and_cuda(self):
        torch_installed = _is_package_installed("torch")
        torch_version = _get_package_version("torch") or "Not installed"

        cuda_available = "Unknown"
        cuda_version = "N/A"

        if torch_installed:
            try:
                import torch
                cuda_available = "Yes" if torch.cuda.is_available() else "No"
                if torch.cuda.is_available():
                    cuda_version = torch.version.cuda or "Unknown"
                else:
                    cuda_version = "N/A (CUDA not available)"
            except Exception as e:
                cuda_available = "Error"
                cuda_version = f"Error checking CUDA: {e}"

        expected_torch = ">= 2.1.0"
        expected_cuda = "12.1+ (recommended)"

        if not torch_installed:
            status = CheckStatus.FAIL
            message = "PyTorch is not installed"
            fix_suggestion = (
                "Install PyTorch by running: pip install torch torchvision "
                "--extra-index-url https://download.pytorch.org/whl/cu121\n"
                "Or visit https://pytorch.org/get-started/locally/ for installation instructions."
            )
        else:
            try:
                version_ok = parse_version(torch_version) in SpecifierSet(">=2.0.0")
            except Exception:
                version_ok = False

            if not version_ok:
                status = CheckStatus.FAIL
                message = f"PyTorch version {torch_version} is too old"
                fix_suggestion = (
                    "Upgrade PyTorch to version 2.1.0 or newer. "
                    "Run: pip install --upgrade torch torchvision "
                    "--extra-index-url https://download.pytorch.org/whl/cu121"
                )
            elif cuda_available == "No":
                status = CheckStatus.WARNING
                message = "PyTorch installed but CUDA not available. Will use CPU (slow)."
                fix_suggestion = (
                    "If you have an NVIDIA GPU, install CUDA-enabled PyTorch:\n"
                    "pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121\n"
                    "Make sure NVIDIA drivers and CUDA toolkit are installed on your system."
                )
            else:
                status = CheckStatus.PASS
                message = "PyTorch with CUDA is properly configured"
                fix_suggestion = ""

        self.report.add_result(CheckResult(
            name="PyTorch Installation",
            category=CheckCategory.PYTORCH,
            status=status,
            current=f"torch {torch_version}",
            expected=expected_torch,
            message=message,
            fix_suggestion=fix_suggestion,
        ))

        self.report.add_result(CheckResult(
            name="CUDA Availability",
            category=CheckCategory.PYTORCH,
            status=CheckStatus.PASS if cuda_available == "Yes" else CheckStatus.WARNING,
            current=cuda_version,
            expected=expected_cuda,
            message="CUDA is available" if cuda_available == "Yes" else "CUDA is not available",
            fix_suggestion=(
                "Install NVIDIA GPU drivers and CUDA toolkit, then reinstall PyTorch with CUDA support."
                if cuda_available != "Yes" else ""
            ),
        ))

    def check_xformers(self):
        xformers_installed = _is_package_installed("xformers")
        xformers_version = _get_package_version("xformers") or "Not installed"

        if xformers_installed:
            status = CheckStatus.PASS
            message = "xformers is installed (enables memory-efficient attention)"
            fix_suggestion = ""
        else:
            status = CheckStatus.WARNING
            message = "xformers not installed. Will use fallback attention (higher VRAM usage)."
            fix_suggestion = (
                "For better performance and lower VRAM usage, install xformers:\n"
                "pip install xformers==0.0.23 --no-dependencies\n"
                "Note: xformers is optional but recommended."
            )

        self.report.add_result(CheckResult(
            name="xformers",
            category=CheckCategory.DEPENDENCIES,
            status=status,
            current=xformers_version,
            expected="0.0.23 (recommended)",
            message=message,
            fix_suggestion=fix_suggestion,
        ))

    def check_gradio(self):
        gradio_installed = _is_package_installed("gradio")
        gradio_version = _get_package_version("gradio") or "Not installed"

        expected = ">= 3.41.0, < 4.0.0"

        if not gradio_installed:
            status = CheckStatus.FAIL
            message = "gradio is not installed"
            fix_suggestion = "Install gradio: pip install gradio==3.41.2"
        else:
            try:
                v = parse_version(gradio_version)
                version_ok = v >= parse_version("3.41.0") and v < parse_version("4.0.0")
            except Exception:
                version_ok = False

            if not version_ok:
                status = CheckStatus.FAIL
                message = f"gradio version {gradio_version} is incompatible"
                fix_suggestion = (
                    "Install a compatible gradio version. Fooocus requires gradio 3.x:\n"
                    "pip install gradio==3.41.2"
                )
            else:
                status = CheckStatus.PASS
                message = "gradio version is compatible"
                fix_suggestion = ""

        self.report.add_result(CheckResult(
            name="gradio",
            category=CheckCategory.DEPENDENCIES,
            status=status,
            current=gradio_version,
            expected=expected,
            message=message,
            fix_suggestion=fix_suggestion,
        ))

    def check_key_dependencies(self):
        key_packages = [
            ("transformers", "4.42.4"),
            ("safetensors", "0.4.3"),
            ("accelerate", "0.32.1"),
            ("pillow", "10.4.0"),
            ("numpy", "1.26.4"),
            ("opencv-contrib-python-headless", "4.10.0.84"),
            ("tqdm", "4.66.4"),
            ("pyyaml", "6.0.1"),
            ("packaging", "24.1"),
        ]

        for package_name, expected_version in key_packages:
            installed_version = _get_package_version(package_name)

            if installed_version is None:
                status = CheckStatus.FAIL
                message = f"{package_name} is not installed"
                fix_suggestion = (
                    f"Install {package_name}: pip install {package_name}=={expected_version}"
                )
                current = "Not installed"
            else:
                try:
                    v_installed = parse_version(installed_version)
                    v_expected = parse_version(expected_version)
                    version_ok = v_installed >= v_expected
                except Exception:
                    version_ok = True

                if version_ok:
                    status = CheckStatus.PASS
                    message = f"{package_name} is installed"
                    fix_suggestion = ""
                else:
                    status = CheckStatus.WARNING
                    message = f"{package_name} version may be outdated"
                    fix_suggestion = (
                        f"Consider updating {package_name}: pip install --upgrade {package_name}"
                    )
                current = installed_version

            self.report.add_result(CheckResult(
                name=package_name,
                category=CheckCategory.DEPENDENCIES,
                status=status,
                current=current,
                expected=f">= {expected_version}",
                message=message,
                fix_suggestion=fix_suggestion,
            ))

    def _check_directory(self, name: str, path: str, category: str = CheckCategory.DIRECTORIES,
                         check_write: bool = True) -> str:
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path) if exists else False

        if not exists:
            status_dir = CheckStatus.WARNING
            message = f"{name} directory does not exist"
            fix_dir = (
                f"Directory will be created automatically on first use. "
                f"Path: {path}"
            )
            current = "Does not exist"
        elif not is_dir:
            status_dir = CheckStatus.FAIL
            message = f"{name} path exists but is not a directory"
            fix_dir = f"Remove or rename the file at {path}, or change the config to use a different path."
            current = "File (not directory)"
        else:
            status_dir = CheckStatus.PASS
            message = f"{name} directory exists"
            fix_dir = ""
            current = path

        self.report.add_result(CheckResult(
            name=f"{name} Directory",
            category=category,
            status=status_dir,
            current=current,
            expected="Directory must exist",
            message=message,
            fix_suggestion=fix_dir,
        ))

        if check_write and (exists and is_dir or not exists):
            parent_dir = os.path.dirname(path) if not exists else path
            check_path = path if exists else parent_dir
            can_write = _check_write_permission(check_path) if os.path.exists(check_path) else False

            if can_write:
                status_perm = CheckStatus.PASS
                msg_perm = f"{name} directory is writable"
                fix_perm = ""
            else:
                status_perm = CheckStatus.FAIL
                msg_perm = f"{name} directory is not writable"
                fix_perm = (
                    f"Check permissions for {path}. "
                    f"Ensure the user running Fooocus has write access to this directory."
                )

            self.report.add_result(CheckResult(
                name=f"{name} Write Permission",
                category=CheckCategory.PERMISSIONS,
                status=status_perm,
                current="Writable" if can_write else "Not writable",
                expected="Writable",
                message=msg_perm,
                fix_suggestion=fix_perm,
            ))

        return path

    def check_model_directories(self):
        model_dirs = [
            ("Checkpoints", "models/checkpoints"),
            ("LoRA", "models/loras"),
            ("Embeddings", "models/embeddings"),
            ("VAE", "models/vae"),
            ("VAE Approx", "models/vae_approx"),
            ("ControlNet", "models/controlnet"),
            ("CLIP Vision", "models/clip_vision"),
            ("Upscale Models", "models/upscale_models"),
            ("Inpaint Models", "models/inpaint"),
            ("Prompt Expansion", "models/prompt_expansion/fooocus_expansion"),
        ]

        for name, rel_path in model_dirs:
            abs_path = os.path.join(self.root_dir, rel_path)
            self._check_directory(name, abs_path, CheckCategory.DIRECTORIES, check_write=True)

    def check_output_directory(self):
        output_dir = os.path.join(self.root_dir, "outputs")
        self._check_directory("Output", output_dir, CheckCategory.DIRECTORIES, check_write=True)

    def check_temp_directory(self):
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), "fooocus")
        self._check_directory("Temp", temp_dir, CheckCategory.DIRECTORIES, check_write=True)

    def check_config_file(self):
        config_path = os.path.join(self.root_dir, "config.txt")
        presets_dir = os.path.join(self.root_dir, "presets")
        default_preset = os.path.join(presets_dir, "default.json")

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    json.load(f)
                status = CheckStatus.PASS
                message = "Config file exists and is valid JSON"
                current = "Valid"
                fix_suggestion = ""
            except json.JSONDecodeError as e:
                status = CheckStatus.FAIL
                message = f"Config file is invalid JSON: {e}"
                current = "Invalid JSON"
                fix_suggestion = (
                    f"Fix the config file at {config_path}. "
                    f"Common issues: missing commas, trailing commas, or incorrect quoting. "
                    f"You can delete the file and Fooocus will recreate defaults on next launch."
                )
            except Exception as e:
                status = CheckStatus.WARNING
                message = f"Could not read config file: {e}"
                current = "Read error"
                fix_suggestion = f"Check file permissions for {config_path}."
        else:
            status = CheckStatus.WARNING
            message = "Config file does not exist (will be created with defaults)"
            current = "Not found"
            fix_suggestion = "Config file will be created automatically on first launch."

        self.report.add_result(CheckResult(
            name="Config File",
            category=CheckCategory.CONFIG,
            status=status,
            current=current,
            expected="Valid JSON file",
            message=message,
            fix_suggestion=fix_suggestion,
        ))

        if os.path.exists(default_preset):
            try:
                with open(default_preset, "r", encoding="utf-8") as f:
                    json.load(f)
                preset_status = CheckStatus.PASS
                preset_message = "Default preset is valid"
                preset_current = "Valid"
                preset_fix = ""
            except Exception as e:
                preset_status = CheckStatus.FAIL
                preset_message = f"Default preset is invalid: {e}"
                preset_current = "Invalid"
                preset_fix = "Reinstall or restore the default.json preset file."
        else:
            preset_status = CheckStatus.FAIL
            preset_message = "Default preset file is missing"
            preset_current = "Missing"
            preset_fix = "Reinstall Fooocus or restore presets/default.json from the repository."

        self.report.add_result(CheckResult(
            name="Default Preset",
            category=CheckCategory.CONFIG,
            status=preset_status,
            current=preset_current,
            expected="Valid preset file",
            message=preset_message,
            fix_suggestion=preset_fix,
        ))

    def check_disk_space(self):
        min_free_gb = 10
        min_free_bytes = min_free_gb * 1024 * 1024 * 1024

        free_bytes = _get_free_disk_space(self.root_dir)
        free_str = _format_bytes(free_bytes)

        if free_bytes < 0:
            status = CheckStatus.WARNING
            message = "Could not determine free disk space"
            fix_suggestion = "Check disk permissions and ensure there is at least 10 GB of free space."
        elif free_bytes < min_free_bytes:
            status = CheckStatus.FAIL
            message = f"Low disk space. Models require significant storage."
            fix_suggestion = (
                f"Free up disk space. At least {min_free_gb} GB recommended for models and outputs. "
                f"Consider moving models to a different drive using config path settings."
            )
        else:
            status = CheckStatus.PASS
            message = "Sufficient disk space available"
            fix_suggestion = ""

        self.report.add_result(CheckResult(
            name="Free Disk Space",
            category=CheckCategory.DISK,
            status=status,
            current=free_str,
            expected=f">= {min_free_gb} GB",
            message=message,
            fix_suggestion=fix_suggestion,
        ))


def format_report(report: PreflightReport, use_colors: bool = True) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  Fooocus Environment Preflight Check")
    lines.append("=" * 72)
    lines.append("")

    summary_color = "\033[32m" if not report.has_errors else "\033[31m"
    reset = "\033[0m" if use_colors else ""

    lines.append(f"  Total checks: {len(report.results)}")
    lines.append(f"  Passed: {summary_color}{len(report.passed)}{reset}")
    lines.append(f"  Failed: {summary_color if report.has_errors else ''}{len(report.failed)}{reset}")
    lines.append(f"  Warnings: {len(report.warnings)}")
    lines.append("")

    current_category = None
    for result in report.results:
        if result.category != current_category:
            current_category = result.category
            lines.append("-" * 72)
            lines.append(f"  [{current_category}]")
            lines.append("-" * 72)

        if use_colors:
            if result.status == CheckStatus.PASS:
                status_str = f"\033[32m✓ PASS\033[0m"
            elif result.status == CheckStatus.FAIL:
                status_str = f"\033[31m✗ FAIL\033[0m"
            elif result.status == CheckStatus.WARNING:
                status_str = f"\033[33m⚠ WARN\033[0m"
            else:
                status_str = f"  SKIP"
        else:
            status_str = f"[{result.status[:4]}]"

        lines.append(f"  {status_str}  {result.name}")
        lines.append(f"         Current:  {result.current}")
        lines.append(f"         Expected: {result.expected}")

        if result.message and result.status != CheckStatus.PASS:
            lines.append(f"         Message:  {result.message}")

        if result.fix_suggestion and result.status != CheckStatus.PASS:
            if use_colors:
                lines.append(f"         \033[36mFix: {result.fix_suggestion}\033[0m")
            else:
                lines.append(f"         Fix: {result.fix_suggestion}")

        lines.append("")

    if report.has_errors:
        lines.append("=" * 72)
        lines.append("  \033[31mPREFLIGHT CHECK FAILED\033[0m" if use_colors else "  PREFLIGHT CHECK FAILED")
        lines.append("=" * 72)
        lines.append("")
        lines.append("  Please fix the issues above before launching Fooocus.")
        lines.append("")
        lines.append("  Quick fixes:")
        lines.append("    - Install missing packages: pip install -r requirements_versions.txt")
        lines.append("    - Install PyTorch with CUDA: see https://pytorch.org/get-started/")
        lines.append("    - Check disk space and directory permissions")
        lines.append("")
    elif report.has_warnings:
        lines.append("=" * 72)
        lines.append("  \033[33mPREFLIGHT CHECK PASSED WITH WARNINGS\033[0m" if use_colors else "  PREFLIGHT CHECK PASSED WITH WARNINGS")
        lines.append("=" * 72)
        lines.append("")
        lines.append("  Fooocus may still work, but some features may be limited.")
        lines.append("")
    else:
        lines.append("=" * 72)
        lines.append("  \033[32mALL PREFLIGHT CHECKS PASSED\033[0m" if use_colors else "  ALL PREFLIGHT CHECKS PASSED")
        lines.append("=" * 72)
        lines.append("")

    return "\n".join(lines)


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


_RUN_CACHE: Dict[str, PreflightReport] = {}


def run_preflight(root_dir: Optional[str] = None,
                  policy: Optional[str] = None,
                  mode: Optional[str] = None,
                  exit_on_error: bool = False,
                  print_report: bool = True,
                  use_colors: bool = True,
                  as_json: bool = False,
                  stage: Optional[str] = None,
                  call_exit: bool = True) -> Optional[PreflightReport]:
    """
    运行环境预检。

    **推荐用法：只传 policy= 名称**，例如 ``run_preflight(policy="launch_early")``。
    所有 mode / stage / 退出码 / 缓存键规则都由 :class:`PreflightPolicy` 统一管理，
    避免不同入口散传参数导致行为不一致。

    Parameters
    ----------
    policy : str or None
        标准策略名称，如 ``launch_early`` / ``healthcheck`` / ``ci`` / ``cli_default``。
        优先于 mode / stage / exit_on_error 参数。
        可用列表： ``PreflightPolicy.all_policies().keys()``
    mode : {"report", "healthcheck", "strict"} or None
        兼容参数。未指定 policy 时有效。
    exit_on_error : bool
        **已废弃**，等价于 ``mode='strict'``。会打印 deprecation warning。
    stage : str or None
        兼容参数。用于缓存键，未指定 policy 时有效。
    call_exit : bool
        是否在内部调用 sys.exit。False 时仅将退出码写入 ``report._exit_code``。
    """
    try:
        effective_policy = _resolve_policy(policy, mode, stage, exit_on_error)
    except ValueError as e:
        print(f"[Preflight] Error: {e}", file=sys.stderr)
        if call_exit:
            sys.exit(2)
        raise

    force_json = _env_flag("FOOOCUS_PREFLIGHT_JSON", default=False)
    effective_json = force_json or as_json

    skip = _env_flag("FOOOCUS_SKIP_PREFLIGHT", default=False)
    if skip:
        if print_report:
            if effective_json:
                payload = {
                    "skipped": True,
                    "reason": "FOOOCUS_SKIP_PREFLIGHT=1",
                    "policy": effective_policy.name,
                    "mode": effective_policy.mode,
                }
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"[Preflight] Skipped (FOOOCUS_SKIP_PREFLIGHT=1, "
                    f"policy={effective_policy.name}, mode={effective_policy.mode})"
                )
        if call_exit:
            sys.exit(0)
        return None

    # 缓存键使用 policy.stage，保证同一 stage 的不同入口（如 launch.py early 和 CI early）共享同一缓存
    cache_key = f"{os.path.abspath(root_dir or '.')}:{effective_policy.stage}"
    cached = _RUN_CACHE.get(cache_key)
    if cached is not None and not effective_json:
        if print_report:
            print(
                f"[Preflight] policy={effective_policy.name} "
                f"(stage={effective_policy.stage}) skipped (cached from earlier run)"
            )
        cached._exit_code = _mode_exit_code(effective_policy.mode, cached)
        cached._policy = effective_policy.name
        if call_exit and cached._exit_code != 0:
            sys.exit(cached._exit_code)
        return cached

    checker = PreflightChecker(root_dir=root_dir)
    report = checker.run_all()
    _RUN_CACHE[cache_key] = report
    report._exit_code = _mode_exit_code(effective_policy.mode, report)
    report._policy = effective_policy.name

    if print_report:
        if effective_json:
            payload = report.to_dict()
            payload["policy"] = effective_policy.name
            payload["mode"] = effective_policy.mode
            payload["stage"] = effective_policy.stage
            payload["exit_code"] = report._exit_code
            print(json.dumps(payload, indent=2))
        else:
            print(format_report(report, use_colors=use_colors))
            if effective_policy.mode != CheckMode.REPORT and report._exit_code != 0:
                print(
                    f"[Preflight] policy={effective_policy.name} (mode={effective_policy.mode}): "
                    f"exit code {report._exit_code} will be returned "
                    f"({len(report.failed)} failures detected).",
                    file=sys.stderr
                )

    if call_exit and report._exit_code != 0:
        sys.exit(report._exit_code)

    return report


def _main():
    import argparse
    import warnings

    parser = argparse.ArgumentParser(
        description="Fooocus Environment Preflight Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available policies:\n"
            + "\n".join(
                f"  {name:20s} - {p.description}"
                for name, p in sorted(PreflightPolicy.all_policies().items())
            )
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory of the Fooocus installation"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="运行策略名称（推荐）。可用列表见下方。"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(CheckMode._ALL),
        default=None,
        help="运行模式 (兼容参数，推荐改用 --policy)。"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="阶段标签，用于缓存键 (兼容参数，推荐改用 --policy)。"
    )
    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        help="[已废弃] 有 FAIL 就非零退出。等价于 --policy strict。"
    )
    parser.add_argument(
        "--no-exit",
        action="store_true",
        help="[已废弃] 无论是否有错误退出码都为 0。等价于 --policy report。将打印 deprecation warning。"
    )
    parser.add_argument(
        "--no-colors",
        action="store_true",
        help="Disable colored output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable format"
    )

    args = parser.parse_args()

    # 解析最终 policy
    if args.policy is not None:
        resolved_policy = args.policy
    elif args.no_exit:
        # --no-exit → report 模式
        if not args.json:
            print(
                "[Preflight] Warning: --no-exit is deprecated, "
                "use --policy report instead.",
                file=sys.stderr
            )
        resolved_policy = "report"
    elif args.exit_on_error or args.mode or args.stage:
        # 使用兼容参数走 _resolve_policy 动态合成
        resolved_policy = None
    else:
        # 独立 CLI 默认严格模式
        resolved_policy = "cli_default"

    try:
        report = run_preflight(
            root_dir=args.root,
            policy=resolved_policy,
            mode=args.mode if resolved_policy is None else None,
            stage=args.stage if resolved_policy is None else None,
            exit_on_error=args.exit_on_error if resolved_policy is None else False,
            print_report=True,
            use_colors=not args.no_colors and not args.json,
            as_json=args.json,
            call_exit=True
        )
    except ValueError:
        sys.exit(2)


if __name__ == "__main__":
    _main()
