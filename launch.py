import os
import ssl
import sys

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

import platform
import fooocus_version

from build_launcher import build_launcher
from modules.launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content
from modules.model_loader import load_file_from_url

REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if TRY_INSTALL_XFORMERS:
        if REINSTALL_ALL or not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.23')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    return


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]


def run_preflight_checks(project_root, strict=False):
    """
    启动前预检：调用 service 层的统一 preflight 编排。
    与 python -m modules.tools all 使用完全相同的底层逻辑和错误等级判断。

    错误等级：
      - BLOCKING (exit code 2)：Python 版本不兼容、核心依赖缺失、配置 JSON/schema 错误
      - WARNING  (exit code 1)：目录不存在、可选依赖缺失、schema 未知 key、模型数为 0
      - INFO     (exit code 0)：正常信息

    strict=True 时，所有 WARNING 升级为 BLOCKING。

    返回：退出码 (0/1/2)
    """
    try:
        from modules.services.preflight import run_preflight, Severity

        print('[Preflight] Running preflight checks...')
        print(f'[Preflight] Mode: {"strict" if strict else "normal"}')
        print()

        pf_result = run_preflight(
            project_root=project_root,
            check_env=True,
            check_config=True,
            check_resources=False,
            strict=strict,
            skip_torch=False,
        )

        # 打印所有检查项（BLOCKING 红色风格，WARNING 黄色风格，INFO 普通）
        blocking_items = [c for c in pf_result.checks if c.severity == Severity.BLOCKING]
        warning_items = [c for c in pf_result.checks if c.severity == Severity.WARNING]
        info_items = [c for c in pf_result.checks if c.severity == Severity.INFO]

        if blocking_items:
            print(f'[Preflight] BLOCKING issues ({len(blocking_items)}):')
            for c in blocking_items:
                print(f'  ✗ [BLOCKING] {c.name}: {c.message}')
                if c.suggestion:
                    print(f'      ↳ Suggestion: {c.suggestion}')
            print()

        if warning_items:
            print(f'[Preflight] Warnings ({len(warning_items)}):')
            for c in warning_items:
                print(f'  ⚠ [WARNING] {c.name}: {c.message}')
            print()

        if info_items:
            print(f'[Preflight] Info ({len(info_items)} items shown in verbose mode)')
            # 默认只打印摘要，不刷屏

        # 摘要
        print(f'[Preflight] Summary: {pf_result.blocking_count} blocking, '
              f'{pf_result.warning_count} warnings, {pf_result.info_count} info')

        exit_code = pf_result.to_dict()["exit_code"]

        if pf_result.blocking_count > 0:
            print(f'[Preflight] FAILED with {pf_result.blocking_count} blocking issue(s). '
                  f'Exit code: {exit_code}')
            print('[Preflight] Fix the blocking issues above before starting Fooocus.')
            print('[Preflight] You can also run: python -m modules.tools all --json  for full details')
        elif pf_result.warning_count > 0:
            print(f'[Preflight] Passed with {pf_result.warning_count} warning(s). '
                  f'Exit code: {exit_code}')
            if strict:
                print('[Preflight] Strict mode: warnings promoted to blocking. Startup aborted.')
            else:
                print('[Preflight] Startup will continue.')
        else:
            print(f'[Preflight] All checks passed. Exit code: {exit_code}')

        print()
        return exit_code

    except Exception as e:
        # preflight 本身异常不应该阻止启动（除非 strict 模式）
        print(f'[Preflight] WARNING: Preflight checks encountered an internal error: {e}')
        print('[Preflight]          Startup will continue normally.')
        import traceback
        traceback.print_exc()
        print()
        return 1


def ini_args():
    from args_manager import args
    return args


prepare_environment()
build_launcher()
args = ini_args()

# 启动前预检：blocking 级错误直接退出
preflight_strict = bool(getattr(args, 'preflight_strict', False)) or os.environ.get('FOOOCUS_PREFLIGHT_STRICT', '0') == '1'
preflight_exit_code = run_preflight_checks(root, strict=preflight_strict)
if preflight_exit_code >= 2:
    print(f'[Preflight] Aborting startup due to blocking preflight issues (exit code {preflight_exit_code}).')
    sys.exit(preflight_exit_code)

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

if args.hf_mirror is not None:
    os.environ['HF_MIRROR'] = str(args.hf_mirror)
    print("Set hf_mirror to:", args.hf_mirror)

from modules import config
from modules.hash_cache import init_cache

os.environ["U2NET_HOME"] = config.path_inpaint

os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads):
    from modules.util import get_file_from_folder_list

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

    if args.disable_preset_download:
        print('Skipped model download.')
        return default_model, checkpoint_downloads

    if not args.always_download_new_model:
        if not os.path.isfile(get_file_from_folder_list(default_model, config.paths_checkpoints)):
            for alternative_model_name in previous_default_models:
                if os.path.isfile(get_file_from_folder_list(alternative_model_name, config.paths_checkpoints)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    for file_name, url in checkpoint_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_checkpoints))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_loras))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_vae, file_name=file_name)

    return default_model, checkpoint_downloads


config.default_base_model_name, config.checkpoint_downloads = download_models(
    config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
    config.embeddings_downloads, config.lora_downloads, config.vae_downloads)

config.update_files()
init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)

from modules import diagnostics as _diagnostics

if getattr(args, 'disable_diagnostics', False):
    _diagnostics.set_diagnostics_enabled(False)
    print('[Diagnostics] Unified diagnostic logging is disabled.')
else:
    if getattr(args, 'diagnostics_log_file', None):
        try:
            _diagnostics.set_log_to_file(True, args.diagnostics_log_file)
            print(f'[Diagnostics] Logging to file: {args.diagnostics_log_file}')
        except Exception as e:
            print(f'[Diagnostics] Warning: Failed to setup log file: {str(e)}')

    if getattr(args, 'disable_diagnostics_redaction', False):
        _diagnostics.set_sensitive_data_redacted(False)
        print('[Diagnostics] WARNING: Sensitive data redaction is DISABLED. Full paths and prompts may appear in logs.')

    if getattr(args, 'diagnostics_prompt_max_chars', 120) != 120:
        _diagnostics.set_prompt_max_chars(args.diagnostics_prompt_max_chars)
        print(f'[Diagnostics] Prompt max chars set to: {args.diagnostics_prompt_max_chars}')

from webui import *
