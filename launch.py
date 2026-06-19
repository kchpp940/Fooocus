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


def ini_args():
    from args_manager import args
    return args


prepare_environment()
build_launcher()
args = ini_args()

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

if args.hf_mirror is not None:
    os.environ['HF_MIRROR'] = str(args.hf_mirror)
    print("Set hf_mirror to:", args.hf_mirror)

from modules import config
from modules.hash_cache import init_cache
from modules.manifest import (
    load_manifest, check_manifest_resources, print_manifest_report,
    build_default_manifest, save_manifest, get_manifest_path,
    sha256_file, ManifestResolver, ManifestResolutionError,
)

os.environ["U2NET_HOME"] = config.path_inpaint

os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")


def get_manifest_to_use():
    manifest_path = getattr(args, 'manifest_path', None)
    if manifest_path and os.path.exists(manifest_path):
        print(f'[Manifest] Using manifest from: {manifest_path}')
        return load_manifest(manifest_path)

    default_manifest_path = get_manifest_path(config.get_data_dir())
    if os.path.exists(default_manifest_path):
        print(f'[Manifest] Using default manifest from: {default_manifest_path}')
        return load_manifest(default_manifest_path)

    print('[Manifest] No manifest file found, generating from default config...')
    manifest = build_default_manifest(
        config.checkpoint_downloads,
        config.lora_downloads,
        config.embeddings_downloads,
        config.vae_downloads,
        vae_approx_filenames,
        'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        fooocus_version=fooocus_version.version,
    )
    return manifest


def run_manifest_check():
    if not getattr(args, 'manifest_check', False) and not getattr(args, 'manifest_strict', False):
        return True

    manifest = get_manifest_to_use()
    if not manifest.resources:
        print('[Manifest] No resources defined in manifest, skipping check.')
        return True

    check_hash = getattr(args, 'manifest_check_hash', False)
    models_root = config.get_models_dir()
    result = check_manifest_resources(manifest, models_root, check_hash=check_hash)
    ok = print_manifest_report(result, manifest)

    if not ok and getattr(args, 'manifest_strict', False):
        print()
        print('[ERROR] Manifest check failed in strict mode. Aborting launch.')
        print(f'[ERROR] Models directory: {models_root}')
        print('[ERROR] Please download the required models and place them in the correct directories,')
        print('[ERROR] or disable strict mode with --no-manifest-strict.')
        return False

    return True


def handle_generate_manifest():
    output_path = getattr(args, 'generate_manifest', None)
    if not output_path:
        return

    print(f'[Manifest] Generating manifest to: {output_path}')
    manifest = build_default_manifest(
        config.checkpoint_downloads,
        config.lora_downloads,
        config.embeddings_downloads,
        config.vae_downloads,
        vae_approx_filenames,
        'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        fooocus_version=fooocus_version.version,
    )

    if getattr(args, 'manifest_check_hash', False):
        models_root = config.get_models_dir()
        print('[Manifest] Computing file hashes...')
        for item in manifest.resources.values():
            from modules.manifest import resolve_resource_path
            file_path = resolve_resource_path(item, models_root)
            if os.path.exists(file_path):
                try:
                    item.sha256 = sha256_file(file_path)
                    print(f'  [OK] {item.name}: {item.sha256[:16]}...')
                except Exception as e:
                    print(f'  [!!] {item.name}: failed to hash - {e}')

    save_manifest(manifest, output_path)
    print(f'[Manifest] Manifest generated successfully.')
    sys.exit(0)


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


handle_generate_manifest()

if not run_manifest_check():
    print()
    print('[FATAL] Launch aborted due to manifest check failure.')
    sys.exit(1)

_manifest = get_manifest_to_use()
_strict_mode = getattr(args, 'manifest_strict', False)
_check_hash = getattr(args, 'manifest_check_hash', False)
ManifestResolver.initialize(_manifest, config.get_models_dir(), strict=_strict_mode, check_hash=_check_hash)
print(f'[Manifest] Resolver initialized (strict={_strict_mode}, check_hash={_check_hash}, resources={len(_manifest.resources)})')

try:
    config.default_base_model_name, config.checkpoint_downloads = download_models(
        config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
        config.embeddings_downloads, config.lora_downloads, config.vae_downloads)
except ManifestResolutionError as e:
    print()
    print('=' * 70)
    print('  FATAL: Offline/Strict Mode - Required Resource Missing')
    print('=' * 70)
    print()
    print(str(e))
    print()
    print('  The application cannot start because a required resource is missing')
    print('  and network downloads are blocked in strict/offline mode.')
    print()
    print('  To resolve this:')
    print('  1. Download the file manually and place it at the expected path above')
    print('  2. Or add it to your manifest.json with the correct category and URL')
    print('  3. Or run without --manifest-strict to allow automatic downloads')
    print()
    print(f'  Data directory: {config.get_data_dir()}')
    print(f'  Models directory: {config.get_models_dir()}')
    print(f'  Manifest file: {getattr(args, "manifest_path", None) or get_manifest_path(config.get_data_dir())}')
    print()
    sys.exit(1)

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
