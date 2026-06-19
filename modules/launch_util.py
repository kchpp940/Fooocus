import os
import importlib
import importlib.util
import shutil
import subprocess
import sys
import re
import logging

from modules.services.environment_inspector import (
    is_package_installed,
    check_requirements_strict,
)

logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)  # sshh...
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

re_requirement = re.compile(r"\s*([-\w]+)\s*(?:==\s*([-+.\w]+))?\s*")

python = sys.executable
default_command_live = (os.environ.get('LAUNCH_LIVE_OUTPUT') == "1")
index_url = os.environ.get('INDEX_URL', "")

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def is_installed(package):
    """Wrapper 保持 API 兼容；底层使用 modules.services.environment_inspector.is_package_installed。"""
    return is_package_installed(package)


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def run_pip(command, desc=None, live=default_command_live):
    try:
        index_url_line = f' --index-url {index_url}' if index_url != '' else ''
        return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}",
                   errdesc=f"Couldn't install {desc}", live=live)
    except Exception as e:
        print(e)
        print(f'CMD Failed {desc}: {command}')
        return None


def requirements_met(requirements_file):
    """Wrapper 保持 API 兼容；底层使用 modules.services.environment_inspector.check_requirements_strict。
    保持原有行为：遇到不满足的包就打印并返回 False。
    """
    result = check_requirements_strict(requirements_file)
    if not result.get("found", False):
        return False
    for issue in result.get("issues", []):
        status = issue.get("status", "")
        if status == "ok":
            continue
        package = issue.get("package", "?")
        detail = issue.get("detail", "")
        if status == "version_mismatch":
            print(f"Version mismatch for {package}: {detail}")
            return False
        elif status == "missing":
            print(f"Missing package: {package} — {detail}")
            return False
        elif status == "error":
            print(f"Error checking {package}: {detail}")
            return False
    return result.get("all_met", False)


def delete_folder_content(folder, prefix=None):
    result = True

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'{prefix}Failed to delete {file_path}. Reason: {e}')
            result = False

    return result