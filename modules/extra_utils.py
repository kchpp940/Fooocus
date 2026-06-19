import os
from ast import literal_eval


def makedirs_with_log(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(f'Directory {path} could not be created, reason: {error}')


def get_files_from_folder(folder_path, extensions=None, name_filter=None):
    """Wrapper 保持 API 兼容；底层使用 modules.services.resource_scanner.scan_directory。
    保持原有行为：目录不存在时抛出 ValueError。

    注意：这里使用惰性导入以避免循环依赖
    （config_schema → extra_utils → resource_scanner → config_inspector → config_schema）。
    """
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")
    from modules.services.resource_scanner import scan_directory
    return scan_directory(folder_path, extensions=extensions, name_filter=name_filter)


def try_eval_env_var(value: str, expected_type=None):
    try:
        value_eval = value
        if expected_type is bool:
            value_eval = value.title()
        value_eval = literal_eval(value_eval)
        if expected_type is not None and not isinstance(value_eval, expected_type):
            return value
        return value_eval
    except:
        return value
