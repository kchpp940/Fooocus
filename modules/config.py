import os
import json
import math
import numbers
import tempfile

import args_manager
import modules.flags
import modules.sdxl_styles

from modules.model_loader import load_file_from_url
from modules.extra_utils import makedirs_with_log, get_files_from_folder, try_eval_env_var
from modules.flags import OutputFormat, Performance, MetadataScheme
from modules.config_schema import (
    ConfigSchema,
    ConfigLoadResult,
    ConfigSource,
    ConfigField,
    build_fooocus_schema,
)

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_user_data_dir():
    env_dir = os.getenv('FOOOCUS_USER_DATA_DIR')
    if env_dir:
        user_dir = os.path.abspath(env_dir)
    else:
        config_path_val = os.getenv('config_path')
        if config_path_val:
            user_dir = os.path.dirname(os.path.abspath(config_path_val))
        else:
            home = os.path.expanduser('~')
            user_dir = os.path.join(home, '.fooocus')
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def get_user_presets_dir():
    user_presets_dir = os.path.join(get_user_data_dir(), 'user_presets')
    os.makedirs(user_presets_dir, exist_ok=True)
    return user_presets_dir


def get_config_path(key, default_value):
    env = os.getenv(key)
    if env is not None and isinstance(env, str):
        print(f"Environment: {key} = {env}")
        return env
    else:
        return os.path.abspath(default_value)


config_schema: ConfigSchema = build_fooocus_schema(_root_dir)
config_result: ConfigLoadResult = ConfigLoadResult()

wildcards_max_bfs_depth = 64

config_path = get_config_path('config_path', os.path.join(_root_dir, 'config.txt'))
config_example_path = get_config_path('config_example_path', os.path.join(_root_dir, 'config_modification_tutorial.txt'))

config_dict: dict = {}
always_save_keys: list = []
visited_keys: list = []

loaded_preset_content: dict = {}
loaded_config_file_content: dict = {}


def get_ui_default(key: str, default: any = None) -> any:
    return config_schema.get_ui_default(config_result, key, default)


def get_config_value(key: str, default: any = None) -> any:
    return config_schema.get_value(config_result, key, default)


def get_config_source(key: str) -> str:
    return config_schema.get_source_detail(config_result, key)


def export_for_preset(include_schema_defaults: bool = False) -> dict:
    return config_schema.export_for_preset(config_result, include_schema_defaults)


def export_for_config_file(include_schema_defaults: bool = False) -> dict:
    return config_schema.export_for_config_file(config_result, include_schema_defaults)


def import_and_validate_preset_data(data: dict) -> tuple:
    return config_schema.import_and_validate(data, ConfigSource.USER_PRESET, 'user_preset_save')


def get_export_metadata() -> dict:
    return config_schema.get_export_metadata(config_result)


def export_from_ui_values(ui_values: dict) -> dict:
    return config_schema.export_from_ui_values(ui_values)


def get_preset_binding_map() -> dict:
    return config_schema.get_preset_binding_map()


def get_diagnostics():
    return config_schema.build_diagnostics(config_result)


def format_diagnostics(show_warnings_only: bool = False) -> str:
    return config_schema.format_diagnostics_terminal(config_result, show_warnings_only)


def run_config_preflight(root_dir: str = None, output_json: bool = False,
                         include_config_txt: bool = True,
                         include_user_presets: bool = False,
                         include_deprecated_user_path: bool = False):
    if root_dir is None:
        root_dir = _root_dir
    return config_schema.ConfigSchema.standalone_check(
        root_dir=root_dir,
        output_json=output_json,
        include_config_txt=include_config_txt,
        include_user_presets=include_user_presets,
        include_deprecated_user_path=include_deprecated_user_path,
    )


def _apply_cli_overrides():
    cli_issues = args_manager.sync_cli_args_to_config_result(config_result)
    config_result.issues.extend(cli_issues)

    if args_manager.args.preset:
        config_dict['default_performance_preset_cli'] = args_manager.args.preset


def _load_all_configs():
    global config_dict, loaded_preset_content, loaded_config_file_content

    builtin_preset_path = os.path.join(_root_dir, 'presets', 'default.json')
    builtin_preset_data = {}
    try:
        with open(builtin_preset_path, "r", encoding="utf-8") as json_file:
            builtin_preset_data = json.load(json_file)
            issues = config_schema.load_dict(
                config_result, builtin_preset_data,
                ConfigSource.BUILTIN_PRESET,
                f'preset:default.json'
            )
            config_result.issues.extend(issues)
    except Exception as e:
        print(f'Load default preset failed.')
        print(e)

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as json_file:
                config_file_data = json.load(json_file)
                loaded_config_file_content = dict(config_file_data)
                issues = config_schema.load_dict(
                    config_result, config_file_data,
                    ConfigSource.CONFIG_FILE,
                    f'config:{os.path.basename(config_path)}'
                )
                config_result.issues.extend(issues)
        except Exception as e:
            print(f'Failed to load config file "{config_path}" . The reason is: {str(e)}')
            print('Please make sure that:')
            print(f'1. The file "{config_path}" is a valid text file, and you have access to read it.')
            print('2. Use "\\\\" instead of "\\" when describing paths.')
            print('3. There is no "," before the last "}".')
            print('4. All key/value formats are correct.')

    _try_load_deprecated_user_path_config()

    preset_name = args_manager.args.preset
    if preset_name:
        preset_data = _try_get_preset_content(preset_name)
        loaded_preset_content = dict(preset_data)
        preset_is_user = _is_user_preset(preset_name)
        source = ConfigSource.USER_PRESET if preset_is_user else ConfigSource.BUILTIN_PRESET
        issues = config_schema.load_dict(
            config_result, preset_data,
            source,
            f'preset:{preset_name}'
        )
        config_result.issues.extend(issues)

    env_issues = config_schema.load_env(config_result)
    config_result.issues.extend(env_issues)

    config_schema.apply_defaults(config_result)

    _apply_cli_overrides()

    for key, cv in config_result.values.items():
        if cv.value is not None:
            config_dict[key] = cv.value

    print(config_schema.format_summary(config_result))


def _try_load_deprecated_user_path_config():
    global config_dict

    deprecated_user_path_config = os.path.join(_root_dir, 'user_path_config.txt')
    if not os.path.exists(deprecated_user_path_config):
        return

    try:
        deprecated_config_dict = json.load(open(deprecated_user_path_config, "r", encoding="utf-8"))

        def replace_config(old_key, new_key):
            if old_key in deprecated_config_dict:
                issues = config_schema.apply_value(
                    config_result, old_key, deprecated_config_dict[old_key],
                    ConfigSource.DEPRECATED_USER_PATH_CONFIG,
                    f'deprecated:{old_key}'
                )
                config_result.issues.extend(issues)
                del deprecated_config_dict[old_key]

        replace_config('modelfile_path', 'path_checkpoints')
        replace_config('lorafile_path', 'path_loras')
        replace_config('embeddings_path', 'path_embeddings')
        replace_config('vae_approx_path', 'path_vae_approx')
        replace_config('upscale_models_path', 'path_upscale_models')
        replace_config('inpaint_models_path', 'path_inpaint')
        replace_config('controlnet_models_path', 'path_controlnet')
        replace_config('clip_vision_models_path', 'path_clip_vision')
        replace_config('fooocus_expansion_path', 'path_fooocus_expansion')
        replace_config('temp_outputs_path', 'path_outputs')

        if deprecated_config_dict.get("default_model", None) == 'juggernautXL_version6Rundiffusion.safetensors':
            os.replace(deprecated_user_path_config, os.path.join(_root_dir, 'user_path_config-deprecated.txt'))
            print('Config updated successfully in silence. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return

        if input("Newer models and configs are available. "
                 "Download and update files? [Y/n]:") in ['n', 'N', 'No', 'no', 'NO']:
            issues = config_schema.load_dict(
                config_result, deprecated_config_dict,
                ConfigSource.DEPRECATED_USER_PATH_CONFIG,
                'deprecated:user_path_config.txt'
            )
            config_result.issues.extend(issues)
            print('Loading using deprecated old models and deprecated old configs.')
            return
        else:
            os.replace(deprecated_user_path_config, os.path.join(_root_dir, 'user_path_config-deprecated.txt'))
            print('Config updated successfully by user. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return
    except Exception as e:
        print('Processing deprecated config failed')
        print(e)
    return


USER_PRESET_PREFIX = '[User] '


def get_builtin_presets_dir():
    return os.path.join(_root_dir, 'presets')


def _is_user_preset(preset_name):
    return isinstance(preset_name, str) and preset_name.startswith(USER_PRESET_PREFIX)


def is_user_preset(preset_name):
    return _is_user_preset(preset_name)


def strip_user_prefix(preset_name):
    if is_user_preset(preset_name):
        return preset_name[len(USER_PRESET_PREFIX):]
    return preset_name


def add_user_prefix(preset_name):
    if not is_user_preset(preset_name):
        return USER_PRESET_PREFIX + preset_name
    return preset_name


def get_builtin_presets():
    preset_folder = get_builtin_presets_dir()
    if not os.path.exists(preset_folder):
        return []
    return [f[:-5] for f in os.listdir(preset_folder) if f.endswith('.json')]


def get_user_presets():
    preset_folder = get_user_presets_dir()
    if not os.path.exists(preset_folder):
        return []
    return [add_user_prefix(f[:-5]) for f in os.listdir(preset_folder) if f.endswith('.json')]


def get_presets():
    presets = ['initial']
    builtin = get_builtin_presets()
    user = get_user_presets()
    return presets + builtin + user


def update_presets():
    global available_presets
    available_presets = get_presets()


def _try_get_preset_content(preset):
    if not isinstance(preset, str):
        return {}

    if preset == 'initial':
        return {}

    if is_user_preset(preset):
        preset_name = strip_user_prefix(preset)
        preset_path = os.path.join(get_user_presets_dir(), f'{preset_name}.json')
    else:
        preset_path = os.path.join(get_builtin_presets_dir(), f'{preset}.json')

    try:
        if os.path.exists(preset_path):
            with open(preset_path, "r", encoding="utf-8") as json_file:
                json_content = json.load(json_file)
                print(f'Loaded preset: {preset_path}')
                return json_content
        else:
            raise FileNotFoundError
    except Exception as e:
        print(f'Load preset [{preset_path}] failed')
        print(e)
    return {}


def try_get_preset_content(preset):
    return _try_get_preset_content(preset)


def save_user_preset(preset_name, preset_data):
    if not isinstance(preset_name, str) or preset_name.strip() == '':
        return False, 'Preset name cannot be empty'

    preset_name = strip_user_prefix(preset_name).strip()

    invalid_chars = '<>:"/\\|?*'
    if any(c in preset_name for c in invalid_chars):
        return False, f'Preset name contains invalid characters: {invalid_chars}'

    builtin_presets = get_builtin_presets()
    if preset_name in builtin_presets:
        return False, 'Cannot overwrite built-in preset'

    cleaned_data, issues = import_and_validate_preset_data(preset_data)
    if issues:
        warn_msgs = []
        for issue in issues:
            warn_msgs.append(f"[{issue.issue_type.value}] {issue.key}: {issue.message}")
        print(f'[Preset Save Validation] {len(issues)} issue(s):')
        for msg in warn_msgs:
            print(f'  - {msg}')

    preset_path = os.path.join(get_user_presets_dir(), f'{preset_name}.json')

    try:
        with open(preset_path, "w", encoding="utf-8") as json_file:
            json.dump(cleaned_data, json_file, indent=4, ensure_ascii=False)
        print(f'User preset saved: {preset_path} ({len(cleaned_data)} keys, {len(issues)} filtered)')
        update_presets()
        return True, add_user_prefix(preset_name)
    except Exception as e:
        print(f'Save user preset [{preset_name}] failed')
        print(e)
        return False, str(e)


def delete_user_preset(preset_name):
    if not is_user_preset(preset_name):
        return False, 'Can only delete user presets'

    preset_name = strip_user_prefix(preset_name)
    preset_path = os.path.join(get_user_presets_dir(), f'{preset_name}.json')

    try:
        if os.path.exists(preset_path):
            os.remove(preset_path)
            print(f'User preset deleted: {preset_path}')
            update_presets()
            return True, 'Deleted successfully'
        else:
            return False, 'Preset not found'
    except Exception as e:
        print(f'Delete user preset [{preset_name}] failed')
        print(e)
        return False, str(e)


def rename_user_preset(old_preset_name, new_preset_name):
    if not is_user_preset(old_preset_name):
        return False, 'Can only rename user presets'

    old_name = strip_user_prefix(old_preset_name)
    new_name = strip_user_prefix(new_preset_name).strip()

    if new_name == '':
        return False, 'New preset name cannot be empty'

    invalid_chars = '<>:"/\\|?*'
    if any(c in new_name for c in invalid_chars):
        return False, f'Preset name contains invalid characters: {invalid_chars}'

    builtin_presets = get_builtin_presets()
    if new_name in builtin_presets:
        return False, 'Name conflicts with built-in preset'

    user_presets = get_user_presets()
    if add_user_prefix(new_name) in user_presets and new_name != old_name:
        return False, 'A user preset with this name already exists'

    old_path = os.path.join(get_user_presets_dir(), f'{old_name}.json')
    new_path = os.path.join(get_user_presets_dir(), f'{new_name}.json')

    try:
        if not os.path.exists(old_path):
            return False, 'Source preset not found'
        os.rename(old_path, new_path)
        print(f'User preset renamed: {old_path} -> {new_path}')
        update_presets()
        return True, add_user_prefix(new_name)
    except Exception as e:
        print(f'Rename user preset [{old_name}] -> [{new_name}] failed')
        print(e)
        return False, str(e)


def duplicate_user_preset(source_preset_name, new_preset_name):
    source_content = try_get_preset_content(source_preset_name)
    if not source_content:
        return False, 'Source preset content is empty or not found'

    return save_user_preset(new_preset_name, source_content)


def get_preset_details(preset_name):
    content = try_get_preset_content(preset_name)
    if not content:
        if preset_name == 'initial':
            return {'name': preset_name, 'type': 'initial', 'details': {}, 'description': 'Initial default settings'}
        return None

    validated_content, issues = import_and_validate_preset_data(content)

    result = {
        'name': preset_name,
        'type': 'user' if is_user_preset(preset_name) else 'builtin',
        'details': {},
        'description': ''
    }

    display_keys = {
        'default_model': 'Base Model',
        'default_refiner': 'Refiner Model',
        'default_refiner_switch': 'Refiner Switch',
        'default_loras': 'LoRAs',
        'default_cfg_scale': 'CFG Scale',
        'default_sample_sharpness': 'Sharpness',
        'default_cfg_tsnr': 'Adaptive CFG (TSNR)',
        'default_clip_skip': 'CLIP Skip',
        'default_sampler': 'Sampler',
        'default_scheduler': 'Scheduler',
        'default_performance': 'Performance',
        'default_styles': 'Default Styles',
        'default_aspect_ratio': 'Aspect Ratio',
        'default_overwrite_step': 'Steps (overwrite)',
        'default_vae': 'VAE',
        'default_inpaint_engine_version': 'Inpaint Engine',
    }

    for config_key, display_name in display_keys.items():
        if config_key in validated_content:
            value = validated_content[config_key]
            if config_key == 'default_loras' and isinstance(value, list):
                lora_strs = []
                for lora in value:
                    if isinstance(lora, list) and len(lora) >= 2:
                        if len(lora) == 3:
                            enabled, name, weight = lora
                            status = '✓' if enabled else '✗'
                            lora_strs.append(f'{status} {name} (w={weight})')
                        else:
                            name, weight = lora[0], lora[1]
                            lora_strs.append(f'{name} (w={weight})')
                result['details'][display_name] = lora_strs if lora_strs else 'None'
            elif config_key == 'default_styles' and isinstance(value, list):
                result['details'][display_name] = value
            else:
                result['details'][display_name] = value

    return result


available_presets = get_presets()


def _resolve_paths_from_schema():
    resolved = {}

    def _resolve_single(key, default_rel, as_array=False, make_dir=True):
        val = get_config_value(key, None)
        if val is not None:
            if isinstance(val, str):
                if not os.path.isabs(val):
                    val = os.path.abspath(os.path.join(os.path.dirname(__file__), val))
                if make_dir:
                    makedirs_with_log(val)
                if as_array:
                    resolved[key] = [val]
                else:
                    resolved[key] = val
                return resolved[key]
            elif isinstance(val, list) and as_array:
                paths = []
                for p in val:
                    if isinstance(p, str):
                        if not os.path.isabs(p):
                            p = os.path.abspath(os.path.join(os.path.dirname(__file__), p))
                        if make_dir:
                            makedirs_with_log(p)
                        paths.append(p)
                resolved[key] = paths
                return paths

        if isinstance(default_rel, list):
            dp = []
            for path in default_rel:
                abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
                dp.append(abs_path)
                os.makedirs(abs_path, exist_ok=True)
            resolved[key] = dp
        else:
            dp = os.path.abspath(os.path.join(os.path.dirname(__file__), default_rel))
            os.makedirs(dp, exist_ok=True)
            if as_array:
                resolved[key] = [dp]
            else:
                resolved[key] = dp
        config_dict[key] = resolved[key]
        return resolved[key]

    resolved['path_checkpoints'] = _resolve_single('path_checkpoints', ['../models/checkpoints/'], True)
    resolved['path_loras'] = _resolve_single('path_loras', ['../models/loras/'], True)
    resolved['path_embeddings'] = _resolve_single('path_embeddings', '../models/embeddings/')
    resolved['path_vae_approx'] = _resolve_single('path_vae_approx', '../models/vae_approx/')
    resolved['path_vae'] = _resolve_single('path_vae', '../models/vae/')
    resolved['path_upscale_models'] = _resolve_single('path_upscale_models', '../models/upscale_models/')
    resolved['path_inpaint'] = _resolve_single('path_inpaint', '../models/inpaint/')
    resolved['path_controlnet'] = _resolve_single('path_controlnet', '../models/controlnet/')
    resolved['path_clip_vision'] = _resolve_single('path_clip_vision', '../models/clip_vision/')
    resolved['path_fooocus_expansion'] = _resolve_single('path_fooocus_expansion', '../models/prompt_expansion/fooocus_expansion')
    resolved['path_wildcards'] = _resolve_single('path_wildcards', '../wildcards/')
    resolved['path_safety_checker'] = _resolve_single('path_safety_checker', '../models/safety_checker/')
    resolved['path_sam'] = _resolve_single('path_sam', '../models/sam/')

    output_val = get_config_value('path_outputs', None)
    if output_val and isinstance(output_val, str):
        if not os.path.isabs(output_val):
            output_val = os.path.abspath(os.path.join(os.path.dirname(__file__), output_val))
        makedirs_with_log(output_val)
        resolved['path_outputs'] = output_val
    else:
        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/'))
        os.makedirs(dp, exist_ok=True)
        resolved['path_outputs'] = dp
    if args_manager.args.output_path:
        resolved['path_outputs'] = os.path.abspath(args_manager.args.output_path)
        makedirs_with_log(resolved['path_outputs'])
    config_dict['path_outputs'] = resolved['path_outputs']

    return resolved


def get_path_output() -> str:
    return path_outputs


_load_all_configs()

paths_resolved = _resolve_paths_from_schema()

paths_checkpoints = paths_resolved['path_checkpoints']
paths_loras = paths_resolved['path_loras']
path_embeddings = paths_resolved['path_embeddings']
path_vae_approx = paths_resolved['path_vae_approx']
path_vae = paths_resolved['path_vae']
path_upscale_models = paths_resolved['path_upscale_models']
path_inpaint = paths_resolved['path_inpaint']
path_controlnet = paths_resolved['path_controlnet']
path_clip_vision = paths_resolved['path_clip_vision']
path_fooocus_expansion = paths_resolved['path_fooocus_expansion']
path_wildcards = paths_resolved['path_wildcards']
path_safety_checker = paths_resolved['path_safety_checker']
path_sam = paths_resolved['path_sam']
path_outputs = paths_resolved['path_outputs']


def _init_temp_path():
    tp = get_config_value('temp_path', None)
    default_tp = os.path.join(tempfile.gettempdir(), 'fooocus')

    if args_manager.args.temp_path:
        tp = args_manager.args.temp_path

    if tp and tp != '' and tp != default_tp:
        try:
            if not os.path.isabs(tp):
                tp = os.path.abspath(tp)
            os.makedirs(tp, exist_ok=True)
            print(f'Using temp path {tp}')
            return tp
        except Exception as e:
            print(f'Could not create temp path {tp}. Reason: {e}')
            print(f'Using default temp path {default_tp} instead.')

    os.makedirs(default_tp, exist_ok=True)
    return default_tp


temp_path = _init_temp_path()


temp_path_cleanup_on_launch = get_config_value('temp_path_cleanup_on_launch', True)

default_base_model_name = default_model = get_config_value('default_model', 'model.safetensors')
previous_default_models = get_config_value('previous_default_models', [])
default_refiner_model_name = default_refiner = get_config_value('default_refiner', 'None')
default_refiner_switch = get_config_value('default_refiner_switch', 0.8)
default_loras_min_weight = get_config_value('default_loras_min_weight', -2)
default_loras_max_weight = get_config_value('default_loras_max_weight', 2)
default_loras = get_config_value('default_loras', [
    [True, "None", 1.0], [True, "None", 1.0], [True, "None", 1.0],
    [True, "None", 1.0], [True, "None", 1.0]
])
default_loras = [(y[0], y[1], y[2]) if len(y) == 3 else (True, y[0], y[1]) for y in default_loras]
default_max_lora_number = get_config_value('default_max_lora_number', 5)
default_cfg_scale = get_config_value('default_cfg_scale', 7.0)
default_sample_sharpness = get_config_value('default_sample_sharpness', 2.0)
default_sampler = get_config_value('default_sampler', 'dpmpp_2m_sde_gpu')
default_scheduler = get_config_value('default_scheduler', 'karras')
default_vae = get_config_value('default_vae', modules.flags.default_vae)
default_styles = get_config_value('default_styles', ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])
default_prompt_negative = get_config_value('default_prompt_negative', '')
default_prompt = get_config_value('default_prompt', '')
default_performance = get_config_value('default_performance', Performance.SPEED.value)
default_image_prompt_checkbox = get_ui_default('default_image_prompt_checkbox', False)
default_enhance_checkbox = get_ui_default('default_enhance_checkbox', False)
default_advanced_checkbox = get_ui_default('default_advanced_checkbox', False)
default_developer_debug_mode_checkbox = get_ui_default('default_developer_debug_mode_checkbox', False)
default_image_prompt_advanced_checkbox = get_ui_default('default_image_prompt_advanced_checkbox', False)
default_max_image_number = get_config_value('default_max_image_number', 32)
default_output_format = get_config_value('default_output_format', 'png')
default_image_number = get_config_value('default_image_number', 2)
checkpoint_downloads = get_config_value('checkpoint_downloads', {})
lora_downloads = get_config_value('lora_downloads', {})
embeddings_downloads = get_config_value('embeddings_downloads', {})
vae_downloads = get_config_value('vae_downloads', {})
available_aspect_ratios = get_config_value('available_aspect_ratios', modules.flags.sdxl_aspect_ratios)
default_aspect_ratio = get_config_value('default_aspect_ratio', '1152*896' if '1152*896' in available_aspect_ratios else available_aspect_ratios[0])
default_inpaint_engine_version = get_config_value('default_inpaint_engine_version', 'v2.6')
default_selected_image_input_tab_id = get_ui_default('default_selected_image_input_tab_id', modules.flags.default_input_image_tab)
default_uov_method = get_ui_default('default_uov_method', modules.flags.disabled)
default_controlnet_image_count = get_config_value('default_controlnet_image_count', 4)

default_ip_images = {}
default_ip_stop_ats = {}
default_ip_weights = {}
default_ip_types = {}

for image_count in range(default_controlnet_image_count):
    image_count += 1
    default_ip_images[image_count] = get_ui_default(f'default_ip_image_{image_count}', 'None')
    if default_ip_images[image_count] == 'None':
        default_ip_images[image_count] = None

    default_ip_types[image_count] = get_ui_default(f'default_ip_type_{image_count}', modules.flags.default_ip)
    default_end, default_weight = modules.flags.default_parameters[default_ip_types[image_count]]
    default_ip_stop_ats[image_count] = get_ui_default(f'default_ip_stop_at_{image_count}', default_end)
    default_ip_weights[image_count] = get_ui_default(f'default_ip_weight_{image_count}', default_weight)

default_inpaint_advanced_masking_checkbox = get_ui_default('default_inpaint_advanced_masking_checkbox', False)
default_inpaint_method = get_ui_default('default_inpaint_method', modules.flags.inpaint_option_default)
default_cfg_tsnr = get_config_value('default_cfg_tsnr', 7.0)
default_clip_skip = get_config_value('default_clip_skip', 2)
default_overwrite_step = get_config_value('default_overwrite_step', -1)
default_overwrite_switch = get_config_value('default_overwrite_switch', -1)
default_overwrite_upscale = get_config_value('default_overwrite_upscale', -1)

example_inpaint_prompts = get_config_value('example_inpaint_prompts', [
    'highly detailed face', 'detailed girl face', 'detailed man face', 'detailed hand', 'beautiful eyes'
])
example_enhance_detection_prompts = get_config_value('example_enhance_detection_prompts', [
    'face', 'eye', 'mouth', 'hair', 'hand', 'body'
])

default_enhance_tabs = get_config_value('default_enhance_tabs', 3)
default_enhance_uov_method = get_ui_default('default_enhance_uov_method', modules.flags.disabled)
default_enhance_uov_processing_order = get_ui_default('default_enhance_uov_processing_order', modules.flags.enhancement_uov_before)
default_enhance_uov_prompt_type = get_ui_default('default_enhance_uov_prompt_type', modules.flags.enhancement_uov_prompt_type_original)
default_sam_max_detections = get_config_value('default_sam_max_detections', 0)
default_black_out_nsfw = get_config_value('default_black_out_nsfw', False)
default_save_only_final_enhanced_image = get_config_value('default_save_only_final_enhanced_image', False)
default_save_metadata_to_images = get_config_value('default_save_metadata_to_images', False)
default_metadata_scheme = get_config_value('default_metadata_scheme', MetadataScheme.FOOOCUS.value)
metadata_created_by = get_config_value('metadata_created_by', '')

default_invert_mask_checkbox = get_ui_default('default_invert_mask_checkbox', False)
default_inpaint_mask_model = get_ui_default('default_inpaint_mask_model', 'isnet-general-use')
default_enhance_inpaint_mask_model = get_ui_default('default_enhance_inpaint_mask_model', 'sam')
default_inpaint_mask_cloth_category = get_ui_default('default_inpaint_mask_cloth_category', 'full')
default_inpaint_mask_sam_model = get_ui_default('default_inpaint_mask_sam_model', 'vit_b')
default_describe_apply_prompts_checkbox = get_ui_default('default_describe_apply_prompts_checkbox', True)
default_describe_content_type = get_ui_default('default_describe_content_type', [modules.flags.describe_type_photo])

config_dict["default_loras"] = default_loras = default_loras[:default_max_lora_number] + [[True, 'None', 1.0] for _ in range(default_max_lora_number - len(default_loras))]


def _build_possible_preset_keys():
    result = {}
    binding_map = config_schema.get_preset_binding_map()
    for binding_name, binding_info in binding_map.items():
        result[binding_info['key']] = binding_name
    for field in config_schema.get_preset_fields():
        if field.key not in result:
            if field.preset_binding is None:
                result[field.key] = field.key
    result['default_loras'] = '<processed>'
    return result

possible_preset_keys = _build_possible_preset_keys()

REWRITE_PRESET = False

if REWRITE_PRESET and isinstance(args_manager.args.preset, str):
    save_path = os.path.join(_root_dir, 'presets', args_manager.args.preset + '.json')
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump(export_for_preset(include_schema_defaults=True), json_file, indent=4)
    print(f'Preset saved to {save_path}. Exiting ...')
    exit(0)


def add_ratio(x):
    a, b = x.replace('*', ' ').split(' ')[:2]
    a, b = int(a), int(b)
    g = math.gcd(a, b)
    return f'{a}×{b} <span style="color: grey;"> \U00002223 {a // g}:{b // g}</span>'


default_aspect_ratio = add_ratio(default_aspect_ratio)
available_aspect_ratios_labels = [add_ratio(x) for x in available_aspect_ratios]

if not os.path.exists(config_path):
    with open(config_path, "w", encoding="utf-8") as json_file:
        json.dump(export_for_config_file(include_schema_defaults=False), json_file, indent=4)
    print(f'Config file created at {config_path} via schema export')

with open(config_example_path, "w", encoding="utf-8") as json_file:
    cpa = config_path.replace("\\", "\\\\")
    json_file.write(f'You can modify your "{cpa}" using the below keys, formats, and examples.\n'
                    f'Do not modify this file. Modifications in this file will not take effect.\n'
                    f'This file is a tutorial and example. Please edit "{cpa}" to really change any settings.\n'
                    + 'Remember to split the paths with "\\\\" rather than "\\", '
                      'and there is no "," before the last "}". \n\n\n')
    json.dump(export_for_config_file(include_schema_defaults=True), json_file, indent=4)

model_filenames = []
lora_filenames = []
vae_filenames = []
wildcard_filenames = []


def get_model_filenames(folder_paths, extensions=None, name_filter=None):
    if extensions is None:
        extensions = ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch']
    files = []

    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]
    for folder in folder_paths:
        files += get_files_from_folder(folder, extensions, name_filter)

    return files


def update_files():
    global model_filenames, lora_filenames, vae_filenames, wildcard_filenames, available_presets
    model_filenames = get_model_filenames(paths_checkpoints)
    lora_filenames = get_model_filenames(paths_loras)
    vae_filenames = get_model_filenames(path_vae)
    wildcard_filenames = get_files_from_folder(path_wildcards, ['.txt'])
    available_presets = get_presets()
    return


def downloading_inpaint_models(v):
    assert v in modules.flags.inpaint_engine_versions

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=path_inpaint,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(path_inpaint, 'fooocus_inpaint_head.pth')
    patch_file = None

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v25.fooocus.patch')

    if v == 'v2.6':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v26.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v26.fooocus.patch')

    return head_file, patch_file


def downloading_sdxl_lcm_lora():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.EXTREME_SPEED.value
    )
    return modules.flags.PerformanceLoRA.EXTREME_SPEED.value


def downloading_sdxl_lightning_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.LIGHTNING.value
    )
    return modules.flags.PerformanceLoRA.LIGHTNING.value


def downloading_sdxl_hyper_sd_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.HYPER_SD.value
    )
    return modules.flags.PerformanceLoRA.HYPER_SD.value


def downloading_controlnet_canny():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=path_controlnet,
        file_name='control-lora-canny-rank128.safetensors'
    )
    return os.path.join(path_controlnet, 'control-lora-canny-rank128.safetensors')


def downloading_controlnet_cpds():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_xl_cpds_128.safetensors'
    )
    return os.path.join(path_controlnet, 'fooocus_xl_cpds_128.safetensors')


def downloading_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=path_clip_vision,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(path_clip_vision, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(path_controlnet, 'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=path_upscale_models,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')


def downloading_safety_checker_model():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin',
        model_dir=path_safety_checker,
        file_name='stable-diffusion-safety-checker.bin'
    )
    return os.path.join(path_safety_checker, 'stable-diffusion-safety-checker.bin')


def download_sam_model(sam_model: str) -> str:
    match sam_model:
        case 'vit_b':
            return downloading_sam_vit_b()
        case 'vit_l':
            return downloading_sam_vit_l()
        case 'vit_h':
            return downloading_sam_vit_h()
        case _:
            raise ValueError(f"sam model {sam_model} does not exist.")


def downloading_sam_vit_b():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth',
        model_dir=path_sam,
        file_name='sam_vit_b_01ec64.pth'
    )
    return os.path.join(path_sam, 'sam_vit_b_01ec64.pth')


def downloading_sam_vit_l():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth',
        model_dir=path_sam,
        file_name='sam_vit_l_0b3195.pth'
    )
    return os.path.join(path_sam, 'sam_vit_l_0b3195.pth')


def downloading_sam_vit_h():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth',
        model_dir=path_sam,
        file_name='sam_vit_h_4b8939.pth'
    )
    return os.path.join(path_sam, 'sam_vit_h_4b8939.pth')
