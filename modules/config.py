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


def _get_schema_config(key: str, default: any = None) -> any:
    val = config_result.get(key, None)
    if val is not None:
        return val
    field = config_schema.get_field(key)
    if field:
        return field.default_value
    return default


def _apply_cli_overrides():
    global config_dict

    if args_manager.args.output_path:
        output_path = os.path.abspath(args_manager.args.output_path)
        makedirs_with_log(output_path)
        config_dict['path_outputs'] = output_path
        print(f'Overriding config value path_outputs with CLI arg {output_path}')
        config_result.values['path_outputs'] = config_result.values.get(
            'path_outputs',
            config_result.values.get('path_outputs')
        )
        try:
            from modules.config_schema import ConfigValue
            config_result.values['path_outputs'] = ConfigValue(
                key='path_outputs',
                value=output_path,
                source=ConfigSource.CLI_ARGUMENT,
                source_detail='cli:output-path'
            )
        except:
            pass

    if args_manager.args.temp_path:
        temp_path_val = os.path.abspath(args_manager.args.temp_path)
        try:
            os.makedirs(temp_path_val, exist_ok=True)
            config_dict['temp_path'] = temp_path_val
            print(f'Overriding config value temp_path with CLI arg {temp_path_val}')
            try:
                from modules.config_schema import ConfigValue
                config_result.values['temp_path'] = ConfigValue(
                    key='temp_path',
                    value=temp_path_val,
                    source=ConfigSource.CLI_ARGUMENT,
                    source_detail='cli:temp-path'
                )
            except:
                pass
        except Exception as e:
            print(f'Could not create temp path from CLI arg {args_manager.args.temp_path}. Reason: {e}')

    if args_manager.args.preset:
        config_dict['default_performance_preset_cli'] = args_manager.args.preset


def _load_all_configs():
    global config_dict, always_save_keys, visited_keys, loaded_preset_content, loaded_config_file_content

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
            config_dict.update(builtin_preset_data)
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
                config_dict.update(config_file_data)
                always_save_keys = list(config_file_data.keys())
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
        config_dict.update(preset_data)

    env_issues = config_schema.load_env(config_result)
    config_result.issues.extend(env_issues)

    for key, field in config_schema.fields.items():
        if field.save_to_config and key not in visited_keys:
            visited_keys.append(key)
        if field.save_to_config and key not in always_save_keys:
            always_save_keys.append(key)

    config_schema.apply_defaults(config_result)

    _apply_cli_overrides()

    for key, cv in config_result.values.items():
        if cv.value is not None:
            config_dict[key] = cv.value
            field = config_schema.get_field(key)
            if field and field.save_to_config:
                if key not in visited_keys:
                    visited_keys.append(key)
                if key not in always_save_keys and cv.source in (ConfigSource.CONFIG_FILE, ConfigSource.ENVIRONMENT_VARIABLE, ConfigSource.CLI_ARGUMENT):
                    always_save_keys.append(key)

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
                if new_key not in config_dict:
                    config_dict[new_key] = deprecated_config_dict[old_key]
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
            config_dict.update(deprecated_config_dict)
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

    preset_path = os.path.join(get_user_presets_dir(), f'{preset_name}.json')

    try:
        with open(preset_path, "w", encoding="utf-8") as json_file:
            json.dump(preset_data, json_file, indent=4, ensure_ascii=False)
        print(f'User preset saved: {preset_path}')
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
        if config_key in content:
            value = content[config_key]
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
        val = _get_schema_config(key, None)
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

    output_val = _get_schema_config('path_outputs', None)
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
    tp = _get_schema_config('temp_path', None)
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


def _get_config_val(key, default_value, validator, disable_empty_as_none=False, expected_type=None):
    default = default_value
    env = os.getenv(key)
    if env is not None:
        env = try_eval_env_var(env, expected_type)
        print(f"Environment: {key} = {env}")
        config_dict[key] = env

    if key not in config_dict:
        config_dict[key] = default
        return default

    v = config_dict.get(key, None)
    if not disable_empty_as_none:
        if v is None or v == '':
            v = 'None'

    try:
        is_valid = validator(v)
    except Exception:
        is_valid = False

    if is_valid:
        return v
    else:
        if v is not None:
            print(f'Failed to load config key: {json.dumps({key:v})} is invalid; will use {json.dumps({key:default})} instead.')
        config_dict[key] = default
        return default


temp_path_cleanup_on_launch = _get_config_val(
    key='temp_path_cleanup_on_launch',
    default_value=True,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)

default_base_model_name = default_model = _get_config_val(
    key='default_model',
    default_value='model.safetensors',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
previous_default_models = _get_config_val(
    key='previous_default_models',
    default_value=[],
    validator=lambda x: isinstance(x, list) and all(isinstance(k, str) for k in x),
    expected_type=list
)
default_refiner_model_name = default_refiner = _get_config_val(
    key='default_refiner',
    default_value='None',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
default_refiner_switch = _get_config_val(
    key='default_refiner_switch',
    default_value=0.8,
    validator=lambda x: isinstance(x, numbers.Number) and 0 <= x <= 1,
    expected_type=numbers.Number
)
default_loras_min_weight = _get_config_val(
    key='default_loras_min_weight',
    default_value=-2,
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10,
    expected_type=numbers.Number
)
default_loras_max_weight = _get_config_val(
    key='default_loras_max_weight',
    default_value=2,
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10,
    expected_type=numbers.Number
)
default_loras = _get_config_val(
    key='default_loras',
    default_value=[
        [True, "None", 1.0], [True, "None", 1.0], [True, "None", 1.0],
        [True, "None", 1.0], [True, "None", 1.0]
    ],
    validator=lambda x: isinstance(x, list) and all(
        len(y) == 3 and isinstance(y[0], bool) and isinstance(y[1], str) and isinstance(y[2], numbers.Number)
        or len(y) == 2 and isinstance(y[0], str) and isinstance(y[1], numbers.Number)
        for y in x),
    expected_type=list
)
default_loras = [(y[0], y[1], y[2]) if len(y) == 3 else (True, y[0], y[1]) for y in default_loras]
default_max_lora_number = _get_config_val(
    key='default_max_lora_number',
    default_value=len(default_loras) if isinstance(default_loras, list) and len(default_loras) > 0 else 5,
    validator=lambda x: isinstance(x, int) and x >= 1,
    expected_type=int
)
default_cfg_scale = _get_config_val(
    key='default_cfg_scale',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_sample_sharpness = _get_config_val(
    key='default_sample_sharpness',
    default_value=2.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_sampler = _get_config_val(
    key='default_sampler',
    default_value='dpmpp_2m_sde_gpu',
    validator=lambda x: x in modules.flags.sampler_list,
    expected_type=str
)
default_scheduler = _get_config_val(
    key='default_scheduler',
    default_value='karras',
    validator=lambda x: x in modules.flags.scheduler_list,
    expected_type=str
)
default_vae = _get_config_val(
    key='default_vae',
    default_value=modules.flags.default_vae,
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
default_styles = _get_config_val(
    key='default_styles',
    default_value=["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],
    validator=lambda x: isinstance(x, list) and all(y in modules.sdxl_styles.legal_style_names for y in x),
    expected_type=list
)
default_prompt_negative = _get_config_val(
    key='default_prompt_negative',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True,
    expected_type=str
)
default_prompt = _get_config_val(
    key='default_prompt',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True,
    expected_type=str
)
default_performance = _get_config_val(
    key='default_performance',
    default_value=Performance.SPEED.value,
    validator=lambda x: x in Performance.values(),
    expected_type=str
)
default_image_prompt_checkbox = _get_config_val(
    key='default_image_prompt_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_enhance_checkbox = _get_config_val(
    key='default_enhance_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_advanced_checkbox = _get_config_val(
    key='default_advanced_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_developer_debug_mode_checkbox = _get_config_val(
    key='default_developer_debug_mode_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_image_prompt_advanced_checkbox = _get_config_val(
    key='default_image_prompt_advanced_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_max_image_number = _get_config_val(
    key='default_max_image_number',
    default_value=32,
    validator=lambda x: isinstance(x, int) and x >= 1,
    expected_type=int
)
default_output_format = _get_config_val(
    key='default_output_format',
    default_value='png',
    validator=lambda x: x in OutputFormat.list(),
    expected_type=str
)
default_image_number = _get_config_val(
    key='default_image_number',
    default_value=2,
    validator=lambda x: isinstance(x, int) and 1 <= x <= default_max_image_number,
    expected_type=int
)
checkpoint_downloads = _get_config_val(
    key='checkpoint_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
lora_downloads = _get_config_val(
    key='lora_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
embeddings_downloads = _get_config_val(
    key='embeddings_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
vae_downloads = _get_config_val(
    key='vae_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
available_aspect_ratios = _get_config_val(
    key='available_aspect_ratios',
    default_value=modules.flags.sdxl_aspect_ratios,
    validator=lambda x: isinstance(x, list) and all('*' in v for v in x) and len(x) > 1,
    expected_type=list
)
default_aspect_ratio = _get_config_val(
    key='default_aspect_ratio',
    default_value='1152*896' if '1152*896' in available_aspect_ratios else available_aspect_ratios[0],
    validator=lambda x: x in available_aspect_ratios,
    expected_type=str
)
default_inpaint_engine_version = _get_config_val(
    key='default_inpaint_engine_version',
    default_value='v2.6',
    validator=lambda x: x in modules.flags.inpaint_engine_versions,
    expected_type=str
)
default_selected_image_input_tab_id = _get_config_val(
    key='default_selected_image_input_tab_id',
    default_value=modules.flags.default_input_image_tab,
    validator=lambda x: x in modules.flags.input_image_tab_ids,
    expected_type=str
)
default_uov_method = _get_config_val(
    key='default_uov_method',
    default_value=modules.flags.disabled,
    validator=lambda x: x in modules.flags.uov_list,
    expected_type=str
)
default_controlnet_image_count = _get_config_val(
    key='default_controlnet_image_count',
    default_value=4,
    validator=lambda x: isinstance(x, int) and x > 0,
    expected_type=int
)
default_ip_images = {}
default_ip_stop_ats = {}
default_ip_weights = {}
default_ip_types = {}

for image_count in range(default_controlnet_image_count):
    image_count += 1
    default_ip_images[image_count] = _get_config_val(
        key=f'default_ip_image_{image_count}',
        default_value='None',
        validator=lambda x: x == 'None' or isinstance(x, str),
        expected_type=str
    )

    if default_ip_images[image_count] == 'None':
        default_ip_images[image_count] = None

    default_ip_types[image_count] = _get_config_val(
        key=f'default_ip_type_{image_count}',
        default_value=modules.flags.default_ip,
        validator=lambda x: x in modules.flags.ip_list,
        expected_type=str
    )

    default_end, default_weight = modules.flags.default_parameters[default_ip_types[image_count]]

    default_ip_stop_ats[image_count] = _get_config_val(
        key=f'default_ip_stop_at_{image_count}',
        default_value=default_end,
        validator=lambda x: isinstance(x, float) and 0 <= x <= 1,
        expected_type=float
    )
    default_ip_weights[image_count] = _get_config_val(
        key=f'default_ip_weight_{image_count}',
        default_value=default_weight,
        validator=lambda x: isinstance(x, float) and 0 <= x <= 2,
        expected_type=float
    )

default_inpaint_advanced_masking_checkbox = _get_config_val(
    key='default_inpaint_advanced_masking_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_inpaint_method = _get_config_val(
    key='default_inpaint_method',
    default_value=modules.flags.inpaint_option_default,
    validator=lambda x: x in modules.flags.inpaint_options,
    expected_type=str
)
default_cfg_tsnr = _get_config_val(
    key='default_cfg_tsnr',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_clip_skip = _get_config_val(
    key='default_clip_skip',
    default_value=2,
    validator=lambda x: isinstance(x, int) and 1 <= x <= modules.flags.clip_skip_max,
    expected_type=int
)
default_overwrite_step = _get_config_val(
    key='default_overwrite_step',
    default_value=-1,
    validator=lambda x: isinstance(x, int),
    expected_type=int
)
default_overwrite_switch = _get_config_val(
    key='default_overwrite_switch',
    default_value=-1,
    validator=lambda x: isinstance(x, int),
    expected_type=int
)
default_overwrite_upscale = _get_config_val(
    key='default_overwrite_upscale',
    default_value=-1,
    validator=lambda x: isinstance(x, numbers.Number)
)
def _validate_example_prompts(x):
    if not isinstance(x, list):
        return False
    for item in x:
        if isinstance(item, str):
            continue
        if isinstance(item, list) and len(item) >= 1 and isinstance(item[0], str):
            continue
        return False
    return True

example_inpaint_prompts = _get_config_val(
    key='example_inpaint_prompts',
    default_value=[
        'highly detailed face', 'detailed girl face', 'detailed man face', 'detailed hand', 'beautiful eyes'
    ],
    validator=_validate_example_prompts,
    expected_type=list
)
example_enhance_detection_prompts = _get_config_val(
    key='example_enhance_detection_prompts',
    default_value=[
        'face', 'eye', 'mouth', 'hair', 'hand', 'body'
    ],
    validator=_validate_example_prompts,
    expected_type=list
)
default_enhance_tabs = _get_config_val(
    key='default_enhance_tabs',
    default_value=3,
    validator=lambda x: isinstance(x, int) and 1 <= x <= 5,
    expected_type=int
)
default_enhance_uov_method = _get_config_val(
    key='default_enhance_uov_method',
    default_value=modules.flags.disabled,
    validator=lambda x: x in modules.flags.uov_list,
    expected_type=int
)
default_enhance_uov_processing_order = _get_config_val(
    key='default_enhance_uov_processing_order',
    default_value=modules.flags.enhancement_uov_before,
    validator=lambda x: x in modules.flags.enhancement_uov_processing_order,
    expected_type=int
)
default_enhance_uov_prompt_type = _get_config_val(
    key='default_enhance_uov_prompt_type',
    default_value=modules.flags.enhancement_uov_prompt_type_original,
    validator=lambda x: x in modules.flags.enhancement_uov_prompt_types,
    expected_type=int
)
default_sam_max_detections = _get_config_val(
    key='default_sam_max_detections',
    default_value=0,
    validator=lambda x: isinstance(x, int) and 0 <= x <= 10,
    expected_type=int
)
default_black_out_nsfw = _get_config_val(
    key='default_black_out_nsfw',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_save_only_final_enhanced_image = _get_config_val(
    key='default_save_only_final_enhanced_image',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_save_metadata_to_images = _get_config_val(
    key='default_save_metadata_to_images',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_metadata_scheme = _get_config_val(
    key='default_metadata_scheme',
    default_value=MetadataScheme.FOOOCUS.value,
    validator=lambda x: x in [y[1] for y in modules.flags.metadata_scheme if y[1] == x],
    expected_type=str
)
metadata_created_by = _get_config_val(
    key='metadata_created_by',
    default_value='',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)

example_inpaint_prompts = [[x] for x in example_inpaint_prompts] if example_inpaint_prompts and isinstance(example_inpaint_prompts[0], str) else example_inpaint_prompts
example_enhance_detection_prompts = [[x] for x in example_enhance_detection_prompts] if example_enhance_detection_prompts and isinstance(example_enhance_detection_prompts[0], str) else example_enhance_detection_prompts

default_invert_mask_checkbox = _get_config_val(
    key='default_invert_mask_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)

default_inpaint_mask_model = _get_config_val(
    key='default_inpaint_mask_model',
    default_value='isnet-general-use',
    validator=lambda x: x in modules.flags.inpaint_mask_models,
    expected_type=str
)

default_enhance_inpaint_mask_model = _get_config_val(
    key='default_enhance_inpaint_mask_model',
    default_value='sam',
    validator=lambda x: x in modules.flags.inpaint_mask_models,
    expected_type=str
)

default_inpaint_mask_cloth_category = _get_config_val(
    key='default_inpaint_mask_cloth_category',
    default_value='full',
    validator=lambda x: x in modules.flags.inpaint_mask_cloth_category,
    expected_type=str
)

default_inpaint_mask_sam_model = _get_config_val(
    key='default_inpaint_mask_sam_model',
    default_value='vit_b',
    validator=lambda x: x in modules.flags.inpaint_mask_sam_model,
    expected_type=str
)

default_describe_apply_prompts_checkbox = _get_config_val(
    key='default_describe_apply_prompts_checkbox',
    default_value=True,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_describe_content_type = _get_config_val(
    key='default_describe_content_type',
    default_value=[modules.flags.describe_type_photo],
    validator=lambda x: all(k in modules.flags.describe_types for k in x),
    expected_type=list
)

config_dict["default_loras"] = default_loras = default_loras[:default_max_lora_number] + [[True, 'None', 1.0] for _ in range(default_max_lora_number - len(default_loras))]

possible_preset_keys = {
    "default_model": "base_model",
    "default_refiner": "refiner_model",
    "default_refiner_switch": "refiner_switch",
    "previous_default_models": "previous_default_models",
    "default_loras_min_weight": "default_loras_min_weight",
    "default_loras_max_weight": "default_loras_max_weight",
    "default_loras": "<processed>",
    "default_cfg_scale": "guidance_scale",
    "default_sample_sharpness": "sharpness",
    "default_cfg_tsnr": "adaptive_cfg",
    "default_clip_skip": "clip_skip",
    "default_sampler": "sampler",
    "default_scheduler": "scheduler",
    "default_overwrite_step": "steps",
    "default_overwrite_switch": "overwrite_switch",
    "default_performance": "performance",
    "default_image_number": "image_number",
    "default_prompt": "prompt",
    "default_prompt_negative": "negative_prompt",
    "default_styles": "styles",
    "default_aspect_ratio": "resolution",
    "default_save_metadata_to_images": "default_save_metadata_to_images",
    "checkpoint_downloads": "checkpoint_downloads",
    "embeddings_downloads": "embeddings_downloads",
    "lora_downloads": "lora_downloads",
    "vae_downloads": "vae_downloads",
    "default_vae": "vae",
    "default_inpaint_engine_version": "inpaint_engine_version",
}

REWRITE_PRESET = False

if REWRITE_PRESET and isinstance(args_manager.args.preset, str):
    save_path = os.path.join(_root_dir, 'presets', args_manager.args.preset + '.json')
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in possible_preset_keys}, json_file, indent=4)
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
        save_keys = [k for k in always_save_keys if k in config_dict]
        json.dump({k: config_dict[k] for k in save_keys}, json_file, indent=4)

with open(config_example_path, "w", encoding="utf-8") as json_file:
    cpa = config_path.replace("\\", "\\\\")
    json_file.write(f'You can modify your "{cpa}" using the below keys, formats, and examples.\n'
                    f'Do not modify this file. Modifications in this file will not take effect.\n'
                    f'This file is a tutorial and example. Please edit "{cpa}" to really change any settings.\n'
                    + 'Remember to split the paths with "\\\\" rather than "\\", '
                      'and there is no "," before the last "}". \n\n\n')
    visit_keys = [k for k in visited_keys if k in config_dict]
    json.dump({k: config_dict[k] for k in visit_keys}, json_file, indent=4)

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
