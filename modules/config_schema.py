import os
import json
import numbers
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ast import literal_eval


class ConfigSource(Enum):
    HARDCODED_FALLBACK = "hardcoded_fallback"
    SCHEMA_DEFAULT = "schema_default"
    BUILTIN_PRESET = "builtin_preset"
    CONFIG_FILE = "config_file"
    DEPRECATED_USER_PATH_CONFIG = "deprecated_user_path_config"
    USER_PRESET = "user_preset"
    CLI_ARGUMENT = "cli_argument"
    ENVIRONMENT_VARIABLE = "environment_variable"
    WEBUI_CONTROL = "webui_control"
    RUNTIME_OVERRIDE = "runtime_override"

    @property
    def priority(self) -> int:
        priorities = {
            ConfigSource.HARDCODED_FALLBACK: 0,
            ConfigSource.SCHEMA_DEFAULT: 10,
            ConfigSource.BUILTIN_PRESET: 20,
            ConfigSource.CONFIG_FILE: 30,
            ConfigSource.DEPRECATED_USER_PATH_CONFIG: 35,
            ConfigSource.USER_PRESET: 40,
            ConfigSource.CLI_ARGUMENT: 60,
            ConfigSource.ENVIRONMENT_VARIABLE: 70,
            ConfigSource.WEBUI_CONTROL: 80,
            ConfigSource.RUNTIME_OVERRIDE: 100,
        }
        return priorities[self]


class ConfigIssueType(Enum):
    UNKNOWN_KEY = "unknown_key"
    TYPE_ERROR = "type_error"
    VALIDATION_ERROR = "validation_error"
    PATH_NOT_FOUND = "path_not_found"
    DEPRECATED_FIELD = "deprecated_field"
    REPLACED_FIELD = "replaced_field"


@dataclass
class ConfigIssue:
    key: str
    issue_type: ConfigIssueType
    message: str
    source: Optional[ConfigSource] = None
    raw_value: Optional[Any] = None
    expected: Optional[str] = None


@dataclass
class ConfigValue:
    key: str
    value: Any
    source: ConfigSource
    source_detail: str = ""

    def __post_init__(self):
        if not self.source_detail:
            self.source_detail = self.source.value


@dataclass
class ConfigField:
    key: str
    default_value: Any
    expected_type: Union[type, Tuple[type, ...]]
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""
    category: str = "general"
    deprecated: bool = False
    deprecated_reason: str = ""
    replaced_by: Optional[str] = None
    is_path: bool = False
    path_is_dir: bool = False
    path_must_exist: bool = False
    path_auto_create: bool = False
    allow_array: bool = False
    cli_arg: Optional[str] = None
    cli_action: str = "store"
    ui_default: bool = True
    save_to_config: bool = True
    post_process: Optional[Callable[[Any], Any]] = None


@dataclass
class ConfigLoadResult:
    values: Dict[str, ConfigValue] = field(default_factory=dict)
    issues: List[ConfigIssue] = field(default_factory=list)
    sources_summary: Dict[str, List[str]] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.values:
            return self.values[key].value
        return default

    def get_source(self, key: str) -> Optional[ConfigSource]:
        if key in self.values:
            return self.values[key].source
        return None

    def issues_by_type(self, issue_type: ConfigIssueType) -> List[ConfigIssue]:
        return [i for i in self.issues if i.issue_type == issue_type]


class ConfigSchema:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._fields: Dict[str, ConfigField] = {}
        self._deprecated_aliases: Dict[str, str] = {}
        self._cli_args_to_keys: Dict[str, str] = {}

    def register_field(self, field: ConfigField):
        self._fields[field.key] = field
        if field.cli_arg:
            self._cli_args_to_keys[field.cli_arg] = field.key
        if field.deprecated and field.replaced_by:
            self._deprecated_aliases[field.key] = field.replaced_by

    def register_deprecated_alias(self, old_key: str, new_key: str):
        self._deprecated_aliases[old_key] = new_key

    @property
    def fields(self) -> Dict[str, ConfigField]:
        return self._fields

    @property
    def deprecated_aliases(self) -> Dict[str, str]:
        return self._deprecated_aliases

    def has_field(self, key: str) -> bool:
        return key in self._fields

    def get_field(self, key: str) -> Optional[ConfigField]:
        return self._fields.get(key)

    def resolve_deprecated(self, key: str) -> Tuple[str, bool]:
        if key in self._deprecated_aliases:
            return self._deprecated_aliases[key], True
        return key, False

    def _try_eval_env_var(self, value: str, expected_type=None):
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

    def _convert_type(self, value: Any, field: ConfigField) -> Tuple[Any, Optional[ConfigIssue]]:
        if value is None:
            return None, None

        expected = field.expected_type

        if expected is Any or isinstance(value, expected):
            return value, None

        try:
            if isinstance(expected, tuple):
                for t in expected:
                    try:
                        if t is bool and isinstance(value, str):
                            return value.lower() in ('true', '1', 'yes', 'on'), None
                        if t is int and isinstance(value, numbers.Number):
                            return int(value), None
                        if t is float and isinstance(value, numbers.Number):
                            return float(value), None
                        if t is str:
                            return str(value), None
                        return t(value), None
                    except (ValueError, TypeError):
                        continue
            else:
                if expected is bool and isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on'), None
                if expected is int and isinstance(value, numbers.Number):
                    return int(value), None
                if expected is float and isinstance(value, numbers.Number):
                    return float(value), None
                if expected is str:
                    return str(value), None
                return expected(value), None
        except (ValueError, TypeError) as e:
            pass

        issue = ConfigIssue(
            key=field.key,
            issue_type=ConfigIssueType.TYPE_ERROR,
            message=f"Type mismatch for '{field.key}': expected {expected}, got {type(value).__name__}",
            raw_value=value,
            expected=str(expected)
        )
        return value, issue

    def _validate_value(self, value: Any, field: ConfigField, source: ConfigSource) -> Tuple[bool, Optional[ConfigIssue]]:
        if field.validator is None:
            return True, None

        try:
            if field.validator(value):
                return True, None
        except Exception as e:
            pass

        issue = ConfigIssue(
            key=field.key,
            issue_type=ConfigIssueType.VALIDATION_ERROR,
            message=f"Validation failed for '{field.key}': value {json.dumps(value)} is not valid",
            source=source,
            raw_value=value
        )
        return False, issue

    def _process_path(self, value: Any, field: ConfigField, source: ConfigSource) -> Tuple[Any, List[ConfigIssue]]:
        issues = []
        if not field.is_path or value is None:
            return value, issues

        paths = value if field.allow_array and isinstance(value, list) else [value]
        processed_paths = []

        for path in paths:
            if not isinstance(path, str):
                issues.append(ConfigIssue(
                    key=field.key,
                    issue_type=ConfigIssueType.TYPE_ERROR,
                    message=f"Path value for '{field.key}' must be string, got {type(path).__name__}",
                    source=source,
                    raw_value=path
                ))
                continue

            if not os.path.isabs(path):
                path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))

            if field.path_auto_create:
                try:
                    if field.path_is_dir:
                        os.makedirs(path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                except OSError as e:
                    issues.append(ConfigIssue(
                        key=field.key,
                        issue_type=ConfigIssueType.PATH_NOT_FOUND,
                        message=f"Could not create path '{path}' for '{field.key}': {e}",
                        source=source,
                        raw_value=value
                    ))

            if field.path_must_exist:
                check_func = os.path.isdir if field.path_is_dir else os.path.exists
                if not check_func(path):
                    issues.append(ConfigIssue(
                        key=field.key,
                        issue_type=ConfigIssueType.PATH_NOT_FOUND,
                        message=f"Path '{path}' for '{field.key}' does not exist",
                        source=source,
                        raw_value=value
                    ))

            processed_paths.append(path)

        if field.allow_array:
            return processed_paths, issues
        else:
            return processed_paths[0] if processed_paths else value, issues

    def apply_value(self, result: ConfigLoadResult, key: str, value: Any,
                    source: ConfigSource, source_detail: str = "",
                    allow_deprecated: bool = True) -> List[ConfigIssue]:
        issues = []

        resolved_key, was_deprecated = self.resolve_deprecated(key)
        if was_deprecated and allow_deprecated:
            issues.append(ConfigIssue(
                key=key,
                issue_type=ConfigIssueType.REPLACED_FIELD,
                message=f"Config key '{key}' is deprecated, using '{resolved_key}' instead",
                source=source,
                raw_value=value,
                expected=resolved_key
            ))
            key = resolved_key

        if not self.has_field(key):
            issues.append(ConfigIssue(
                key=key,
                issue_type=ConfigIssueType.UNKNOWN_KEY,
                message=f"Unknown config key '{key}' - will be ignored",
                source=source,
                raw_value=value
            ))
            return issues

        field = self._fields[key]

        if field.deprecated and not was_deprecated:
            issues.append(ConfigIssue(
                key=key,
                issue_type=ConfigIssueType.DEPRECATED_FIELD,
                message=f"Config key '{key}' is deprecated: {field.deprecated_reason}",
                source=source,
                raw_value=value
            ))

        if source == ConfigSource.ENVIRONMENT_VARIABLE and isinstance(value, str):
            value = self._try_eval_env_var(value, field.expected_type)

        converted_value, type_issue = self._convert_type(value, field)
        if type_issue:
            type_issue.source = source
            issues.append(type_issue)

        is_valid, validation_issue = self._validate_value(converted_value, field, source)
        if validation_issue:
            issues.append(validation_issue)

        if not is_valid or type_issue:
            return issues

        if field.is_path:
            converted_value, path_issues = self._process_path(converted_value, field, source)
            issues.extend(path_issues)

        if field.post_process:
            try:
                converted_value = field.post_process(converted_value)
            except Exception as e:
                issues.append(ConfigIssue(
                    key=key,
                    issue_type=ConfigIssueType.VALIDATION_ERROR,
                    message=f"Post-processing failed for '{key}': {e}",
                    source=source,
                    raw_value=converted_value
                ))

        current = result.values.get(key)
        if current is None or source.priority >= current.source.priority:
            result.values[key] = ConfigValue(
                key=key,
                value=converted_value,
                source=source,
                source_detail=source_detail if source_detail else source.value
            )

            source_key = source.value
            if source_key not in result.sources_summary:
                result.sources_summary[source_key] = []
            if key not in result.sources_summary[source_key]:
                result.sources_summary[source_key].append(key)

        return issues

    def apply_defaults(self, result: ConfigLoadResult):
        for key, field in self._fields.items():
            if key not in result.values:
                default_val = field.default_value

                if field.is_path:
                    default_val, _ = self._process_path(default_val, field, ConfigSource.SCHEMA_DEFAULT)

                if field.post_process:
                    try:
                        default_val = field.post_process(default_val)
                    except:
                        pass

                result.values[key] = ConfigValue(
                    key=key,
                    value=default_val,
                    source=ConfigSource.SCHEMA_DEFAULT,
                    source_detail="schema_default"
                )
                source_key = ConfigSource.SCHEMA_DEFAULT.value
                if source_key not in result.sources_summary:
                    result.sources_summary[source_key] = []
                result.sources_summary[source_key].append(key)

    def load_dict(self, result: ConfigLoadResult, data: Dict[str, Any],
                  source: ConfigSource, source_detail: str = "") -> List[ConfigIssue]:
        all_issues = []
        for key, value in data.items():
            issues = self.apply_value(result, key, value, source, source_detail)
            all_issues.extend(issues)
        return all_issues

    def load_env(self, result: ConfigLoadResult) -> List[ConfigIssue]:
        all_issues = []
        for key in self._fields:
            env_val = os.getenv(key)
            if env_val is not None:
                issues = self.apply_value(
                    result, key, env_val,
                    ConfigSource.ENVIRONMENT_VARIABLE,
                    f"env:{key}"
                )
                all_issues.extend(issues)
        return all_issues

    def load_cli_args(self, result: ConfigLoadResult, cli_args: Any) -> List[ConfigIssue]:
        all_issues = []
        args_dict = vars(cli_args) if not isinstance(cli_args, dict) else cli_args

        for cli_arg, key in self._cli_args_to_keys.items():
            attr_name = cli_arg.lstrip('-').replace('-', '_')
            if attr_name in args_dict and args_dict[attr_name] is not None:
                value = args_dict[attr_name]
                field = self._fields.get(key)
                if field and field.cli_action == 'store_true':
                    pass
                issues = self.apply_value(
                    result, key, value,
                    ConfigSource.CLI_ARGUMENT,
                    f"cli:{cli_arg}"
                )
                all_issues.extend(issues)

        return all_issues

    def format_summary(self, result: ConfigLoadResult) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("CONFIGURATION SOURCE SUMMARY")
        lines.append("=" * 70)

        source_order = sorted(
            result.sources_summary.keys(),
            key=lambda s: ConfigSource(s).priority if s in [e.value for e in ConfigSource] else -1
        )

        for src in source_order:
            keys = result.sources_summary.get(src, [])
            if keys:
                lines.append(f"\n[{src.upper()}] ({len(keys)} keys)")
                for key in sorted(keys):
                    val = result.values.get(key)
                    if val:
                        display_val = json.dumps(val.value) if not isinstance(val.value, (list, dict)) or len(str(val.value)) < 80 else f"<{type(val.value).__name__} len={len(val.value)}>"
                        lines.append(f"  {key} = {display_val}")

        issues = result.issues
        if issues:
            lines.append("\n" + "=" * 70)
            lines.append(f"CONFIGURATION ISSUES ({len(issues)})")
            lines.append("=" * 70)

            by_type: Dict[ConfigIssueType, List[ConfigIssue]] = {}
            for issue in issues:
                if issue.issue_type not in by_type:
                    by_type[issue.issue_type] = []
                by_type[issue.issue_type].append(issue)

            for issue_type, type_issues in by_type.items():
                icon = {
                    ConfigIssueType.UNKNOWN_KEY: "?",
                    ConfigIssueType.TYPE_ERROR: "!",
                    ConfigIssueType.VALIDATION_ERROR: "!",
                    ConfigIssueType.PATH_NOT_FOUND: "X",
                    ConfigIssueType.DEPRECATED_FIELD: "~",
                    ConfigIssueType.REPLACED_FIELD: ">",
                }.get(issue_type, "*")

                lines.append(f"\n  [{icon}] {issue_type.value.upper()} ({len(type_issues)})")
                for issue in type_issues:
                    src_str = f" [{issue.source.value}]" if issue.source else ""
                    lines.append(f"    - {issue.key}{src_str}: {issue.message}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


def build_fooocus_schema(root_dir: str) -> ConfigSchema:
    import modules.flags as flags
    import modules.sdxl_styles as sdxl_styles

    schema = ConfigSchema(root_dir)

    _root_dir = root_dir
    _modules_dir = os.path.join(_root_dir, 'modules')

    schema.register_deprecated_alias('modelfile_path', 'path_checkpoints')
    schema.register_deprecated_alias('lorafile_path', 'path_loras')
    schema.register_deprecated_alias('embeddings_path', 'path_embeddings')
    schema.register_deprecated_alias('vae_approx_path', 'path_vae_approx')
    schema.register_deprecated_alias('upscale_models_path', 'path_upscale_models')
    schema.register_deprecated_alias('inpaint_models_path', 'path_inpaint')
    schema.register_deprecated_alias('controlnet_models_path', 'path_controlnet')
    schema.register_deprecated_alias('clip_vision_models_path', 'path_clip_vision')
    schema.register_deprecated_alias('fooocus_expansion_path', 'path_fooocus_expansion')
    schema.register_deprecated_alias('temp_outputs_path', 'path_outputs')

    path_fields = [
        ConfigField('path_checkpoints', ['../models/checkpoints/'], list,
                    is_path=True, path_is_dir=True, path_auto_create=True, allow_array=True,
                    category='paths', description='Checkpoint model directories',
                    validator=lambda x: isinstance(x, list) and all(isinstance(d, str) for d in x)),
        ConfigField('path_loras', ['../models/loras/'], list,
                    is_path=True, path_is_dir=True, path_auto_create=True, allow_array=True,
                    category='paths', description='LoRA model directories',
                    validator=lambda x: isinstance(x, list) and all(isinstance(d, str) for d in x)),
        ConfigField('path_embeddings', '../models/embeddings/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='Embeddings directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_vae_approx', '../models/vae_approx/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='VAE approx directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_vae', '../models/vae/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='VAE directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_upscale_models', '../models/upscale_models/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='Upscale models directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_inpaint', '../models/inpaint/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='Inpaint models directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_controlnet', '../models/controlnet/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='ControlNet models directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_clip_vision', '../models/clip_vision/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='CLIP vision directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_fooocus_expansion', '../models/prompt_expansion/fooocus_expansion', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='Fooocus expansion directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_wildcards', '../wildcards/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='Wildcards directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_safety_checker', '../models/safety_checker/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='Safety checker directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_sam', '../models/sam/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='SAM models directory',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('path_outputs', '../outputs/', str,
                    is_path=True, path_is_dir=True, path_auto_create=True,
                    category='paths', description='Outputs directory',
                    validator=lambda x: isinstance(x, str),
                    cli_arg='output-path'),
    ]

    default_temp_path = os.path.join(tempfile.gettempdir(), 'fooocus')
    path_fields.append(ConfigField('temp_path', default_temp_path, str,
                                   is_path=True, path_is_dir=True, path_auto_create=True,
                                   category='paths', description='Temp directory',
                                   validator=lambda x: isinstance(x, str),
                                   cli_arg='temp-path'))

    for f in path_fields:
        schema.register_field(f)

    def validate_loras(x):
        if not isinstance(x, list):
            return False
        for y in x:
            if not (
                (len(y) == 3 and isinstance(y[0], bool) and isinstance(y[1], str) and isinstance(y[2], numbers.Number))
                or (len(y) == 2 and isinstance(y[0], str) and isinstance(y[1], numbers.Number))
            ):
                return False
        return True

    def post_process_loras(x):
        if not isinstance(x, list):
            return x
        return [(y[0], y[1], y[2]) if len(y) == 3 else (True, y[0], y[1]) for y in x]

    model_fields = [
        ConfigField('default_model', 'model.safetensors', str,
                    category='models', description='Base checkpoint model',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('previous_default_models', [], list,
                    category='models', description='Previous default models history',
                    validator=lambda x: isinstance(x, list) and all(isinstance(k, str) for k in x)),
        ConfigField('default_refiner', 'None', str,
                    category='models', description='Refiner model',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('default_refiner_switch', 0.8, numbers.Number,
                    category='models', description='Refiner switch point',
                    validator=lambda x: isinstance(x, numbers.Number) and 0 <= x <= 1),
        ConfigField('default_loras_min_weight', -2, numbers.Number,
                    category='models', description='Minimum LoRA weight',
                    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10),
        ConfigField('default_loras_max_weight', 2, numbers.Number,
                    category='models', description='Maximum LoRA weight',
                    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10),
        ConfigField('default_loras', [
            [True, "None", 1.0], [True, "None", 1.0], [True, "None", 1.0],
            [True, "None", 1.0], [True, "None", 1.0]
        ], list,
                    category='models', description='Default LoRA list',
                    validator=validate_loras,
                    post_process=post_process_loras),
        ConfigField('default_max_lora_number', 5, int,
                    category='models', description='Maximum number of LoRA slots',
                    validator=lambda x: isinstance(x, int) and x >= 1),
        ConfigField('default_vae', flags.default_vae, str,
                    category='models', description='VAE override',
                    validator=lambda x: isinstance(x, str)),
    ]

    for f in model_fields:
        schema.register_field(f)

    sampling_fields = [
        ConfigField('default_cfg_scale', 7.0, numbers.Number,
                    category='sampling', description='CFG Scale',
                    validator=lambda x: isinstance(x, numbers.Number)),
        ConfigField('default_sample_sharpness', 2.0, numbers.Number,
                    category='sampling', description='Sample sharpness',
                    validator=lambda x: isinstance(x, numbers.Number)),
        ConfigField('default_cfg_tsnr', 7.0, numbers.Number,
                    category='sampling', description='Adaptive CFG (TSNR)',
                    validator=lambda x: isinstance(x, numbers.Number)),
        ConfigField('default_clip_skip', 2, int,
                    category='sampling', description='CLIP skip layers',
                    validator=lambda x: isinstance(x, int) and 1 <= x <= flags.clip_skip_max),
        ConfigField('default_sampler', 'dpmpp_2m_sde_gpu', str,
                    category='sampling', description='Sampler name',
                    validator=lambda x: x in flags.sampler_list),
        ConfigField('default_scheduler', 'karras', str,
                    category='sampling', description='Scheduler name',
                    validator=lambda x: x in flags.scheduler_list),
        ConfigField('default_performance', flags.Performance.SPEED.value, str,
                    category='sampling', description='Performance mode',
                    validator=lambda x: x in flags.Performance.values(),
                    cli_arg='preset'),
        ConfigField('default_overwrite_step', -1, int,
                    category='sampling', description='Overwrite step count (-1 = auto)',
                    validator=lambda x: isinstance(x, int)),
        ConfigField('default_overwrite_switch', -1, int,
                    category='sampling', description='Overwrite switch step (-1 = auto)',
                    validator=lambda x: isinstance(x, int)),
        ConfigField('default_overwrite_upscale', -1, numbers.Number,
                    category='sampling', description='Overwrite upscale (-1 = auto)',
                    validator=lambda x: isinstance(x, numbers.Number)),
    ]

    for f in sampling_fields:
        schema.register_field(f)

    ConfigField('default_prompt', '', str,
                category='prompts', description='Default positive prompt',
                validator=lambda x: isinstance(x, str))
    prompt_fields = [
        ConfigField('default_prompt', '', str,
                    category='prompts', description='Default positive prompt',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('default_prompt_negative', '', str,
                    category='prompts', description='Default negative prompt',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('default_styles', ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"], list,
                    category='prompts', description='Default styles',
                    validator=lambda x: isinstance(x, list) and all(y in sdxl_styles.legal_style_names for y in x)),
    ]

    for f in prompt_fields:
        schema.register_field(f)

    ui_checkbox_fields = [
        ConfigField('default_image_prompt_checkbox', False, bool,
                    category='ui', description='Show image prompt section',
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('default_enhance_checkbox', False, bool,
                    category='ui', description='Show enhance section',
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('default_advanced_checkbox', False, bool,
                    category='ui', description='Show advanced settings',
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('default_developer_debug_mode_checkbox', False, bool,
                    category='ui', description='Developer debug mode',
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('default_image_prompt_advanced_checkbox', False, bool,
                    category='ui', description='Show advanced image prompt options',
                    validator=lambda x: isinstance(x, bool)),
    ]

    for f in ui_checkbox_fields:
        schema.register_field(f)

    output_fields = [
        ConfigField('default_max_image_number', 32, int,
                    category='output', description='Max images per batch',
                    validator=lambda x: isinstance(x, int) and x >= 1),
        ConfigField('default_output_format', 'png', str,
                    category='output', description='Output image format',
                    validator=lambda x: x in flags.OutputFormat.list()),
        ConfigField('default_image_number', 2, int,
                    category='output', description='Default image count',
                    validator=lambda x: isinstance(x, int) and 1 <= x <= 32),
        ConfigField('default_save_metadata_to_images', False, bool,
                    category='output', description='Save metadata to images',
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('default_metadata_scheme', flags.MetadataScheme.FOOOCUS.value, str,
                    category='output', description='Metadata scheme',
                    validator=lambda x: x in [y[1] for y in flags.metadata_scheme]),
        ConfigField('metadata_created_by', '', str,
                    category='output', description='Created by metadata field',
                    validator=lambda x: isinstance(x, str)),
        ConfigField('default_save_only_final_enhanced_image', False, bool,
                    category='output', description='Save only final enhanced image',
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('default_black_out_nsfw', False, bool,
                    category='output', description='Black out NSFW content',
                    validator=lambda x: isinstance(x, bool)),
    ]

    for f in output_fields:
        schema.register_field(f)

    aspect_ratio_default = '1152*896' if '1152*896' in flags.sdxl_aspect_ratios else flags.sdxl_aspect_ratios[0]
    image_fields = [
        ConfigField('available_aspect_ratios', flags.sdxl_aspect_ratios, list,
                    category='image', description='Available aspect ratios',
                    validator=lambda x: isinstance(x, list) and all('*' in v for v in x) and len(x) > 1,
                    save_to_config=False),
        ConfigField('default_aspect_ratio', aspect_ratio_default, str,
                    category='image', description='Default aspect ratio',
                    validator=lambda x: x in flags.sdxl_aspect_ratios),
        ConfigField('default_inpaint_engine_version', 'v2.6', str,
                    category='image', description='Inpaint engine version',
                    validator=lambda x: x in flags.inpaint_engine_versions),
    ]

    for f in image_fields:
        schema.register_field(f)

    ip_fields = [
        ConfigField('default_controlnet_image_count', 4, int,
                    category='image_prompt', description='Number of image prompt slots',
                    validator=lambda x: isinstance(x, int) and x > 0),
        ConfigField('default_uov_method', flags.disabled, str,
                    category='image_prompt', description='Upscale or Variation method',
                    validator=lambda x: x in flags.uov_list),
    ]

    for i in range(1, 5):
        default_end, default_weight = flags.default_parameters[flags.default_ip]
        ip_fields.extend([
            ConfigField(f'default_ip_image_{i}', 'None', str,
                        category='image_prompt', description=f'Image prompt image #{i}',
                        validator=lambda x: x == 'None' or (isinstance(x, str)),
                        post_process=lambda x: None if x == 'None' else x),
            ConfigField(f'default_ip_type_{i}', flags.default_ip, str,
                        category='image_prompt', description=f'Image prompt type #{i}',
                        validator=lambda x: x in flags.ip_list),
            ConfigField(f'default_ip_stop_at_{i}', default_end, float,
                        category='image_prompt', description=f'Image prompt stop at #{i}',
                        validator=lambda x: isinstance(x, float) and 0 <= x <= 1),
            ConfigField(f'default_ip_weight_{i}', default_weight, float,
                        category='image_prompt', description=f'Image prompt weight #{i}',
                        validator=lambda x: isinstance(x, float) and 0 <= x <= 2),
        ])

    for f in ip_fields:
        schema.register_field(f)

    inpaint_fields = [
        ConfigField('default_inpaint_advanced_masking_checkbox', False, bool,
                    category='inpaint', description='Enable advanced masking',
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('default_inpaint_method', flags.inpaint_option_default, str,
                    category='inpaint', description='Inpaint method',
                    validator=lambda x: x in flags.inpaint_options),
        ConfigField('default_invert_mask_checkbox', False, bool,
                    category='inpaint', description='Invert mask',
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('default_inpaint_mask_model', 'isnet-general-use', str,
                    category='inpaint', description='Mask generation model',
                    validator=lambda x: x in flags.inpaint_mask_models),
        ConfigField('default_enhance_inpaint_mask_model', 'sam', str,
                    category='inpaint', description='Enhance mask model',
                    validator=lambda x: x in flags.inpaint_mask_models),
        ConfigField('default_inpaint_mask_cloth_category', 'full', str,
                    category='inpaint', description='Cloth seg category',
                    validator=lambda x: x in flags.inpaint_mask_cloth_category),
        ConfigField('default_inpaint_mask_sam_model', 'vit_b', str,
                    category='inpaint', description='SAM model variant',
                    validator=lambda x: x in flags.inpaint_mask_sam_model),
        ConfigField('default_sam_max_detections', 0, int,
                    category='inpaint', description='Max SAM detections (0 = all)',
                    validator=lambda x: isinstance(x, int) and 0 <= x <= 10),
    ]

    for f in inpaint_fields:
        schema.register_field(f)

    describe_fields = [
        ConfigField('default_describe_apply_prompts_checkbox', True, bool,
                    category='describe', description='Apply prompts from describe',
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('default_describe_content_type', [flags.describe_type_photo], list,
                    category='describe', description='Describe content type filter',
                    validator=lambda x: all(k in flags.describe_types for k in x)),
    ]

    for f in describe_fields:
        schema.register_field(f)

    enhance_fields = [
        ConfigField('default_enhance_tabs', 3, int,
                    category='enhance', description='Number of enhance tabs',
                    validator=lambda x: isinstance(x, int) and 1 <= x <= 5),
        ConfigField('default_enhance_uov_method', flags.disabled, str,
                    category='enhance', description='Enhance UOV method',
                    validator=lambda x: x in flags.uov_list),
        ConfigField('default_enhance_uov_processing_order', flags.enhancement_uov_before, str,
                    category='enhance', description='Enhance UOV processing order',
                    validator=lambda x: x in flags.enhancement_uov_processing_order),
        ConfigField('default_enhance_uov_prompt_type', flags.enhancement_uov_prompt_type_original, str,
                    category='enhance', description='Enhance UOV prompt source',
                    validator=lambda x: x in flags.enhancement_uov_prompt_types),
    ]

    for f in enhance_fields:
        schema.register_field(f)

    example_fields = [
        ConfigField('example_inpaint_prompts', [
            'highly detailed face', 'detailed girl face', 'detailed man face',
            'detailed hand', 'beautiful eyes'
        ], list,
                    category='examples', description='Example inpaint prompts',
                    validator=lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x),
                    post_process=lambda x: [[v] for v in x] if x and isinstance(x[0], str) else x),
        ConfigField('example_enhance_detection_prompts', [
            'face', 'eye', 'mouth', 'hair', 'hand', 'body'
        ], list,
                    category='examples', description='Example enhance detection prompts',
                    validator=lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x),
                    post_process=lambda x: [[v] for v in x] if x and isinstance(x[0], str) else x),
    ]

    for f in example_fields:
        schema.register_field(f)

    input_tab_fields = [
        ConfigField('default_selected_image_input_tab_id', flags.default_input_image_tab, str,
                    category='ui', description='Default input image tab',
                    validator=lambda x: x in flags.input_image_tab_ids),
    ]

    for f in input_tab_fields:
        schema.register_field(f)

    download_fields = [
        ConfigField('checkpoint_downloads', {}, dict,
                    category='downloads', description='Checkpoint URL map',
                    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())),
        ConfigField('lora_downloads', {}, dict,
                    category='downloads', description='LoRA URL map',
                    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())),
        ConfigField('embeddings_downloads', {}, dict,
                    category='downloads', description='Embedding URL map',
                    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())),
        ConfigField('vae_downloads', {}, dict,
                    category='downloads', description='VAE URL map',
                    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())),
    ]

    for f in download_fields:
        schema.register_field(f)

    misc_fields = [
        ConfigField('temp_path_cleanup_on_launch', True, bool,
                    category='misc', description='Clean temp path on launch',
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('wildcards_max_bfs_depth', 64, int,
                    category='misc', description='Wildcards max BFS depth',
                    validator=lambda x: isinstance(x, int) and x > 0,
                    save_to_config=False),
    ]

    for f in misc_fields:
        schema.register_field(f)

    return schema
