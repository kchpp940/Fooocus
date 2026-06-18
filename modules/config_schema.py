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


class ConfigSeverity(Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"

    @property
    def priority(self) -> int:
        return {
            ConfigSeverity.INFO: 0,
            ConfigSeverity.WARN: 10,
            ConfigSeverity.ERROR: 20,
        }[self]


ISSUE_SEVERITY_MAP: Dict[ConfigIssueType, ConfigSeverity] = {
    ConfigIssueType.UNKNOWN_KEY: ConfigSeverity.WARN,
    ConfigIssueType.TYPE_ERROR: ConfigSeverity.ERROR,
    ConfigIssueType.VALIDATION_ERROR: ConfigSeverity.ERROR,
    ConfigIssueType.PATH_NOT_FOUND: ConfigSeverity.ERROR,
    ConfigIssueType.DEPRECATED_FIELD: ConfigSeverity.WARN,
    ConfigIssueType.REPLACED_FIELD: ConfigSeverity.WARN,
}


@dataclass
class ConfigIssue:
    key: str
    issue_type: ConfigIssueType
    message: str
    source: Optional[ConfigSource] = None
    raw_value: Optional[Any] = None
    expected: Optional[str] = None

    @property
    def severity(self) -> ConfigSeverity:
        return ISSUE_SEVERITY_MAP.get(self.issue_type, ConfigSeverity.WARN)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'type': self.issue_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'source': self.source.value if self.source else None,
            'raw_value': None if self.raw_value is None else (str(self.raw_value)[:200]),
            'expected': self.expected,
        }


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
    preset_binding: Optional[str] = None
    ui_to_config_transform: Optional[Callable[[Any], Any]] = None


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

    def issues_by_severity(self, severity: ConfigSeverity) -> List[ConfigIssue]:
        return [i for i in self.issues if i.severity == severity]

    @property
    def errors(self) -> List[ConfigIssue]:
        return self.issues_by_severity(ConfigSeverity.ERROR)

    @property
    def warnings(self) -> List[ConfigIssue]:
        return self.issues_by_severity(ConfigSeverity.WARN)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


@dataclass
class ConfigDiagnostics:
    schema_version: str = "1.0"
    config_ok: bool = True
    total_issues: int = 0
    error_count: int = 0
    warn_count: int = 0
    info_count: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    source_counts: Dict[str, int] = field(default_factory=dict)
    write_back_preview: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'ok': self.config_ok,
            'exit_code': 0 if self.config_ok else 3,
            'counts': {
                'total': self.total_issues,
                'errors': self.error_count,
                'warnings': self.warn_count,
                'info': self.info_count,
            },
            'issues': self.issues,
            'sources': self.source_counts,
            'write_back_preview': self.write_back_preview,
            'summary': self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_result(cls, schema: 'ConfigSchema', result: ConfigLoadResult) -> 'ConfigDiagnostics':
        errors = result.errors
        warnings = result.warnings
        infos = result.issues_by_severity(ConfigSeverity.INFO)

        source_counts = {}
        for issue in result.issues:
            src = issue.source.value if issue.source else 'unknown'
            source_counts[src] = source_counts.get(src, 0) + 1

        preset_data = schema.export_for_preset(result, include_schema_defaults=False)
        cfg_data = schema.export_for_config_file(result, include_schema_defaults=False)

        return cls(
            schema_version="1.0",
            config_ok=not result.has_errors,
            total_issues=len(result.issues),
            error_count=len(errors),
            warn_count=len(warnings),
            info_count=len(infos),
            issues=[i.to_dict() for i in result.issues],
            source_counts=source_counts,
            write_back_preview={
                'preset_keys': list(preset_data.keys()),
                'preset_count': len(preset_data),
                'config_file_keys': list(cfg_data.keys()),
                'config_file_count': len(cfg_data),
            },
            summary={
                'total_fields': len(schema.fields),
                'loaded_fields': len(result.values),
                'sources': {k: len(v) for k, v in result.sources_summary.items()},
            },
        )


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
                        if t is int and isinstance(value, (numbers.Number, str)):
                            return int(value), None
                        if t is float and isinstance(value, (numbers.Number, str)):
                            return float(value), None
                        if t is str:
                            return str(value), None
                        return t(value), None
                    except (ValueError, TypeError):
                        continue
            else:
                if expected is bool and isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on'), None
                if expected is int and isinstance(value, (numbers.Number, str)):
                    return int(value), None
                if expected is float and isinstance(value, (numbers.Number, str)):
                    return float(value), None
                if expected is numbers.Number and isinstance(value, str):
                    try:
                        return float(value), None
                    except (ValueError, TypeError):
                        try:
                            return int(value), None
                        except (ValueError, TypeError):
                            pass
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
            field = self._fields.get(key)
            if not field:
                continue

            if attr_name in args_dict:
                value = args_dict[attr_name]
                if field.cli_action == 'store_true':
                    if value is True:
                        issues = self.apply_value(
                            result, key, value,
                            ConfigSource.CLI_ARGUMENT,
                            f"cli:{cli_arg}"
                        )
                        all_issues.extend(issues)
                elif value is not None:
                    issues = self.apply_value(
                        result, key, value,
                        ConfigSource.CLI_ARGUMENT,
                        f"cli:{cli_arg}"
                    )
                    all_issues.extend(issues)

        return all_issues

    def get_value(self, result: ConfigLoadResult, key: str, default: Any = None) -> Any:
        val = result.get(key, None)
        if val is not None:
            return val
        field = self.get_field(key)
        if field:
            return field.default_value
        return default

    def get_ui_default(self, result: ConfigLoadResult, key: str, default: Any = None) -> Any:
        val = result.get(key, None)
        if val is not None:
            return val
        field = self.get_field(key)
        if field:
            return field.default_value
        return default

    def get_source(self, result: ConfigLoadResult, key: str) -> Optional[ConfigSource]:
        return result.get_source(key)

    def get_source_detail(self, result: ConfigLoadResult, key: str) -> str:
        if key in result.values:
            return result.values[key].source_detail
        return "schema_default"

    def get_cli_fields(self) -> List[ConfigField]:
        return [f for f in self._fields.values() if f.cli_arg is not None]

    def get_preset_fields(self) -> List[ConfigField]:
        return [f for f in self._fields.values() if f.save_to_config and not f.is_path and not f.key.startswith('cli_')]

    def get_config_file_fields(self) -> List[ConfigField]:
        return [f for f in self._fields.values() if f.save_to_config and not f.key.startswith('cli_')]

    def import_and_validate(self, data: Dict[str, Any], source: ConfigSource,
                            source_detail: str = "") -> Tuple[Dict[str, Any], List[ConfigIssue]]:
        cleaned = {}
        issues = []

        for raw_key, raw_val in data.items():
            key, is_deprecated = self.resolve_deprecated(raw_key)

            if is_deprecated:
                issues.append(ConfigIssue(
                    issue_type=ConfigIssueType.DEPRECATED_FIELD,
                    key=raw_key,
                    message=f"Deprecated field '{raw_key}' - automatically mapped to '{key}'",
                    source=source,
                    raw_value=raw_val
                ))

            if not self.has_field(key):
                issues.append(ConfigIssue(
                    issue_type=ConfigIssueType.UNKNOWN_KEY,
                    key=raw_key,
                    message=f"Unknown config key '{raw_key}' - will be ignored during save",
                    source=source,
                    raw_value=raw_val
                ))
                continue

            field = self.get_field(key)
            if not field.save_to_config:
                issues.append(ConfigIssue(
                    issue_type=ConfigIssueType.UNKNOWN_KEY,
                    key=raw_key,
                    message=f"Key '{raw_key}' is marked as save_to_config=False - will be filtered out",
                    source=source,
                    raw_value=raw_val
                ))
                continue

            converted, type_issue = self._convert_type(raw_val, field)
            if type_issue:
                issues.append(type_issue)
                continue

            is_valid, validation_issue = self._validate_value(converted, field, source)
            if not is_valid and validation_issue:
                issues.append(validation_issue)
                continue

            cleaned[key] = converted

        return cleaned, issues

    def export_for_preset(self, result: ConfigLoadResult, include_schema_defaults: bool = False) -> Dict[str, Any]:
        exported = {}
        for field in self.get_preset_fields():
            cv = result.values.get(field.key)
            if cv is None:
                if include_schema_defaults and field.default_value is not None:
                    exported[field.key] = field.default_value
                continue
            if include_schema_defaults or cv.source not in (ConfigSource.SCHEMA_DEFAULT, ConfigSource.HARDCODED_FALLBACK):
                exported[field.key] = cv.value
        return exported

    def export_for_config_file(self, result: ConfigLoadResult, include_schema_defaults: bool = False) -> Dict[str, Any]:
        exported = {}
        for field in self.get_config_file_fields():
            cv = result.values.get(field.key)
            if cv is None:
                if include_schema_defaults and field.default_value is not None:
                    exported[field.key] = field.default_value
                continue
            if include_schema_defaults or cv.source not in (ConfigSource.SCHEMA_DEFAULT, ConfigSource.HARDCODED_FALLBACK, ConfigSource.BUILTIN_PRESET):
                exported[field.key] = cv.value
        return exported

    def get_export_metadata(self, result: ConfigLoadResult) -> Dict[str, Dict[str, Any]]:
        metadata = {}
        for key, cv in result.values.items():
            field = self.get_field(key)
            if field is None:
                continue
            metadata[key] = {
                'source': cv.source.value,
                'source_detail': cv.source_detail,
                'save_to_config': field.save_to_config,
                'in_preset_export': field.save_to_config and not field.is_path and not key.startswith('cli_'),
                'in_config_file_export': field.save_to_config and not key.startswith('cli_'),
                'is_schema_default': cv.source in (ConfigSource.SCHEMA_DEFAULT, ConfigSource.HARDCODED_FALLBACK),
            }
        return metadata

    def get_preset_binding_map(self) -> Dict[str, Dict[str, Any]]:
        binding_map = {}
        for field in self.get_preset_fields():
            if field.preset_binding:
                binding_map[field.preset_binding] = {
                    'key': field.key,
                    'expected_type': field.expected_type,
                    'transform': field.ui_to_config_transform,
                    'default_value': field.default_value,
                }
        return binding_map

    def export_from_ui_values(self, ui_values: Dict[str, Any]) -> Dict[str, Any]:
        binding_map = self.get_preset_binding_map()
        raw_data = {}

        for binding_name, binding_info in binding_map.items():
            if binding_name in ui_values:
                val = ui_values[binding_name]
                transform = binding_info.get('transform')
                if transform is not None:
                    try:
                        val = transform(val)
                    except Exception:
                        pass
                raw_data[binding_info['key']] = val

        for field in self.get_preset_fields():
            if field.key not in raw_data and field.key in ('checkpoint_downloads', 'embeddings_downloads', 'lora_downloads', 'vae_downloads'):
                raw_data[field.key] = {}

        cleaned, _ = self.import_and_validate(raw_data, ConfigSource.USER_PRESET, 'ui_preset_save')
        return cleaned

    def build_diagnostics(self, result: ConfigLoadResult) -> ConfigDiagnostics:
        return ConfigDiagnostics.from_result(self, result)

    def format_diagnostics_terminal(self, result: ConfigLoadResult, show_warnings_only: bool = False) -> str:
        lines = []
        errors = result.errors
        warnings = result.warnings

        if show_warnings_only and not errors and not warnings:
            return ""

        if not errors and not warnings:
            lines.append("[Config Diagnostics] ✓ No issues found.")
            return "\n".join(lines)

        lines.append("=" * 70)
        lines.append("CONFIG DIAGNOSTICS")
        lines.append("=" * 70)

        if errors:
            lines.append(f"\n[ERROR] {len(errors)} critical issue(s) - may break startup")
            for issue in errors:
                src = f" [{issue.source.value}]" if issue.source else ""
                lines.append(f"  ✗ {issue.key}{src}: {issue.message}")

        if warnings and not show_warnings_only:
            lines.append(f"\n[WARN] {len(warnings)} warning(s) - safe to ignore but recommended to fix")
            for issue in warnings:
                src = f" [{issue.source.value}]" if issue.source else ""
                lines.append(f"  ! {issue.key}{src}: {issue.message}")

        if errors:
            lines.append(f"\nExit code: 3 ({len(errors)} errors, {len(warnings)} warnings)")
        elif warnings:
            lines.append(f"\nExit code: 0 ({len(warnings)} warnings)")

        return "\n".join(lines)

    @classmethod
    def standalone_check(cls, root_dir: str, output_json: bool = False,
                         include_config_txt: bool = True,
                         include_user_presets: bool = False,
                         include_deprecated_user_path: bool = False) -> Tuple[ConfigDiagnostics, ConfigLoadResult]:
        schema = cls(root_dir)
        result = ConfigLoadResult()

        builtin_preset_path = os.path.join(root_dir, 'presets', 'default.json')
        try:
            with open(builtin_preset_path, "r", encoding="utf-8") as f:
                issues = schema.load_dict(result, json.load(f),
                                          ConfigSource.BUILTIN_PRESET, 'preset:default.json')
                result.issues.extend(issues)
        except Exception as e:
            from .config_schema import ConfigIssue, ConfigIssueType, ConfigSource
            result.issues.append(ConfigIssue(
                key='default_preset',
                issue_type=ConfigIssueType.UNKNOWN_KEY,
                message=f'Failed to load default preset: {e}',
                source=ConfigSource.BUILTIN_PRESET,
            ))

        config_txt_path = os.path.join(root_dir, 'config.txt')
        if include_config_txt and os.path.exists(config_txt_path):
            try:
                with open(config_txt_path, "r", encoding="utf-8") as f:
                    issues = schema.load_dict(result, json.load(f),
                                              ConfigSource.CONFIG_FILE,
                                              f'config:{os.path.basename(config_txt_path)}')
                    result.issues.extend(issues)
            except json.JSONDecodeError as e:
                from .config_schema import ConfigIssue, ConfigIssueType, ConfigSource
                result.issues.append(ConfigIssue(
                    key='config.txt',
                    issue_type=ConfigIssueType.VALIDATION_ERROR,
                    message=f'Invalid JSON in config.txt: {e}',
                    source=ConfigSource.CONFIG_FILE,
                ))

        if include_deprecated_user_path:
            upc = os.path.join(root_dir, 'user_path_config.txt')
            if os.path.exists(upc):
                try:
                    with open(upc, "r", encoding="utf-8") as f:
                        issues = schema.load_dict(result, json.load(f),
                                                  ConfigSource.DEPRECATED_USER_PATH_CONFIG,
                                                  'config:user_path_config.txt')
                        result.issues.extend(issues)
                except Exception:
                    pass

        if include_user_presets:
            from .config_schema import ConfigSource
            user_presets_dir = os.path.join(root_dir, 'presets', 'user')
            if os.path.isdir(user_presets_dir):
                for fname in os.listdir(user_presets_dir):
                    if fname.endswith('.json'):
                        fpath = os.path.join(user_presets_dir, fname)
                        try:
                            with open(fpath, "r", encoding="utf-8") as f:
                                raw = json.load(f)
                                cleaned, issues = schema.import_and_validate(
                                    raw, ConfigSource.USER_PRESET, f'preset:user/{fname}')
                                result.issues.extend(issues)
                        except Exception as e:
                            result.issues.append(ConfigIssue(
                                key=f'user_preset:{fname}',
                                issue_type=ConfigIssueType.VALIDATION_ERROR,
                                message=f'Failed to parse user preset {fname}: {e}',
                                source=ConfigSource.USER_PRESET,
                            ))

        schema.apply_defaults(result)
        diagnostics = schema.build_diagnostics(result)
        return diagnostics, result

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
                        field = self.get_field(key)
                        write_flags = []
                        if field and field.save_to_config:
                            if field.is_path:
                                write_flags.append("cfg")
                            elif not key.startswith('cli_'):
                                write_flags.append("preset+cfg")
                        if write_flags:
                            flag_str = f" [{'|'.join(write_flags)}]"
                        else:
                            flag_str = ""
                        lines.append(f"  {key} = {display_val}{flag_str}  # {val.source_detail}")

        preset_data = self.export_for_preset(result, include_schema_defaults=False)
        config_file_data = self.export_for_config_file(result, include_schema_defaults=False)
        lines.append("\n" + "-" * 70)
        lines.append(f"WRITE-BACK PREVIEW")
        lines.append(f"  save_user_preset() will write: {len(preset_data)} keys (non-path, non-cli, non-default)")
        lines.append(f"  update_config_file() will write: {len(config_file_data)} keys (non-cli, non-default, non-builtin-preset)")
        if preset_data:
            sample_keys = sorted(list(preset_data.keys()))[:8]
            lines.append(f"  preset keys: {', '.join(sample_keys)}{' ...' if len(preset_data) > 8 else ''}")

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
                    validator=lambda x: isinstance(x, str)),
    ]

    default_temp_path = os.path.join(tempfile.gettempdir(), 'fooocus')
    path_fields.append(ConfigField('temp_path', default_temp_path, str,
                                   is_path=True, path_is_dir=True, path_auto_create=True,
                                   category='paths', description='Temp directory',
                                   validator=lambda x: isinstance(x, str)))

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
                    validator=lambda x: isinstance(x, str),
                    preset_binding='base_model',
                    ui_to_config_transform=lambda x: x if x else 'model.safetensors'),
        ConfigField('previous_default_models', [], list,
                    category='models', description='Previous default models history',
                    validator=lambda x: isinstance(x, list) and all(isinstance(k, str) for k in x)),
        ConfigField('default_refiner', 'None', str,
                    category='models', description='Refiner model',
                    validator=lambda x: isinstance(x, str),
                    preset_binding='refiner_model',
                    ui_to_config_transform=lambda x: x if x else 'None'),
        ConfigField('default_refiner_switch', 0.8, numbers.Number,
                    category='models', description='Refiner switch point',
                    validator=lambda x: isinstance(x, numbers.Number) and 0 <= x <= 1,
                    preset_binding='refiner_switch',
                    ui_to_config_transform=lambda x: float(x)),
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
                    post_process=post_process_loras,
                    preset_binding='loras'),
        ConfigField('default_max_lora_number', 5, int,
                    category='models', description='Maximum number of LoRA slots',
                    validator=lambda x: isinstance(x, int) and x >= 1),
        ConfigField('default_vae', flags.default_vae, str,
                    category='models', description='VAE override',
                    validator=lambda x: isinstance(x, str),
                    preset_binding='vae',
                    ui_to_config_transform=lambda x: x if x != flags.default_vae else 'Default (model)'),
    ]

    for f in model_fields:
        schema.register_field(f)

    sampling_fields = [
        ConfigField('default_cfg_scale', 7.0, numbers.Number,
                    category='sampling', description='CFG Scale',
                    validator=lambda x: isinstance(x, numbers.Number),
                    preset_binding='guidance_scale',
                    ui_to_config_transform=lambda x: float(x)),
        ConfigField('default_sample_sharpness', 2.0, numbers.Number,
                    category='sampling', description='Sample sharpness',
                    validator=lambda x: isinstance(x, numbers.Number),
                    preset_binding='sharpness',
                    ui_to_config_transform=lambda x: float(x)),
        ConfigField('default_cfg_tsnr', 7.0, numbers.Number,
                    category='sampling', description='Adaptive CFG (TSNR)',
                    validator=lambda x: isinstance(x, numbers.Number),
                    preset_binding='adaptive_cfg',
                    ui_to_config_transform=lambda x: float(x)),
        ConfigField('default_clip_skip', 2, int,
                    category='sampling', description='CLIP skip layers',
                    validator=lambda x: isinstance(x, int) and 1 <= x <= flags.clip_skip_max,
                    preset_binding='clip_skip',
                    ui_to_config_transform=lambda x: int(x)),
        ConfigField('default_sampler', 'dpmpp_2m_sde_gpu', str,
                    category='sampling', description='Sampler name',
                    validator=lambda x: x in flags.sampler_list,
                    preset_binding='sampler'),
        ConfigField('default_scheduler', 'karras', str,
                    category='sampling', description='Scheduler name',
                    validator=lambda x: x in flags.scheduler_list,
                    preset_binding='scheduler'),
        ConfigField('default_performance', flags.Performance.SPEED.value, str,
                    category='sampling', description='Performance mode',
                    validator=lambda x: x in flags.Performance.values(),
                    preset_binding='performance'),
        ConfigField('default_overwrite_step', -1, int,
                    category='sampling', description='Overwrite step count (-1 = auto)',
                    validator=lambda x: isinstance(x, int),
                    preset_binding='steps',
                    ui_to_config_transform=lambda x: int(x)),
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
                    validator=lambda x: isinstance(x, str),
                    preset_binding='prompt'),
        ConfigField('default_prompt_negative', '', str,
                    category='prompts', description='Default negative prompt',
                    validator=lambda x: isinstance(x, str),
                    preset_binding='negative_prompt'),
        ConfigField('default_styles', ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"], list,
                    category='prompts', description='Default styles',
                    validator=lambda x: isinstance(x, list) and all(y in sdxl_styles.legal_style_names for y in x),
                    preset_binding='styles',
                    ui_to_config_transform=lambda x: list(x) if x else []),
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
                    validator=lambda x: x in flags.sdxl_aspect_ratios,
                    preset_binding='resolution',
                    ui_to_config_transform=lambda x: x.replace('×', '*').split(' ')[0] if '×' in str(x) else str(x).replace('×', '*')),
        ConfigField('default_inpaint_engine_version', 'v2.6', str,
                    category='image', description='Inpaint engine version',
                    validator=lambda x: x in flags.inpaint_engine_versions,
                    preset_binding='inpaint_engine_version'),
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
                    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
                    preset_binding='checkpoint_downloads'),
        ConfigField('lora_downloads', {}, dict,
                    category='downloads', description='LoRA URL map',
                    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
                    preset_binding='lora_downloads'),
        ConfigField('embeddings_downloads', {}, dict,
                    category='downloads', description='Embedding URL map',
                    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
                    preset_binding='embeddings_downloads'),
        ConfigField('vae_downloads', {}, dict,
                    category='downloads', description='VAE URL map',
                    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
                    preset_binding='vae_downloads'),
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

    cli_fields = [
        ConfigField('cli_share', False, bool,
                    category='cli', description='Set whether to share on Gradio',
                    cli_arg='share', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_disable_preset_selection', False, bool,
                    category='cli', description='Disables preset selection in Gradio',
                    cli_arg='disable-preset-selection', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_language', 'default', str,
                    category='cli', description='Translate UI using json files in language folder',
                    cli_arg='language',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, str)),
        ConfigField('cli_disable_offload_from_vram', False, bool,
                    category='cli', description='Force loading models to vram when the unload can be avoided',
                    cli_arg='disable-offload-from-vram', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_theme', None, str,
                    category='cli', description='Launches the UI with light or dark theme',
                    cli_arg='theme',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: x is None or isinstance(x, str)),
        ConfigField('cli_disable_image_log', False, bool,
                    category='cli', description='Prevent writing images and logs to the outputs folder',
                    cli_arg='disable-image-log', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_disable_analytics', False, bool,
                    category='cli', description='Disables analytics for Gradio',
                    cli_arg='disable-analytics', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_disable_metadata', False, bool,
                    category='cli', description='Disables saving metadata to images',
                    cli_arg='disable-metadata', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_disable_preset_download', False, bool,
                    category='cli', description='Disables downloading models for presets',
                    cli_arg='disable-preset-download', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_disable_enhance_output_sorting', False, bool,
                    category='cli', description='Disables enhance output sorting for final image gallery',
                    cli_arg='disable-enhance-output-sorting', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_enable_auto_describe_image', False, bool,
                    category='cli', description='Enables automatic description of uov and enhance image when prompt is empty',
                    cli_arg='enable-auto-describe-image', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_always_download_new_model', False, bool,
                    category='cli', description='Always download newer models',
                    cli_arg='always-download-new-model', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_rebuild_hash_cache', None, int,
                    category='cli', description='Generates missing model and LoRA hashes',
                    cli_arg='rebuild-hash-cache',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: x is None or isinstance(x, int)),
        ConfigField('cli_preset', None, str,
                    category='cli', description='Apply specified UI preset',
                    cli_arg='preset',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: x is None or isinstance(x, str)),
        ConfigField('cli_preflight_check', False, bool,
                    category='cli', description='Run configuration preflight checks and exit',
                    cli_arg='preflight-check', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_preflight_json', False, bool,
                    category='cli', description='Output preflight result as JSON (use with --preflight-check)',
                    cli_arg='preflight-json', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
        ConfigField('cli_preflight_strict', False, bool,
                    category='cli', description='Treat warnings as errors in preflight (use with --preflight-check)',
                    cli_arg='preflight-strict', cli_action='store_true',
                    ui_default=False, save_to_config=False,
                    validator=lambda x: isinstance(x, bool)),
    ]

    for f in cli_fields:
        schema.register_field(f)

    return schema


def _cli_main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog='python -m modules.config_schema',
        description='Fooocus configuration schema validator and diagnostics tool.'
    )
    parser.add_argument('--check', action='store_true',
                        help='Run configuration diagnostics on config.txt, presets, and user_path_config.')
    parser.add_argument('--json', dest='output_json', action='store_true',
                        help='Output diagnostics as machine-readable JSON.')
    parser.add_argument('--all', '-a', dest='include_all', action='store_true',
                        help='Check all sources including user presets and deprecated user_path_config.txt.')
    parser.add_argument('--user-presets', action='store_true',
                        help='Also validate user presets in presets/user/.')
    parser.add_argument('--user-path-config', action='store_true',
                        help='Also validate deprecated user_path_config.txt.')
    parser.add_argument('--root', type=str, default=None,
                        help='Fooocus root directory (defaults to current working directory).')
    parser.add_argument('--strict', action='store_true',
                        help='Exit with non-zero code if warnings or errors exist.')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress human-readable output (combine with --json).')

    args = parser.parse_args(argv)

    root_dir = args.root or os.getcwd()

    if not args.check:
        parser.print_help()
        return 0

    diagnostics, result = ConfigSchema.standalone_check(
        root_dir=root_dir,
        output_json=args.output_json,
        include_config_txt=True,
        include_user_presets=args.include_all or args.user_presets,
        include_deprecated_user_path=args.include_all or args.user_path_config,
    )

    if args.output_json:
        if not args.quiet:
            sys.stderr.write(diagnostics.to_json())
            sys.stderr.write("\n")
        print(diagnostics.to_json())
    elif not args.quiet:
        schema = ConfigSchema(root_dir)
        print(schema.format_diagnostics_terminal(result, show_warnings_only=False))

    if diagnostics.has_errors or (args.strict and diagnostics.warn_count > 0):
        return diagnostics.to_dict()['exit_code']
    return 0


if __name__ == '__main__':
    sys.exit(_cli_main())
