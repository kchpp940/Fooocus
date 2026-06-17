import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

import fooocus_version
import modules.config
import modules.sdxl_styles
from modules.flags import MetadataScheme, Performance, Steps
from modules.flags import SAMPLERS, CIVITAI_NO_KARRAS
from modules.hash_cache import sha256_from_cache
from modules.util import quote, unquote, extract_styles_from_prompt, is_json, get_file_from_folder_list

re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")


class MetadataSource:
    EMBEDDED = 'embedded'
    PRIVATE_LOG = 'private_log'
    PRESET = 'preset'
    UNKNOWN = 'unknown'


@dataclass
class MetadataField:
    key: str
    value: Any = None
    source: str = MetadataSource.UNKNOWN
    valid: bool = True
    error: Optional[str] = None
    raw_value: Any = None

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'value': self.value,
            'source': self.source,
            'valid': self.valid,
            'error': self.error,
        }


@dataclass
class MetadataResult:
    fields: dict = field(default_factory=dict)
    scheme: Optional[MetadataScheme] = None
    source: str = MetadataSource.UNKNOWN
    raw_metadata: Any = None

    def get(self, key: str, default: Any = None) -> Any:
        field = self.fields.get(key)
        if field and field.valid:
            return field.value
        return default

    def set_field(self, key: str, value: Any, source: str = None, valid: bool = True, error: str = None, raw_value: Any = None):
        if source is None:
            source = self.source
        self.fields[key] = MetadataField(
            key=key,
            value=value,
            source=source,
            valid=valid,
            error=error,
            raw_value=raw_value if raw_value is not None else value
        )

    def has(self, key: str) -> bool:
        field = self.fields.get(key)
        return field is not None and field.valid

    def to_dict(self) -> dict:
        return {
            'scheme': self.scheme.value if self.scheme else None,
            'source': self.source,
            'fields': {k: v.to_dict() for k, v in self.fields.items()},
        }

    def to_simple_dict(self) -> dict:
        return {k: v.value for k, v in self.fields.items() if v.valid}

    def merge(self, other: 'MetadataResult', only_missing: bool = True) -> 'MetadataResult':
        for key, field in other.fields.items():
            if only_missing:
                if key not in self.fields or not self.fields[key].valid:
                    self.fields[key] = field
            else:
                self.fields[key] = field
        return self

    def to_labeled_list(self) -> list:
        labeled = []
        for key, field in self.fields.items():
            if field.valid:
                label = key.replace('_', ' ').title()
                labeled.append((label, key, field.value))
        return labeled


@dataclass
class MetadataDiff:
    added: dict = field(default_factory=dict)
    removed: dict = field(default_factory=dict)
    changed: dict = field(default_factory=dict)
    unchanged: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'added': {k: v.to_dict() for k, v in self.added.items()},
            'removed': {k: v.to_dict() for k, v in self.removed.items()},
            'changed': {k: {'old': v[0].to_dict(), 'new': v[1].to_dict()} for k, v in self.changed.items()},
            'unchanged': {k: v.to_dict() for k, v in self.unchanged.items()},
        }

    @property
    def has_differences(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def to_simple_dict(self) -> dict:
        return {
            'added': {k: v.value for k, v in self.added.items()},
            'removed': {k: v.value for k, v in self.removed.items()},
            'changed': {k: {'old': v[0].value, 'new': v[1].value} for k, v in self.changed.items()},
            'unchanged': {k: v.value for k, v in self.unchanged.items()},
        }

    def to_html(self, show_unchanged: bool = False) -> str:
        html_parts = []

        if self.changed:
            html_parts.append('<h3>Changed Parameters</h3>')
            html_parts.append('<table class="metadata"><tr><th>Parameter</th><th>Old Value</th><th>New Value</th><th>Source</th></tr>')
            for key, (old_field, new_field) in sorted(self.changed.items()):
                label = key.replace('_', ' ').title()
                html_parts.append(
                    f'<tr><td class="label">{label}</td>'
                    f'<td class="value">{old_field.value}</td>'
                    f'<td class="value" style="color: #4CAF50;"><b>{new_field.value}</b></td>'
                    f'<td class="value">{new_field.source}</td></tr>'
                )
            html_parts.append('</table>')

        if self.added:
            html_parts.append('<h3>New Parameters</h3>')
            html_parts.append('<table class="metadata"><tr><th>Parameter</th><th>Value</th><th>Source</th></tr>')
            for key, field in sorted(self.added.items()):
                label = key.replace('_', ' ').title()
                html_parts.append(
                    f'<tr><td class="label">{label}</td>'
                    f'<td class="value" style="color: #2196F3;"><b>{field.value}</b></td>'
                    f'<td class="value">{field.source}</td></tr>'
                )
            html_parts.append('</table>')

        if self.removed:
            html_parts.append('<h3>Removed Parameters</h3>')
            html_parts.append('<table class="metadata"><tr><th>Parameter</th><th>Old Value</th><th>Source</th></tr>')
            for key, field in sorted(self.removed.items()):
                label = key.replace('_', ' ').title()
                html_parts.append(
                    f'<tr><td class="label">{label}</td>'
                    f'<td class="value" style="color: #f44336;"><b>{field.value}</b></td>'
                    f'<td class="value">{field.source}</td></tr>'
                )
            html_parts.append('</table>')

        if show_unchanged and self.unchanged:
            html_parts.append('<h3>Unchanged Parameters</h3>')
            html_parts.append('<table class="metadata"><tr><th>Parameter</th><th>Value</th><th>Source</th></tr>')
            for key, field in sorted(self.unchanged.items()):
                label = key.replace('_', ' ').title()
                html_parts.append(
                    f'<tr><td class="label">{label}</td>'
                    f'<td class="value">{field.value}</td>'
                    f'<td class="value">{field.source}</td></tr>'
                )
            html_parts.append('</table>')

        if not html_parts:
            html_parts.append('<p style="color: #4CAF50;">No differences found.</p>')

        return '\n'.join(html_parts)

    def summary(self) -> str:
        parts = []
        if self.changed:
            parts.append(f'{len(self.changed)} changed')
        if self.added:
            parts.append(f'{len(self.added)} added')
        if self.removed:
            parts.append(f'{len(self.removed)} removed')
        if self.unchanged:
            parts.append(f'{len(self.unchanged)} unchanged')
        return ', '.join(parts) if parts else 'No differences'


class MetadataParserBase(ABC):
    @abstractmethod
    def get_scheme(self) -> MetadataScheme:
        raise NotImplementedError

    @abstractmethod
    def parse(self, raw_metadata: Any) -> MetadataResult:
        raise NotImplementedError

    @abstractmethod
    def serialize(self, metadata: MetadataResult, extra_data: dict = None) -> str:
        raise NotImplementedError


class A1111MetadataParser(MetadataParserBase):
    fooocus_to_a1111 = {
        'raw_prompt': 'Raw prompt',
        'raw_negative_prompt': 'Raw negative prompt',
        'negative_prompt': 'Negative prompt',
        'styles': 'Styles',
        'performance': 'Performance',
        'steps': 'Steps',
        'sampler': 'Sampler',
        'scheduler': 'Scheduler',
        'vae': 'VAE',
        'guidance_scale': 'CFG scale',
        'seed': 'Seed',
        'resolution': 'Size',
        'sharpness': 'Sharpness',
        'adm_guidance': 'ADM Guidance',
        'refiner_swap_method': 'Refiner Swap Method',
        'adaptive_cfg': 'Adaptive CFG',
        'clip_skip': 'Clip skip',
        'overwrite_switch': 'Overwrite Switch',
        'freeu': 'FreeU',
        'base_model': 'Model',
        'base_model_hash': 'Model hash',
        'refiner_model': 'Refiner',
        'refiner_model_hash': 'Refiner hash',
        'lora_hashes': 'Lora hashes',
        'lora_weights': 'Lora weights',
        'created_by': 'User',
        'version': 'Version'
    }

    a1111_to_fooocus = {v: k for k, v in fooocus_to_a1111.items()}

    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.A1111

    def parse(self, raw_metadata: str) -> MetadataResult:
        result = MetadataResult(scheme=MetadataScheme.A1111)

        if not raw_metadata or not isinstance(raw_metadata, str):
            return result

        try:
            metadata_prompt = ''
            metadata_negative_prompt = ''
            done_with_prompt = False

            *lines, lastline = raw_metadata.strip().split("\n")
            if len(re_param.findall(lastline)) < 3:
                lines.append(lastline)
                lastline = ''

            for line in lines:
                line = line.strip()
                if line.startswith(f"{self.fooocus_to_a1111['negative_prompt']}:"):
                    done_with_prompt = True
                    line = line[len(f"{self.fooocus_to_a1111['negative_prompt']}:"):].strip()
                if done_with_prompt:
                    metadata_negative_prompt += ('' if metadata_negative_prompt == '' else "\n") + line
                else:
                    metadata_prompt += ('' if metadata_prompt == '' else "\n") + line

            found_styles, prompt, negative_prompt = extract_styles_from_prompt(metadata_prompt, metadata_negative_prompt)

            result.set_field('prompt', prompt)
            result.set_field('negative_prompt', negative_prompt)

            for k, v in re_param.findall(lastline):
                try:
                    fooocus_key = self.a1111_to_fooocus.get(k, k)

                    if v != '' and v[0] == '"' and v[-1] == '"':
                        v = unquote(v)

                    m = re_imagesize.match(v)
                    if m is not None:
                        result.set_field('resolution', str((m.group(1), m.group(2))), raw_value=v)
                    else:
                        result.set_field(fooocus_key, v, raw_value=v)
                except Exception as e:
                    result.set_field(k, None, valid=False, error=str(e), raw_value=v)

            if result.has('raw_prompt'):
                result.set_field('prompt', result.get('raw_prompt'))
                raw_prompt = result.get('raw_prompt').replace("\n", ', ')
                if metadata_prompt != raw_prompt and modules.sdxl_styles.fooocus_expansion not in found_styles:
                    found_styles.append(modules.sdxl_styles.fooocus_expansion)

            if result.has('raw_negative_prompt'):
                result.set_field('negative_prompt', result.get('raw_negative_prompt'))

            result.set_field('styles', str(found_styles))

            if result.has('steps') and not result.has('performance'):
                try:
                    perf = Performance.by_steps(result.get('steps')).value
                    result.set_field('performance', perf)
                except (ValueError, KeyError):
                    pass

            if result.has('sampler'):
                sampler_value = result.get('sampler').replace(' Karras', '')
                for k, v in SAMPLERS.items():
                    if v == sampler_value:
                        result.set_field('sampler', k)
                        break

            for key in ['base_model', 'refiner_model', 'vae']:
                if result.has(key):
                    filenames = modules.config.vae_filenames if key == 'vae' else modules.config.model_filenames
                    self._add_extension_to_filename(result, key, filenames)

            lora_data = ''
            if result.has('lora_weights') and result.get('lora_weights') != '':
                lora_data = result.get('lora_weights')
            elif result.has('lora_hashes') and result.get('lora_hashes') != '':
                hashes = result.get('lora_hashes')
                if hashes.split(', ')[0].count(':') == 2:
                    lora_data = hashes

            if lora_data != '':
                for li, lora in enumerate(lora_data.split(', ')):
                    try:
                        lora_split = lora.split(': ')
                        lora_name = lora_split[0]
                        lora_weight = lora_split[2] if len(lora_split) == 3 else lora_split[1]
                        for filename in modules.config.lora_filenames:
                            path = Path(filename)
                            if lora_name == path.stem:
                                result.set_field(f'lora_combined_{li + 1}', f'{filename} : {lora_weight}')
                                break
                    except Exception as e:
                        result.set_field(f'lora_combined_{li + 1}', None, valid=False, error=str(e), raw_value=lora)

        except Exception as e:
            result.raw_metadata = raw_metadata
            result.set_field('_parse_error', str(e), valid=False, error=str(e))

        return result

    def serialize(self, metadata: MetadataResult, extra_data: dict = None) -> str:
        data = metadata.to_simple_dict()
        if extra_data:
            data.update(extra_data)

        width, height = eval(data.get('resolution', '(1024, 1024)'))

        sampler = data.get('sampler', '')
        scheduler = data.get('scheduler', '')

        if sampler in SAMPLERS and SAMPLERS[sampler] != '':
            sampler = SAMPLERS[sampler]
            if sampler not in CIVITAI_NO_KARRAS and scheduler == 'karras':
                sampler += f' Karras'

        generation_params = {
            self.fooocus_to_a1111['steps']: data.get('steps', 30),
            self.fooocus_to_a1111['sampler']: sampler,
            self.fooocus_to_a1111['seed']: data.get('seed', 0),
            self.fooocus_to_a1111['resolution']: f'{width}x{height}',
            self.fooocus_to_a1111['guidance_scale']: data.get('guidance_scale', 7.0),
            self.fooocus_to_a1111['sharpness']: data.get('sharpness', 2.0),
            self.fooocus_to_a1111['adm_guidance']: data.get('adm_guidance', ''),
            self.fooocus_to_a1111['base_model']: Path(data.get('base_model', '')).stem,
            self.fooocus_to_a1111['base_model_hash']: data.get('base_model_hash', ''),
            self.fooocus_to_a1111['performance']: data.get('performance', ''),
            self.fooocus_to_a1111['scheduler']: scheduler,
            self.fooocus_to_a1111['vae']: Path(data.get('vae', '')).stem,
            self.fooocus_to_a1111['raw_prompt']: data.get('raw_prompt', ''),
            self.fooocus_to_a1111['raw_negative_prompt']: data.get('raw_negative_prompt', ''),
        }

        if data.get('refiner_model') and data['refiner_model'] not in ['', 'None']:
            generation_params[self.fooocus_to_a1111['refiner_model']] = Path(data['refiner_model']).stem
            generation_params[self.fooocus_to_a1111['refiner_model_hash']] = data.get('refiner_model_hash', '')

        for key in ['adaptive_cfg', 'clip_skip', 'overwrite_switch', 'refiner_swap_method', 'freeu']:
            if key in data:
                generation_params[self.fooocus_to_a1111[key]] = data[key]

        loras = data.get('loras', [])
        if len(loras) > 0:
            lora_hashes = []
            lora_weights = []
            for lora_name, lora_weight, lora_hash in loras:
                lora_hashes.append(f'{lora_name}: {lora_hash}')
                lora_weights.append(f'{lora_name}: {lora_weight}')
            generation_params[self.fooocus_to_a1111['lora_hashes']] = ', '.join(lora_hashes)
            generation_params[self.fooocus_to_a1111['lora_weights']] = ', '.join(lora_weights)

        generation_params[self.fooocus_to_a1111['version']] = data.get('version', fooocus_version.version)

        if modules.config.metadata_created_by != '':
            generation_params[self.fooocus_to_a1111['created_by']] = modules.config.metadata_created_by

        generation_params_text = ", ".join(
            [k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])

        full_prompt = data.get('full_prompt', [])
        full_negative_prompt = data.get('full_negative_prompt', [])
        positive_prompt_resolved = ', '.join(full_prompt) if isinstance(full_prompt, list) else str(full_prompt)
        negative_prompt_resolved = ', '.join(full_negative_prompt) if isinstance(full_negative_prompt, list) else str(full_negative_prompt)
        negative_prompt_text = f"\nNegative prompt: {negative_prompt_resolved}" if negative_prompt_resolved else ""

        return f"{positive_prompt_resolved}{negative_prompt_text}\n{generation_params_text}".strip()

    @staticmethod
    def _add_extension_to_filename(result: MetadataResult, key: str, filenames: list):
        value = result.get(key)
        for filename in filenames:
            path = Path(filename)
            if value == path.stem:
                result.set_field(key, filename)
                break


class FooocusMetadataParser(MetadataParserBase):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.FOOOCUS

    def parse(self, raw_metadata: dict) -> MetadataResult:
        result = MetadataResult(scheme=MetadataScheme.FOOOCUS)
        result.raw_metadata = raw_metadata

        if not raw_metadata or not isinstance(raw_metadata, dict):
            return result

        for key, value in raw_metadata.items():
            try:
                if value in ['', 'None']:
                    result.set_field(key, None, valid=False, error='empty value', raw_value=value)
                    continue

                processed_value = value
                if key in ['base_model', 'refiner_model']:
                    processed_value = self._replace_value_with_filename(key, value, modules.config.model_filenames)
                elif key.startswith('lora_combined_'):
                    processed_value = self._replace_value_with_filename(key, value, modules.config.lora_filenames)
                elif key == 'vae':
                    processed_value = self._replace_value_with_filename(key, value, modules.config.vae_filenames)

                if processed_value is None:
                    result.set_field(key, None, valid=False, error='could not resolve filename', raw_value=value)
                else:
                    result.set_field(key, processed_value, raw_value=value)
            except Exception as e:
                result.set_field(key, None, valid=False, error=str(e), raw_value=value)

        return result

    def serialize(self, metadata: MetadataResult, extra_data: dict = None) -> str:
        data = metadata.to_simple_dict()
        if extra_data:
            data.update(extra_data)

        res = {}
        for key, value in data.items():
            if key.startswith('lora_combined_'):
                try:
                    name, weight = value.split(' : ')
                    name = Path(name).stem
                    res[key] = f'{name} : {weight}'
                except Exception:
                    res[key] = value
            else:
                res[key] = value

        res['full_prompt'] = data.get('full_prompt', [])
        res['full_negative_prompt'] = data.get('full_negative_prompt', [])
        res['steps'] = data.get('steps', 30)
        res['base_model'] = data.get('base_model_name', Path(data.get('base_model', '')).stem)
        res['base_model_hash'] = data.get('base_model_hash', '')

        refiner_model = data.get('refiner_model', '')
        if refiner_model not in ['', 'None']:
            res['refiner_model'] = data.get('refiner_model_name', Path(refiner_model).stem)
            res['refiner_model_hash'] = data.get('refiner_model_hash', '')

        res['vae'] = data.get('vae_name', Path(data.get('vae', '')).stem)
        res['loras'] = data.get('loras', [])

        if modules.config.metadata_created_by != '':
            res['created_by'] = modules.config.metadata_created_by

        return json.dumps(dict(sorted(res.items())))

    @staticmethod
    def _replace_value_with_filename(key: str, value: str, filenames: list) -> Optional[str]:
        for filename in filenames:
            path = Path(filename)
            if key.startswith('lora_combined_'):
                try:
                    name, weight = value.split(' : ')
                    if name == path.stem:
                        return f'{filename} : {weight}'
                except Exception:
                    continue
            elif value == path.stem:
                return filename
        return None


class MetadataService:
    _instance = None

    def __init__(self):
        self._parsers = {
            MetadataScheme.FOOOCUS: FooocusMetadataParser(),
            MetadataScheme.A1111: A1111MetadataParser(),
        }

    @classmethod
    def get_instance(cls) -> 'MetadataService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_parser(self, scheme: MetadataScheme) -> MetadataParserBase:
        parser = self._parsers.get(scheme)
        if parser is None:
            raise ValueError(f"No parser found for scheme: {scheme}")
        return parser

    def read_from_image(self, image: Image.Image) -> tuple[Any, Optional[MetadataScheme]]:
        items = (image.info or {}).copy()

        parameters = items.pop('parameters', None)
        metadata_scheme = items.pop('fooocus_scheme', None)
        exif = items.pop('exif', None)

        if parameters is not None and is_json(parameters):
            parameters = json.loads(parameters)
        elif exif is not None:
            exif_data = image.getexif()
            parameters = exif_data.get(0x9286, None)
            metadata_scheme = exif_data.get(0x927C, None)

            if is_json(parameters):
                parameters = json.loads(parameters)

        try:
            metadata_scheme = MetadataScheme(metadata_scheme)
        except (ValueError, TypeError):
            metadata_scheme = None

            if isinstance(parameters, dict):
                metadata_scheme = MetadataScheme.FOOOCUS

            if isinstance(parameters, str):
                metadata_scheme = MetadataScheme.A1111

        return parameters, metadata_scheme

    def parse_from_image(self, image: Image.Image) -> MetadataResult:
        parameters, scheme = self.read_from_image(image)
        result = MetadataResult(source=MetadataSource.EMBEDDED, scheme=scheme, raw_metadata=parameters)

        if parameters is None or scheme is None:
            return result

        try:
            parser = self.get_parser(scheme)
            parsed = parser.parse(parameters)
            result.fields = parsed.fields
            for field in result.fields.values():
                if field.source == MetadataSource.UNKNOWN:
                    field.source = MetadataSource.EMBEDDED
        except Exception as e:
            result.set_field('_parse_error', str(e), valid=False, error=str(e))

        return result

    def parse(self, data: Any, scheme: MetadataScheme = MetadataScheme.FOOOCUS, source: str = MetadataSource.UNKNOWN) -> MetadataResult:
        result = MetadataResult(source=source, scheme=scheme, raw_metadata=data)

        try:
            parser = self.get_parser(scheme)
            parsed = parser.parse(data)
            result.fields = parsed.fields
            for field in result.fields.values():
                if field.source == MetadataSource.UNKNOWN:
                    field.source = source
        except Exception as e:
            result.set_field('_parse_error', str(e), valid=False, error=str(e))

        return result

    def parse_from_dict(self, data: dict, scheme: MetadataScheme = MetadataScheme.FOOOCUS, source: str = MetadataSource.UNKNOWN) -> MetadataResult:
        return self.parse(data, scheme, source)

    def parse_from_preset(self, preset_content: dict) -> MetadataResult:
        result = MetadataResult(source=MetadataSource.PRESET, scheme=MetadataScheme.FOOOCUS)
        result.raw_metadata = preset_content

        if not preset_content or not isinstance(preset_content, dict):
            return result

        try:
            preset_prepared = {}
            items = preset_content

            for settings_key, meta_key in modules.config.possible_preset_keys.items():
                try:
                    if settings_key == "default_loras":
                        loras = getattr(modules.config, settings_key)
                        if settings_key in items:
                            loras = items[settings_key]
                        for index, lora in enumerate(loras[:modules.config.default_max_lora_number]):
                            preset_prepared[f'lora_combined_{index + 1}'] = ' : '.join(map(str, lora))
                    elif settings_key == "default_aspect_ratio":
                        if settings_key in items and items[settings_key] is not None:
                            default_aspect_ratio = items[settings_key]
                            width, height = default_aspect_ratio.split('*')
                        else:
                            default_aspect_ratio = getattr(modules.config, settings_key)
                            width, height = default_aspect_ratio.split('×')
                            height = height[:height.index(" ")]
                        preset_prepared[meta_key] = (width, height)
                    else:
                        preset_prepared[meta_key] = items[settings_key] if settings_key in items and items[settings_key] is not None else getattr(modules.config, settings_key)

                    if settings_key == "default_styles" or settings_key == "default_aspect_ratio":
                        preset_prepared[meta_key] = str(preset_prepared[meta_key])
                except Exception as e:
                    result.set_field(meta_key if meta_key != '<processed>' else settings_key, None,
                                     source=MetadataSource.PRESET, valid=False, error=str(e),
                                     raw_value=items.get(settings_key))

            parser = self.get_parser(MetadataScheme.FOOOCUS)
            parsed = parser.parse(preset_prepared)
            for key, field in parsed.fields.items():
                if key not in result.fields or field.valid:
                    field.source = MetadataSource.PRESET
                    result.fields[key] = field

        except Exception as e:
            result.set_field('_parse_error', str(e), valid=False, error=str(e))

        return result

    def diff(self, metadata_a: MetadataResult, metadata_b: MetadataResult) -> MetadataDiff:
        diff = MetadataDiff()

        all_keys = set(metadata_a.fields.keys()) | set(metadata_b.fields.keys())

        for key in all_keys:
            a_field = metadata_a.fields.get(key)
            b_field = metadata_b.fields.get(key)

            if a_field is None and b_field is not None:
                if b_field.valid:
                    diff.added[key] = b_field
            elif a_field is not None and b_field is None:
                if a_field.valid:
                    diff.removed[key] = a_field
            elif a_field is not None and b_field is not None:
                if not a_field.valid and not b_field.valid:
                    continue
                elif not a_field.valid:
                    diff.added[key] = b_field
                elif not b_field.valid:
                    diff.removed[key] = a_field
                elif a_field.value != b_field.value:
                    diff.changed[key] = (a_field, b_field)
                else:
                    diff.unchanged[key] = a_field

        return diff

    def parse_private_log_file(self, log_html_path: str) -> dict:
        import urllib.parse
        import os

        result = {}
        if not log_html_path or not os.path.exists(log_html_path):
            return result

        try:
            with open(log_html_path, 'r', encoding='utf-8') as f:
                content = f.read()

            pattern = r'<div id="([^"]+)" class="image-container">.*?onclick="to_clipboard\(\'([^\']*)\'"'
            matches = re.findall(pattern, content, re.DOTALL)

            for div_id, js_txt in matches:
                try:
                    json_str = urllib.parse.unquote(js_txt)
                    metadata_dict = json.loads(json_str)
                    image_filename = div_id.replace('_', '.')
                    result[image_filename] = metadata_dict
                except Exception:
                    continue
        except Exception:
            pass

        return result

    def find_private_log_for_image(self, image_path: str) -> Optional[str]:
        import os

        if not image_path:
            return None

        image_dir = os.path.dirname(image_path)
        log_path = os.path.join(image_dir, 'log.html')

        if os.path.exists(log_path):
            return log_path

        return None

    def parse_from_private_log(self, image_path: str, log_html_path: str = None) -> MetadataResult:
        import os

        result = MetadataResult(source=MetadataSource.PRIVATE_LOG)

        if log_html_path is None:
            log_html_path = self.find_private_log_for_image(image_path)

        if not log_html_path:
            return result

        image_filename = os.path.basename(image_path)
        log_data = self.parse_private_log_file(log_html_path)

        if image_filename in log_data:
            metadata_dict = log_data[image_filename]
            result.raw_metadata = metadata_dict

            if 'metadata_scheme' in metadata_dict:
                try:
                    scheme = MetadataScheme(metadata_dict['metadata_scheme'])
                    result.scheme = scheme
                except (ValueError, TypeError):
                    result.scheme = MetadataScheme.FOOOCUS
            else:
                result.scheme = MetadataScheme.FOOOCUS

            parser = self.get_parser(result.scheme)
            parsed = parser.parse(metadata_dict)
            result.fields = parsed.fields
            for field in result.fields.values():
                if field.source == MetadataSource.UNKNOWN:
                    field.source = MetadataSource.PRIVATE_LOG

        return result

    def parse_from_image_with_private_log(self, image_path: str, image_obj: Image.Image = None, log_html_path: str = None) -> MetadataResult:
        import os

        if image_obj is None:
            if not os.path.exists(image_path):
                return MetadataResult()
            image_obj = Image.open(image_path)

        embedded = self.parse_from_image(image_obj)
        private_log = self.parse_from_private_log(image_path, log_html_path)

        result = MetadataResult(
            scheme=embedded.scheme or private_log.scheme,
            source=MetadataSource.EMBEDDED if embedded.fields else MetadataSource.PRIVATE_LOG,
            raw_metadata=embedded.raw_metadata or private_log.raw_metadata
        )

        result.merge(embedded, only_missing=False)
        result.merge(private_log, only_missing=True)

        return result

    def serialize_metadata(self, metadata: MetadataResult, scheme: MetadataScheme = None, extra_data: dict = None) -> str:
        if scheme is None:
            scheme = metadata.scheme or MetadataScheme.FOOOCUS

        parser = self.get_parser(scheme)
        return parser.serialize(metadata, extra_data)

    def get_exif(self, metadata: str, metadata_scheme: str) -> Image.Exif:
        exif = Image.Exif()
        exif[0x9286] = metadata
        exif[0x0131] = 'Fooocus v' + fooocus_version.version
        exif[0x927C] = metadata_scheme
        return exif

    def build_load_parameters(self, metadata: MetadataResult, is_generating: bool, inpaint_mode: str) -> list:
        loaded_parameter_dict = metadata.to_simple_dict()
        results = [len(loaded_parameter_dict) > 0]

        self._get_image_number('image_number', 'Image Number', metadata, results)
        self._get_str('prompt', 'Prompt', metadata, results)
        self._get_str('negative_prompt', 'Negative Prompt', metadata, results)
        self._get_list('styles', 'Styles', metadata, results)
        performance = self._get_str('performance', 'Performance', metadata, results)
        self._get_steps('steps', 'Steps', metadata, results)
        self._get_number('overwrite_switch', 'Overwrite Switch', metadata, results)
        self._get_resolution('resolution', 'Resolution', metadata, results)
        self._get_number('guidance_scale', 'Guidance Scale', metadata, results)
        self._get_number('sharpness', 'Sharpness', metadata, results)
        self._get_adm_guidance('adm_guidance', 'ADM Guidance', metadata, results)
        self._get_str('refiner_swap_method', 'Refiner Swap Method', metadata, results)
        self._get_number('adaptive_cfg', 'CFG Mimicking from TSNR', metadata, results)
        self._get_number('clip_skip', 'CLIP Skip', metadata, results, cast_type=int)
        self._get_str('base_model', 'Base Model', metadata, results)
        self._get_str('refiner_model', 'Refiner Model', metadata, results)
        self._get_number('refiner_switch', 'Refiner Switch', metadata, results)
        self._get_str('sampler', 'Sampler', metadata, results)
        self._get_str('scheduler', 'Scheduler', metadata, results)
        self._get_str('vae', 'VAE', metadata, results)
        self._get_seed('seed', 'Seed', metadata, results)
        self._get_inpaint_engine_version('inpaint_engine_version', 'Inpaint Engine Version', metadata, results, inpaint_mode)
        self._get_inpaint_method('inpaint_method', 'Inpaint Mode', metadata, results)

        if is_generating:
            results.append(gr.update())
        else:
            results.append(gr.update(visible=True))

        results.append(gr.update(visible=False))

        self._get_freeu('freeu', 'FreeU', metadata, results)

        performance_filename = None
        if performance is not None and performance in Performance.values():
            perf = Performance(performance)
            performance_filename = perf.lora_filename()

        for i in range(modules.config.default_max_lora_number):
            self._get_lora(f'lora_combined_{i + 1}', f'LoRA {i + 1}', metadata, results, performance_filename)

        return results

    @staticmethod
    def _get_str(key: str, fallback: str | None, metadata: MetadataResult, results: list, default=None) -> str | None:
        try:
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            assert isinstance(h, str)
            results.append(h)
            return h
        except Exception:
            import gradio as gr
            results.append(gr.update())
            return None

    @staticmethod
    def _get_list(key: str, fallback: str | None, metadata: MetadataResult, results: list, default=None):
        try:
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            h = eval(h)
            assert isinstance(h, list)
            results.append(h)
        except Exception:
            import gradio as gr
            results.append(gr.update())

    @staticmethod
    def _get_number(key: str, fallback: str | None, metadata: MetadataResult, results: list, default=None, cast_type=float):
        try:
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            assert h is not None
            h = cast_type(h)
            results.append(h)
        except Exception:
            import gradio as gr
            results.append(gr.update())

    @staticmethod
    def _get_image_number(key: str, fallback: str | None, metadata: MetadataResult, results: list, default=None):
        try:
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            assert h is not None
            h = int(h)
            h = min(h, modules.config.default_max_image_number)
            results.append(h)
        except Exception:
            results.append(1)

    @staticmethod
    def _get_steps(key: str, fallback: str | None, metadata: MetadataResult, results: list, default=None):
        try:
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            assert h is not None
            h = int(h)
            performance_name = metadata.get('performance', '').replace(' ', '_').replace('-', '_').casefold()
            performance_candidates = [k for k in Steps.keys() if k.casefold() == performance_name and Steps[k] == h]
            if len(performance_candidates) == 0:
                results.append(h)
                return
            results.append(-1)
        except Exception:
            results.append(-1)

    @staticmethod
    def _get_resolution(key: str, fallback: str | None, metadata: MetadataResult, results: list, default=None):
        try:
            import gradio as gr
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            width, height = eval(h)
            formatted = modules.config.add_ratio(f'{width}*{height}')
            if formatted in modules.config.available_aspect_ratios_labels:
                results.append(formatted)
                results.append(-1)
                results.append(-1)
            else:
                results.append(gr.update())
                results.append(int(width))
                results.append(int(height))
        except Exception:
            import gradio as gr
            results.append(gr.update())
            results.append(gr.update())
            results.append(gr.update())

    @staticmethod
    def _get_seed(key: str, fallback: str | None, metadata: MetadataResult, results: list, default=None):
        try:
            import gradio as gr
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            assert h is not None
            h = int(h)
            results.append(False)
            results.append(h)
        except Exception:
            import gradio as gr
            results.append(gr.update())
            results.append(gr.update())

    @staticmethod
    def _get_inpaint_engine_version(key: str, fallback: str | None, metadata: MetadataResult, results: list, inpaint_mode: str, default=None) -> str | None:
        try:
            import gradio as gr
            import modules.flags
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            assert isinstance(h, str) and h in modules.flags.inpaint_engine_versions
            if inpaint_mode != modules.flags.inpaint_option_detail:
                results.append(h)
            else:
                results.append(gr.update())
            results.append(h)
            return h
        except Exception:
            import gradio as gr
            results.append(gr.update())
            results.append('empty')
            return None

    @staticmethod
    def _get_inpaint_method(key: str, fallback: str | None, metadata: MetadataResult, results: list, default=None) -> str | None:
        try:
            import modules.flags
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            assert isinstance(h, str) and h in modules.flags.inpaint_options
            results.append(h)
            for i in range(modules.config.default_enhance_tabs):
                results.append(h)
            return h
        except Exception:
            import gradio as gr
            results.append(gr.update())
            for i in range(modules.config.default_enhance_tabs):
                results.append(gr.update())

    @staticmethod
    def _get_adm_guidance(key: str, fallback: str | None, metadata: MetadataResult, results: list, default=None):
        try:
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            p, n, e = eval(h)
            results.append(float(p))
            results.append(float(n))
            results.append(float(e))
        except Exception:
            import gradio as gr
            results.append(gr.update())
            results.append(gr.update())
            results.append(gr.update())

    @staticmethod
    def _get_freeu(key: str, fallback: str | None, metadata: MetadataResult, results: list, default=None):
        try:
            h = metadata.get(key, metadata.get(fallback, default) if fallback else default)
            b1, b2, s1, s2 = eval(h)
            results.append(True)
            results.append(float(b1))
            results.append(float(b2))
            results.append(float(s1))
            results.append(float(s2))
        except Exception:
            import gradio as gr
            results.append(False)
            results.append(gr.update())
            results.append(gr.update())
            results.append(gr.update())
            results.append(gr.update())

    @staticmethod
    def _get_lora(key: str, fallback: str | None, metadata: MetadataResult, results: list, performance_filename: str | None):
        try:
            raw_value = metadata.get(key, metadata.get(fallback) if fallback else None)
            split_data = raw_value.split(' : ')
            enabled = True
            name = split_data[0]
            weight = split_data[1]

            if len(split_data) == 3:
                enabled = split_data[0] == 'True'
                name = split_data[1]
                weight = split_data[2]

            if name == performance_filename:
                raise Exception('performance LoRA')

            weight = float(weight)
            results.append(enabled)
            results.append(name)
            results.append(weight)
        except Exception:
            results.append(True)
            results.append('None')
            results.append(1)

    def build_metadata_from_ui_params(self, ui_params: list) -> MetadataResult:
        import gradio as gr
        result = MetadataResult(source='current_ui', scheme=MetadataScheme.FOOOCUS)

        try:
            idx = 0
            result.set_field('__has_data__', str(len(ui_params) > idx), source='current_ui')
            idx += 1
            result.set_field('image_number', str(ui_params[idx]), source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('prompt', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('negative_prompt', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('styles', str(ui_params[idx]), source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('performance', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1

            steps_val = ui_params[idx]
            if isinstance(steps_val, int) and steps_val == -1:
                if result.has('performance') and result.get('performance') in Steps:
                    steps_val = Steps[result.get('performance')]
            result.set_field('steps', str(steps_val), source='current_ui', raw_value=ui_params[idx])
            idx += 1

            result.set_field('overwrite_switch', str(ui_params[idx]), source='current_ui', raw_value=ui_params[idx])
            idx += 1

            aspect_ratio = ui_params[idx]
            idx += 1
            overwrite_width = ui_params[idx]
            idx += 1
            overwrite_height = ui_params[idx]
            idx += 1

            if isinstance(overwrite_width, int) and overwrite_width > 0 and isinstance(overwrite_height, int) and overwrite_height > 0:
                result.set_field('resolution', str((overwrite_width, overwrite_height)), source='current_ui',
                                raw_value=(overwrite_width, overwrite_height))
            else:
                try:
                    if '×' in aspect_ratio:
                        ratio_str = aspect_ratio.split('×')[0]
                        width, height = ratio_str.split('*')
                        result.set_field('resolution', str((int(width), int(height))), source='current_ui',
                                        raw_value=aspect_ratio)
                except Exception:
                    pass

            result.set_field('guidance_scale', str(ui_params[idx]), source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('sharpness', str(ui_params[idx]), source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('adm_guidance', str((ui_params[idx], ui_params[idx + 1], ui_params[idx + 2])),
                            source='current_ui',
                            raw_value=(ui_params[idx], ui_params[idx + 1], ui_params[idx + 2]))
            idx += 3
            result.set_field('refiner_swap_method', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('adaptive_cfg', str(ui_params[idx]), source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('clip_skip', str(ui_params[idx]), source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('base_model', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('refiner_model', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('refiner_switch', str(ui_params[idx]), source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('sampler', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('scheduler', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('vae', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('seed_random', str(ui_params[idx]), source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('seed', str(ui_params[idx]), source='current_ui', raw_value=ui_params[idx])
            idx += 1
            result.set_field('inpaint_engine_version', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1
            idx += 1
            result.set_field('inpaint_method', ui_params[idx], source='current_ui', raw_value=ui_params[idx])
            idx += 1
            for _ in range(modules.config.default_enhance_tabs):
                idx += 1
            idx += 1
            idx += 1

            freeu_enabled = ui_params[idx]
            idx += 1
            if freeu_enabled:
                b1, b2, s1, s2 = ui_params[idx], ui_params[idx + 1], ui_params[idx + 2], ui_params[idx + 3]
                result.set_field('freeu', str((b1, b2, s1, s2)), source='current_ui',
                                raw_value=(freeu_enabled, b1, b2, s1, s2))
            idx += 4

            for i in range(modules.config.default_max_lora_number):
                enabled = ui_params[idx]
                name = ui_params[idx + 1]
                weight = ui_params[idx + 2]
                idx += 3
                if name != 'None':
                    result.set_field(f'lora_combined_{i + 1}', f'{enabled} : {name} : {weight}',
                                    source='current_ui',
                                    raw_value=(enabled, name, weight))

        except Exception as e:
            result.set_field('_parse_error', str(e), valid=False, error=str(e))

        return result

    def build_load_parameters_from_diff(self, diff: MetadataDiff, is_generating: bool, inpaint_mode: str) -> list:
        merged_metadata = MetadataResult(source='diff_merge')

        for field in diff.added.values():
            merged_metadata.fields[field.key] = field
        for key, (_, new_field) in diff.changed.items():
            merged_metadata.fields[key] = new_field

        return self.build_load_parameters(merged_metadata, is_generating, inpaint_mode)

    def build_load_parameters_from_dict(self, metadata_dict: dict, is_generating: bool, inpaint_mode: str) -> list:
        metadata = self.parse(metadata_dict, MetadataScheme.FOOOCUS)
        return self.build_load_parameters(metadata, is_generating, inpaint_mode)


def get_metadata_service() -> MetadataService:
    return MetadataService.get_instance()
