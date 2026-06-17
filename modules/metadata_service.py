import json
import os
import re
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

import gradio as gr
from PIL import Image

import fooocus_version
import modules.config
import modules.sdxl_styles
from modules.flags import MetadataScheme, Performance, Steps, OutputFormat
from modules.flags import SAMPLERS, CIVITAI_NO_KARRAS
from modules.hash_cache import sha256_from_cache
from modules.util import quote, unquote, extract_styles_from_prompt, is_json, get_file_from_folder_list

re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")
re_log_div_id = re.compile(r'id="([^"]+)"')
re_log_clipboard = re.compile(r"to_clipboard\('([^']+)'\)")


class MetadataSource:
    EMBEDDED = "embedded"
    PRIVATE_LOG = "private_log"
    PRESET = "preset"
    UNKNOWN = "unknown"


@dataclass
class LogMatchDiagnostics:
    available: bool = False
    log_found: bool = False
    matched_by: str = "none"
    candidate_count: int = 0
    matched_entry_id: str = ""
    matched_image_src: str = ""
    rejected_reasons: list = field(default_factory=list)
    filled_field_count: int = 0
    note: str = ""

    def to_display_dict(self) -> dict:
        return {
            "available": self.available,
            "log_found": self.log_found,
            "matched_by": self.matched_by,
            "candidate_count": self.candidate_count,
            "matched_entry_id": self.matched_entry_id,
            "matched_image_src": self.matched_image_src,
            "rejected_reasons": self.rejected_reasons,
            "filled_field_count": self.filled_field_count,
            "note": self.note,
        }


@dataclass
class FieldResult:
    key: str
    label: str
    value: Any = None
    valid: bool = True
    error: str = ""
    source: str = MetadataSource.UNKNOWN
    raw_value: Any = None


@dataclass
class ParsedMetadata:
    fields: dict = field(default_factory=dict)
    source: str = MetadataSource.UNKNOWN
    scheme: Optional[MetadataScheme] = None
    raw: Any = None
    diagnostics: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        if key in self.fields and self.fields[key].valid:
            return self.fields[key].value
        return default

    def get_field(self, key: str) -> Optional[FieldResult]:
        return self.fields.get(key)

    def to_dict(self) -> dict:
        return {k: f.value for k, f in self.fields.items() if f.valid}

    def to_display_dict(self) -> dict:
        result = {}
        for k, f in self.fields.items():
            if f.valid:
                result[f.label] = f.value
            else:
                result[f.label] = f"<error: {f.error}>"
        return result

    def has_errors(self) -> bool:
        return any(not f.valid for f in self.fields.values())

    def errors(self) -> dict:
        return {k: f.error for k, f in self.fields.items() if not f.valid}

    def get_log_diagnostics(self) -> Optional[LogMatchDiagnostics]:
        return self.diagnostics.get("log_match")

    def set_log_diagnostics(self, diag: LogMatchDiagnostics) -> None:
        self.diagnostics["log_match"] = diag


@dataclass
class DiffItem:
    key: str
    label: str
    left_value: Any = None
    right_value: Any = None
    same: bool = False
    left_source: str = MetadataSource.UNKNOWN
    right_source: str = MetadataSource.UNKNOWN


@dataclass
class MetadataDiff:
    items: list = field(default_factory=list)
    same_count: int = 0
    diff_count: int = 0

    def to_display_list(self) -> list:
        return [
            {
                "label": item.label,
                "left": item.left_value,
                "right": item.right_value,
                "same": item.same,
                "left_source": item.left_source,
                "right_source": item.right_source,
            }
            for item in self.items
        ]


class MetadataService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._field_definitions = self._build_field_definitions()

    def _build_field_definitions(self) -> dict:
        return {
            "image_number": {
                "label": "Image Number",
                "type": "int",
                "fallbacks": ["Image Number"],
                "default": 1,
                "validator": lambda v: isinstance(v, int) and 1 <= v <= modules.config.default_max_image_number,
                "transform": lambda v: min(int(v), modules.config.default_max_image_number)
            },
            "prompt": {
                "label": "Prompt",
                "type": "str",
                "fallbacks": ["Prompt", "Raw prompt", "raw_prompt"],
                "default": "",
                "validator": lambda v: isinstance(v, str)
            },
            "negative_prompt": {
                "label": "Negative Prompt",
                "type": "str",
                "fallbacks": ["Negative Prompt", "Negative prompt", "Raw negative prompt", "raw_negative_prompt"],
                "default": "",
                "validator": lambda v: isinstance(v, str)
            },
            "styles": {
                "label": "Styles",
                "type": "list",
                "fallbacks": ["Styles"],
                "default": [],
                "validator": lambda v: isinstance(v, list),
                "transform": lambda v: self._safe_eval_list(v)
            },
            "performance": {
                "label": "Performance",
                "type": "str",
                "fallbacks": ["Performance"],
                "default": Performance.SPEED.value,
                "validator": lambda v: isinstance(v, str) and v in Performance.values()
            },
            "steps": {
                "label": "Steps",
                "type": "int",
                "fallbacks": ["Steps"],
                "default": Steps.SPEED.value,
                "validator": lambda v: isinstance(v, int) and v > 0,
                "transform": lambda v: int(v)
            },
            "overwrite_switch": {
                "label": "Overwrite Switch",
                "type": "int",
                "fallbacks": ["Overwrite Switch"],
                "default": -1,
                "validator": lambda v: isinstance(v, int),
                "transform": lambda v: int(v)
            },
            "resolution": {
                "label": "Resolution",
                "type": "tuple",
                "fallbacks": ["Resolution", "Size"],
                "default": (1024, 1024),
                "validator": lambda v: isinstance(v, (tuple, list)) and len(v) == 2,
                "transform": lambda v: self._safe_eval_resolution(v)
            },
            "guidance_scale": {
                "label": "Guidance Scale",
                "type": "float",
                "fallbacks": ["CFG scale", "Guidance Scale"],
                "default": 7.0,
                "validator": lambda v: isinstance(v, (int, float)),
                "transform": lambda v: float(v)
            },
            "sharpness": {
                "label": "Sharpness",
                "type": "float",
                "fallbacks": ["Sharpness"],
                "default": 2.0,
                "validator": lambda v: isinstance(v, (int, float)),
                "transform": lambda v: float(v)
            },
            "adm_guidance": {
                "label": "ADM Guidance",
                "type": "tuple",
                "fallbacks": ["ADM Guidance"],
                "default": (1.5, 0.8, 0.3),
                "validator": lambda v: isinstance(v, (tuple, list)) and len(v) == 3,
                "transform": lambda v: self._safe_eval_tuple(v, 3)
            },
            "refiner_swap_method": {
                "label": "Refiner Swap Method",
                "type": "str",
                "fallbacks": ["Refiner Swap Method"],
                "default": "joint",
                "validator": lambda v: isinstance(v, str)
            },
            "adaptive_cfg": {
                "label": "CFG Mimicking from TSNR",
                "type": "float",
                "fallbacks": ["Adaptive CFG", "adaptive_cfg"],
                "default": 7.0,
                "validator": lambda v: isinstance(v, (int, float)),
                "transform": lambda v: float(v)
            },
            "clip_skip": {
                "label": "CLIP Skip",
                "type": "int",
                "fallbacks": ["Clip skip", "CLIP Skip"],
                "default": 2,
                "validator": lambda v: isinstance(v, int) and 1 <= v <= 12,
                "transform": lambda v: int(v)
            },
            "base_model": {
                "label": "Base Model",
                "type": "str",
                "fallbacks": ["Base Model", "Model"],
                "default": "None",
                "validator": lambda v: isinstance(v, str)
            },
            "base_model_hash": {
                "label": "Base Model Hash",
                "type": "str",
                "fallbacks": ["Model hash", "base_model_hash"],
                "default": "",
                "validator": lambda v: isinstance(v, str)
            },
            "refiner_model": {
                "label": "Refiner Model",
                "type": "str",
                "fallbacks": ["Refiner Model", "Refiner"],
                "default": "None",
                "validator": lambda v: isinstance(v, str)
            },
            "refiner_model_hash": {
                "label": "Refiner Model Hash",
                "type": "str",
                "fallbacks": ["Refiner hash", "refiner_model_hash"],
                "default": "",
                "validator": lambda v: isinstance(v, str)
            },
            "refiner_switch": {
                "label": "Refiner Switch",
                "type": "float",
                "fallbacks": ["Refiner Switch"],
                "default": 0.8,
                "validator": lambda v: isinstance(v, (int, float)),
                "transform": lambda v: float(v)
            },
            "sampler": {
                "label": "Sampler",
                "type": "str",
                "fallbacks": ["Sampler"],
                "default": "dpmpp_2m_sde_gpu",
                "validator": lambda v: isinstance(v, str) and v in SAMPLERS
            },
            "scheduler": {
                "label": "Scheduler",
                "type": "str",
                "fallbacks": ["Scheduler"],
                "default": "karras",
                "validator": lambda v: isinstance(v, str)
            },
            "vae": {
                "label": "VAE",
                "type": "str",
                "fallbacks": ["VAE"],
                "default": "Default (model)",
                "validator": lambda v: isinstance(v, str)
            },
            "seed": {
                "label": "Seed",
                "type": "int",
                "fallbacks": ["Seed"],
                "default": 0,
                "validator": lambda v: isinstance(v, int),
                "transform": lambda v: int(v)
            },
            "inpaint_engine_version": {
                "label": "Inpaint Engine Version",
                "type": "str",
                "fallbacks": ["Inpaint Engine Version"],
                "default": "v2.6",
                "validator": lambda v: isinstance(v, str) and v in modules.flags.inpaint_engine_versions
            },
            "inpaint_method": {
                "label": "Inpaint Mode",
                "type": "str",
                "fallbacks": ["Inpaint Mode"],
                "default": modules.flags.inpaint_option_default,
                "validator": lambda v: isinstance(v, str) and v in modules.flags.inpaint_options
            },
            "freeu": {
                "label": "FreeU",
                "type": "tuple",
                "fallbacks": ["FreeU"],
                "default": None,
                "validator": lambda v: isinstance(v, (tuple, list)) and len(v) == 4,
                "transform": lambda v: self._safe_eval_tuple(v, 4)
            },
            "version": {
                "label": "Version",
                "type": "str",
                "fallbacks": ["Version"],
                "default": "",
                "validator": lambda v: isinstance(v, str)
            },
            "created_by": {
                "label": "User",
                "type": "str",
                "fallbacks": ["User", "created_by"],
                "default": "",
                "validator": lambda v: isinstance(v, str)
            },
            "lora_hashes": {
                "label": "Lora hashes",
                "type": "str",
                "fallbacks": ["Lora hashes"],
                "default": "",
                "validator": lambda v: isinstance(v, str)
            },
            "lora_weights": {
                "label": "Lora weights",
                "type": "str",
                "fallbacks": ["Lora weights"],
                "default": "",
                "validator": lambda v: isinstance(v, str)
            },
            "full_prompt": {
                "label": "Full Prompt",
                "type": "list",
                "fallbacks": ["full_prompt"],
                "default": [],
                "validator": lambda v: isinstance(v, list)
            },
            "full_negative_prompt": {
                "label": "Full Negative Prompt",
                "type": "list",
                "fallbacks": ["full_negative_prompt"],
                "default": [],
                "validator": lambda v: isinstance(v, list)
            },
            "loras": {
                "label": "LoRAs",
                "type": "list",
                "fallbacks": ["loras"],
                "default": [],
                "validator": lambda v: isinstance(v, list)
            }
        }

    def _safe_eval_list(self, v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                result = eval(v)
                if isinstance(result, list):
                    return result
            except Exception:
                pass
        raise ValueError(f"Cannot convert to list: {v}")

    def _safe_eval_resolution(self, v):
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        if isinstance(v, str):
            m = re_imagesize.match(v.strip())
            if m:
                return (int(m.group(1)), int(m.group(2)))
            try:
                result = eval(v)
                if isinstance(result, (tuple, list)) and len(result) == 2:
                    return (int(result[0]), int(result[1]))
            except Exception:
                pass
        raise ValueError(f"Cannot convert to resolution: {v}")

    def _safe_eval_tuple(self, v, expected_len):
        if isinstance(v, (tuple, list)) and len(v) == expected_len:
            return tuple(float(x) for x in v)
        if isinstance(v, str):
            try:
                result = eval(v)
                if isinstance(result, (tuple, list)) and len(result) == expected_len:
                    return tuple(float(x) for x in result)
            except Exception:
                pass
        raise ValueError(f"Cannot convert to tuple of length {expected_len}: {v}")

    def parse_from_image(self, image: Image.Image) -> ParsedMetadata:
        raw, scheme = self._read_info_from_image(image)
        if raw is None:
            result = ParsedMetadata(source=MetadataSource.EMBEDDED, scheme=None, raw=None)
            return result

        return self.parse_raw(raw, scheme, source=MetadataSource.EMBEDDED)

    def parse_raw(self, raw: Any, scheme: Optional[MetadataScheme] = None,
                  source: str = MetadataSource.UNKNOWN) -> ParsedMetadata:
        result = ParsedMetadata(source=source, scheme=scheme, raw=raw)

        if raw is None:
            return result

        data = {}
        if isinstance(raw, dict):
            data = raw.copy()
        elif isinstance(raw, str):
            if is_json(raw):
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {}
            else:
                data = {"__a1111_text__": raw}

        if "__a1111_text__" in data or (scheme == MetadataScheme.A1111 and isinstance(raw, str)):
            text = data.get("__a1111_text__", raw if isinstance(raw, str) else "")
            data = self._parse_a1111_text(text)

        data = self._normalize_model_filenames(data)
        data = self._normalize_loras(data)

        for key, field_def in self._field_definitions.items():
            result.fields[key] = self._parse_field(key, field_def, data, source)

        return result

    def _parse_field(self, key: str, field_def: dict, data: dict, source: str) -> FieldResult:
        label = field_def["label"]
        result = FieldResult(key=key, label=label, source=source)

        value = None
        found = False

        if key in data and data[key] is not None:
            value = data[key]
            found = True
        else:
            for fallback in field_def.get("fallbacks", []):
                if fallback in data and data[fallback] is not None:
                    value = data[fallback]
                    found = True
                    break

        if not found:
            result.valid = True
            result.value = field_def.get("default")
            return result

        result.raw_value = value

        try:
            if "transform" in field_def:
                value = field_def["transform"](value)

            if "validator" in field_def:
                if not field_def["validator"](value):
                    result.valid = False
                    result.error = f"Validation failed for value: {value}"
                    result.value = field_def.get("default")
                    return result

            result.value = value
            result.valid = True
        except Exception as e:
            result.valid = False
            result.error = str(e)
            result.value = field_def.get("default")

        return result

    def _parse_a1111_text(self, text: str) -> dict:
        metadata_prompt = ''
        metadata_negative_prompt = ''

        done_with_prompt = False

        *lines, lastline = text.strip().split("\n")
        if len(re_param.findall(lastline)) < 3:
            lines.append(lastline)
            lastline = ''

        for line in lines:
            line = line.strip()
            if line.startswith("Negative prompt:"):
                done_with_prompt = True
                line = line[len("Negative prompt:"):].strip()
            if done_with_prompt:
                metadata_negative_prompt += ('' if metadata_negative_prompt == '' else "\n") + line
            else:
                metadata_prompt += ('' if metadata_prompt == '' else "\n") + line

        found_styles, prompt, negative_prompt = extract_styles_from_prompt(metadata_prompt, metadata_negative_prompt)

        data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt
        }

        a1111_to_fooocus = {
            'Raw prompt': 'raw_prompt',
            'Raw negative prompt': 'raw_negative_prompt',
            'Negative prompt': 'negative_prompt',
            'Styles': 'styles',
            'Performance': 'performance',
            'Steps': 'steps',
            'Sampler': 'sampler',
            'Scheduler': 'scheduler',
            'VAE': 'vae',
            'CFG scale': 'guidance_scale',
            'Seed': 'seed',
            'Size': 'resolution',
            'Sharpness': 'sharpness',
            'ADM Guidance': 'adm_guidance',
            'Refiner Swap Method': 'refiner_swap_method',
            'Adaptive CFG': 'adaptive_cfg',
            'Clip skip': 'clip_skip',
            'Overwrite Switch': 'overwrite_switch',
            'FreeU': 'freeu',
            'Model': 'base_model',
            'Model hash': 'base_model_hash',
            'Refiner': 'refiner_model',
            'Refiner hash': 'refiner_model_hash',
            'Lora hashes': 'lora_hashes',
            'Lora weights': 'lora_weights',
            'User': 'created_by',
            'Version': 'version'
        }

        for k, v in re_param.findall(lastline):
            try:
                if v != '' and v[0] == '"' and v[-1] == '"':
                    v = unquote(v)

                m = re_imagesize.match(v)
                if m is not None:
                    data['resolution'] = (m.group(1), m.group(2))
                else:
                    fooocus_key = a1111_to_fooocus.get(k, k)
                    data[fooocus_key] = v
            except Exception:
                pass

        if 'raw_prompt' in data:
            data['prompt'] = data['raw_prompt']
            raw_prompt = data['raw_prompt'].replace("\n", ', ')
            if metadata_prompt != raw_prompt and modules.sdxl_styles.fooocus_expansion not in found_styles:
                found_styles.append(modules.sdxl_styles.fooocus_expansion)

        if 'raw_negative_prompt' in data:
            data['negative_prompt'] = data['raw_negative_prompt']

        data['styles'] = list(found_styles)

        if 'steps' in data and 'performance' not in data:
            try:
                steps_int = int(data['steps'])
                data['performance'] = Performance.by_steps(steps_int).value
            except (ValueError, KeyError):
                pass

        if 'sampler' in data:
            sampler = data['sampler'].replace(' Karras', '')
            if 'Karras' in data.get('sampler', '') and 'scheduler' not in data:
                data['scheduler'] = 'karras'
            for k, v in SAMPLERS.items():
                if v == sampler:
                    data['sampler'] = k
                    break

        return data

    def _normalize_model_filenames(self, data: dict) -> dict:
        result = data.copy()

        for key in ['base_model', 'refiner_model']:
            if key in result and result[key]:
                result[key] = self._resolve_filename(result[key], modules.config.model_filenames)

        if 'vae' in result and result['vae']:
            result['vae'] = self._resolve_filename(result['vae'], modules.config.vae_filenames)

        return result

    def _resolve_filename(self, stem_or_name: str, filenames: list) -> str:
        if not stem_or_name or stem_or_name in ['', 'None']:
            return stem_or_name

        for filename in filenames:
            path = Path(filename)
            if stem_or_name == path.stem or stem_or_name == filename:
                return filename

        return stem_or_name

    def _normalize_loras(self, data: dict) -> dict:
        result = data.copy()

        lora_data = ''
        if result.get('lora_weights', '') != '':
            lora_data = result['lora_weights']
        elif result.get('lora_hashes', '') != '' and result['lora_hashes'].split(', ')[0].count(':') == 2:
            lora_data = result['lora_hashes']

        if lora_data != '':
            for li, lora in enumerate(lora_data.split(', ')):
                lora_split = lora.split(': ')
                lora_name = lora_split[0]
                lora_weight = lora_split[2] if len(lora_split) == 3 else lora_split[1]
                resolved_name = self._resolve_filename(lora_name, modules.config.lora_filenames)
                result[f'lora_combined_{li + 1}'] = f'{resolved_name} : {lora_weight}'

        return result

    def _read_info_from_image(self, image: Image.Image) -> tuple:
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

    def get_lora_field(self, index: int, data: dict, performance_filename: Optional[str] = None) -> dict:
        key = f'lora_combined_{index}'
        fallback = f'LoRA {index}'

        result = {
            'enabled': True,
            'name': 'None',
            'weight': 1.0,
            'valid': True,
            'error': ''
        }

        raw_value = None
        if key in data:
            raw_value = data[key]
        elif fallback in data:
            raw_value = data[fallback]

        if raw_value is None:
            return result

        try:
            if isinstance(raw_value, str):
                split_data = raw_value.split(' : ')
                enabled = True
                name = split_data[0]
                weight = split_data[1]

                if len(split_data) == 3:
                    enabled = split_data[0] == 'True'
                    name = split_data[1]
                    weight = split_data[2]

                if name == performance_filename:
                    result['name'] = 'None'
                    result['enabled'] = True
                    result['weight'] = 1.0
                    return result

                weight = float(weight)
                resolved_name = self._resolve_filename(name, modules.config.lora_filenames)

                result['enabled'] = enabled
                result['name'] = resolved_name if resolved_name else name
                result['weight'] = weight
            elif isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
                if len(raw_value) == 3:
                    result['enabled'] = raw_value[0]
                    result['name'] = raw_value[1]
                    result['weight'] = float(raw_value[2])
                else:
                    result['name'] = raw_value[0]
                    result['weight'] = float(raw_value[1])
        except Exception as e:
            result['valid'] = False
            result['error'] = str(e)

        return result

    def get_all_loras(self, data: dict, max_count: int, performance_filename: Optional[str] = None) -> list:
        loras = []
        for i in range(1, max_count + 1):
            loras.append(self.get_lora_field(i, data, performance_filename))
        return loras

    def diff(self, left: ParsedMetadata, right: ParsedMetadata, keys: Optional[list] = None) -> MetadataDiff:
        result = MetadataDiff()

        if keys is None:
            all_keys = set()
            all_keys.update(left.fields.keys())
            all_keys.update(right.fields.keys())
            keys = sorted(all_keys)

        for key in keys:
            label = self._field_definitions.get(key, {}).get("label", key)
            left_field = left.get_field(key)
            right_field = right.get_field(key)

            left_val = left_field.value if left_field else None
            right_val = right_field.value if right_field else None

            same = left_val == right_val

            item = DiffItem(
                key=key,
                label=label,
                left_value=left_val,
                right_value=right_val,
                same=same,
                left_source=left_field.source if left_field else MetadataSource.UNKNOWN,
                right_source=right_field.source if right_field else MetadataSource.UNKNOWN,
            )
            result.items.append(item)

            if same:
                result.same_count += 1
            else:
                result.diff_count += 1

        return result

    def parse_from_image_with_log(self, image_path: str) -> ParsedMetadata:
        image_path_obj = Path(image_path).resolve()
        try:
            with Image.open(image_path_obj) as img:
                parsed = self.parse_from_image(img)
        except Exception:
            parsed = ParsedMetadata(source=MetadataSource.UNKNOWN, scheme=None, raw=None)

        log_path = image_path_obj.parent / "log.html"
        diag = LogMatchDiagnostics(log_found=log_path.exists())

        if log_path.exists():
            log_parsed, diag = self.parse_from_log_html(
                str(log_path),
                image_filename=image_path_obj.name,
                image_absolute_dir=str(image_path_obj.parent.resolve())
            )
            if log_parsed is not None:
                before_sources = {k: f.source for k, f in parsed.fields.items()}
                parsed = self.merge_metadata(parsed, log_parsed)
                filled = 0
                for k, f in parsed.fields.items():
                    if f.source == MetadataSource.PRIVATE_LOG and before_sources.get(k) != MetadataSource.PRIVATE_LOG:
                        filled += 1
                diag.filled_field_count = filled
                diag.available = True

        parsed.set_log_diagnostics(diag)
        return parsed

    def parse_from_pil_with_log(self, pil_image, optional_filepath: str | None = None) -> ParsedMetadata:
        parsed = self.parse_from_image(pil_image)
        diag = LogMatchDiagnostics(log_found=False)

        if optional_filepath:
            image_path_obj = Path(optional_filepath).resolve()
            log_path = image_path_obj.parent / "log.html"
            diag.log_found = log_path.exists()
            if log_path.exists():
                log_parsed, diag = self.parse_from_log_html(
                    str(log_path),
                    image_filename=image_path_obj.name,
                    image_absolute_dir=str(image_path_obj.parent.resolve())
                )
                if log_parsed is not None:
                    before_sources = {k: f.source for k, f in parsed.fields.items()}
                    parsed = self.merge_metadata(parsed, log_parsed)
                    filled = 0
                    for k, f in parsed.fields.items():
                        if f.source == MetadataSource.PRIVATE_LOG and before_sources.get(k) != MetadataSource.PRIVATE_LOG:
                            filled += 1
                    diag.filled_field_count = filled
                    diag.available = True

        parsed.set_log_diagnostics(diag)
        return parsed

    def parse_from_log_html(self, log_html_path: str, image_filename: str | None = None,
                            image_absolute_dir: str | None = None) -> tuple[Optional[ParsedMetadata], LogMatchDiagnostics]:
        log_path = Path(log_html_path).resolve()
        diag = LogMatchDiagnostics(log_found=log_path.exists())

        if not log_path.exists():
            diag.note = "log.html not found next to the image file."
            return None, diag

        try:
            html_content = log_path.read_text(encoding='utf-8')
        except Exception as e:
            diag.note = f"Failed to read log.html: {e}"
            return None, diag

        log_dir = log_path.parent.resolve()

        all_entries = self._extract_all_log_entries(html_content)
        diag.candidate_count = len(all_entries)

        if len(all_entries) == 0:
            diag.note = "log.html found but no image entries were parsed."
            return None, diag

        selected_raw = None
        selected_entry_id = ""
        selected_src = ""
        matched_by = "none"

        if image_filename is not None:
            expected_id = Path(image_filename).name.replace('.', '_')
            expected_basename = Path(image_filename).name

            id_only_candidates = []
            src_and_id_candidates = []
            rejected = []

            for entry_id, entry_data in all_entries.items():
                entry_src = entry_data.get("__image_src__", "")
                id_match = (entry_id == expected_id)
                src_basename_match = bool(entry_src) and Path(entry_src).name == expected_basename
                path_match = False

                if src_basename_match and image_absolute_dir:
                    try:
                        entry_path = (log_dir / entry_src).resolve()
                        expected_path = (Path(image_absolute_dir) / expected_basename).resolve()
                        path_match = (str(entry_path) == str(expected_path))
                    except Exception:
                        path_match = False

                if id_match and not src_basename_match and entry_src:
                    rejected.append(f"entry id='{entry_id}' matches div_id, but image src '{entry_src}' basename ≠ '{expected_basename}' — rejected (possible copy).")
                    continue

                if id_match and src_basename_match and image_absolute_dir and not path_match:
                    rejected.append(f"entry id='{entry_id}' matches basename, but resolved path differs: log suggests '{log_dir / entry_src}' vs image's '{image_absolute_dir}/{expected_basename}' — rejected.")
                    continue

                if id_match and src_basename_match:
                    src_and_id_candidates.append((entry_id, entry_data, "id_and_src_basename"))
                    continue

                if id_match and not entry_src:
                    id_only_candidates.append((entry_id, entry_data, "id_only_no_src"))
                    continue

                if (not id_match) and src_basename_match:
                    if image_absolute_dir:
                        try:
                            entry_path = (log_dir / entry_src).resolve()
                            expected_path = (Path(image_absolute_dir) / expected_basename).resolve()
                            if str(entry_path) == str(expected_path):
                                src_and_id_candidates.append((entry_id, entry_data, "src_basename_and_full_path"))
                                continue
                        except Exception:
                            pass
                    rejected.append(f"entry id='{entry_id}' has basename match '{entry_src}' but no id match — ignored (needs div_id alignment).")
                    continue

            diag.rejected_reasons = rejected

            final_candidates = src_and_id_candidates if src_and_id_candidates else id_only_candidates

            if len(final_candidates) == 1:
                selected_entry_id, selected_raw, matched_by = final_candidates[0]
                selected_src = selected_raw.get("__image_src__", "")
            elif len(final_candidates) > 1:
                for entry_id, entry_data, reason in final_candidates:
                    entry_src = entry_data.get("__image_src__", "")
                    if entry_src and Path(entry_src).name == expected_basename:
                        selected_entry_id = entry_id
                        selected_raw = entry_data
                        selected_src = entry_src
                        matched_by = reason + "_deduped_by_basename"
                        break
                if selected_raw is None:
                    selected_entry_id, selected_raw, matched_by = final_candidates[0]
                    selected_src = selected_raw.get("__image_src__", "")
                    matched_by = matched_by + "_first_among_multiple"
                diag.rejected_reasons.insert(0, f"Multiple candidates ({len(final_candidates)}); picked id='{selected_entry_id}' matched_by={matched_by}.")
            else:
                diag.note = f"No log entry matched image '{image_filename}'. " \
                            f"Checked {len(all_entries)} entries. {len(rejected)} candidates were rejected."
        else:
            for entry_id, entry_data in all_entries.items():
                selected_entry_id = entry_id
                selected_raw = entry_data
                selected_src = entry_data.get("__image_src__", "")
                matched_by = "first_entry_no_image_specified"
                break

        diag.matched_by = matched_by
        diag.matched_entry_id = selected_entry_id
        diag.matched_image_src = selected_src

        if selected_raw is None:
            diag.available = False
            return None, diag

        clean_raw = {k: v for k, v in selected_raw.items() if not k.startswith("__")}
        parsed_log = self.parse_raw(clean_raw, source=MetadataSource.PRIVATE_LOG)
        diag.available = True
        diag.note = diag.note or f"Matched via {matched_by}."
        return parsed_log, diag

    def _extract_all_log_entries(self, html_content: str) -> dict:
        result = {}

        split_parts = html_content.split('<!--fooocus-log-split-->')
        if len(split_parts) >= 2:
            middle = split_parts[1]
        else:
            middle = html_content

        container_pattern = re.compile(
            r'<div\s+id="([^"]+)"\s+class="image-container">',
            re.DOTALL
        )

        ids_positions = [(m.group(1), m.start(), m.end()) for m in container_pattern.finditer(middle)]

        for idx, (entry_id, start_tag_start, start_tag_end) in enumerate(ids_positions):
            if idx + 1 < len(ids_positions):
                next_start = ids_positions[idx + 1][1]
                container_html = middle[start_tag_end:next_start]
            else:
                container_html = middle[start_tag_end:]

            end_tag = container_html.rfind('</div>')
            if end_tag != -1:
                container_html = container_html[:end_tag]

            entry = {}

            img_match = re.search(r"<img\s+src='([^']+)'", container_html)
            if img_match:
                entry["__image_src__"] = img_match.group(1)

            clipboard_match = re.search(r"to_clipboard\('([^']+)'\)", container_html)
            if clipboard_match:
                try:
                    raw_json_str = urllib.parse.unquote(clipboard_match.group(1))
                    clipboard_data = json.loads(raw_json_str)
                    if isinstance(clipboard_data, dict):
                        entry.update(clipboard_data)
                except Exception:
                    pass

            label_value_pairs = re.findall(
                r"<tr><td\s+class='label'>(.*?)</td><td\s+class='value'>(.*?)</td></tr>",
                container_html,
                re.DOTALL
            )
            for label_html, value_html in label_value_pairs:
                label = re.sub(r"<[^>]+>", "", label_html).strip()
                value = re.sub(r"<[^>]+>", "", value_html).strip()
                value = value.replace(" </br> ", "\n")
                value = value.replace("<br>", "\n").replace("</br>", "\n")
                for key, defn in self._field_definitions.items():
                    if defn.get("label") == label:
                        if key not in entry:
                            entry[key] = value
                        break

            if len(entry) > 0:
                result[entry_id] = entry

        return result

    @staticmethod
    def _is_empty_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value == "":
            return True
        if isinstance(value, (list, tuple)) and len(value) == 0:
            return True
        return False

    def merge_metadata(self, embedded: ParsedMetadata, private_log: ParsedMetadata) -> ParsedMetadata:
        merged = ParsedMetadata(
            source=f"{embedded.source}+{private_log.source}" if embedded.source != private_log.source else embedded.source,
            scheme=embedded.scheme if embedded.scheme is not None else private_log.scheme,
            raw={"embedded": embedded.raw, "private_log": private_log.raw}
        )

        all_keys = set()
        all_keys.update(embedded.fields.keys())
        all_keys.update(private_log.fields.keys())

        for key in all_keys:
            embedded_field = embedded.get_field(key)
            private_field = private_log.get_field(key)
            label = self._field_definitions.get(key, {}).get("label", key)

            use_embedded = False
            use_private = False

            if embedded_field is not None:
                if embedded_field.valid and not self._is_empty_value(embedded_field.value):
                    use_embedded = True
                elif private_field is not None and private_field.valid:
                    use_private = True
                else:
                    use_embedded = True
            else:
                if private_field is not None and private_field.valid:
                    use_private = True

            if use_embedded and embedded_field is not None:
                merged.fields[key] = FieldResult(
                    key=key,
                    label=label,
                    value=embedded_field.value,
                    valid=embedded_field.valid,
                    error=embedded_field.error,
                    source=embedded_field.source,
                    raw_value=embedded_field.raw_value
                )
                continue

            if use_private and private_field is not None:
                merged.fields[key] = FieldResult(
                    key=key,
                    label=label,
                    value=private_field.value,
                    valid=private_field.valid,
                    error=private_field.error,
                    source=private_field.source,
                    raw_value=private_field.raw_value
                )
                continue

            if embedded_field is not None:
                merged.fields[key] = embedded_field
            elif private_field is not None:
                merged.fields[key] = private_field

        return merged

    def diff_and_get_fill_parameters(self, base: Any, target: Any, is_generating: bool,
                                     inpaint_mode: str,
                                     base_scheme: Optional[MetadataScheme] = None,
                                     target_scheme: Optional[MetadataScheme] = None,
                                     fill_mode: str = "diff_only") -> tuple:
        base_parsed = self._coerce_to_parsed(base, base_scheme)
        target_parsed = self._coerce_to_parsed(target, target_scheme)

        diff = self.diff(base_parsed, target_parsed)

        if fill_mode == "target_all":
            fill_dict = target_parsed.to_dict()
        elif fill_mode == "diff_only":
            fill_dict = {}
            for item in diff.items:
                if not item.same and item.right_value is not None:
                    fill_dict[item.key] = item.right_value
        else:
            fill_dict = base_parsed.to_dict()

        fill_params = self.load_parameters(fill_dict, is_generating, inpaint_mode)

        return diff, fill_params

    def _coerce_to_parsed(self, value: Any, scheme: Optional[MetadataScheme] = None) -> ParsedMetadata:
        if isinstance(value, ParsedMetadata):
            return value
        if isinstance(value, (dict, str)):
            return self.parse_raw(value, scheme=scheme)
        if isinstance(value, Path) or (isinstance(value, str) and Path(value).exists() and Path(value).suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']):
            try:
                return self.parse_from_image_with_log(str(value))
            except Exception:
                pass
        try:
            from PIL import Image as PILImage
            if isinstance(value, PILImage.Image):
                return self.parse_from_image(value)
        except Exception:
            pass
        return ParsedMetadata(source=MetadataSource.UNKNOWN, scheme=None, raw=None)

    def build_output_metadata(self, task_data: dict, scheme: MetadataScheme) -> str:
        if scheme == MetadataScheme.FOOOCUS:
            return self._build_fooocus_metadata(task_data)
        else:
            return self._build_a1111_metadata(task_data)

    def _build_fooocus_metadata(self, task_data: dict) -> str:
        result = {}

        field_order = [
            'prompt', 'negative_prompt', 'styles', 'performance', 'steps',
            'sampler', 'scheduler', 'guidance_scale', 'sharpness',
            'adm_guidance', 'adaptive_cfg', 'clip_skip', 'seed',
            'resolution', 'overwrite_switch', 'refiner_swap_method',
            'base_model', 'base_model_hash', 'refiner_model', 'refiner_model_hash',
            'refiner_switch', 'vae', 'inpaint_engine_version', 'inpaint_method',
            'freeu', 'version', 'created_by', 'full_prompt', 'full_negative_prompt',
            'loras'
        ]

        for key in field_order:
            if key in task_data:
                result[key] = task_data[key]

        return json.dumps(dict(sorted(result.items())))

    def _build_a1111_metadata(self, task_data: dict) -> str:
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

        data = task_data
        width, height = data.get('resolution', (1024, 1024))
        if isinstance(width, str) or isinstance(height, str):
            try:
                width, height = int(width), int(height)
            except (ValueError, TypeError):
                width, height = 1024, 1024

        sampler = data.get('sampler', 'dpmpp_2m_sde_gpu')
        scheduler = data.get('scheduler', 'karras')

        if sampler in SAMPLERS and SAMPLERS[sampler] != '':
            sampler_name = SAMPLERS[sampler]
            if sampler_name not in CIVITAI_NO_KARRAS and scheduler == 'karras':
                sampler_name += f' Karras'
        else:
            sampler_name = sampler

        generation_params = {
            'Steps': data.get('steps', 30),
            'Sampler': sampler_name,
            'Seed': data.get('seed', 0),
            'Size': f'{width}x{height}',
            'CFG scale': data.get('guidance_scale', 7.0),
            'Sharpness': data.get('sharpness', 2.0),
            'ADM Guidance': data.get('adm_guidance', ''),
            'Model': Path(data.get('base_model', '')).stem if data.get('base_model') else '',
            'Model hash': data.get('base_model_hash', ''),
            'Performance': data.get('performance', ''),
            'Scheduler': scheduler,
            'VAE': Path(data.get('vae', '')).stem if data.get('vae') else '',
            'Raw prompt': data.get('raw_prompt', data.get('prompt', '')),
            'Raw negative prompt': data.get('raw_negative_prompt', data.get('negative_prompt', '')),
        }

        refiner_model = data.get('refiner_model', '')
        if refiner_model and refiner_model not in ['', 'None']:
            generation_params['Refiner'] = Path(refiner_model).stem
            generation_params['Refiner hash'] = data.get('refiner_model_hash', '')

        for key in ['adaptive_cfg', 'clip_skip', 'overwrite_switch', 'refiner_swap_method', 'freeu']:
            if key in data and data[key] is not None:
                a1111_key = fooocus_to_a1111.get(key, key)
                generation_params[a1111_key] = data[key]

        loras = data.get('loras', [])
        if len(loras) > 0:
            lora_hashes = []
            lora_weights = []
            for lora in loras:
                if len(lora) >= 3:
                    lora_name = Path(lora[0]).stem if lora[0] else ''
                    lora_hashes.append(f'{lora_name}: {lora[2]}')
                    lora_weights.append(f'{lora_name}: {lora[1]}')
                elif len(lora) >= 2:
                    lora_name = Path(lora[0]).stem if lora[0] else ''
                    lora_weights.append(f'{lora_name}: {lora[1]}')
            if lora_hashes:
                generation_params['Lora hashes'] = ', '.join(lora_hashes)
            if lora_weights:
                generation_params['Lora weights'] = ', '.join(lora_weights)

        if 'version' in data:
            generation_params['Version'] = data['version']

        if modules.config.metadata_created_by != '':
            generation_params['User'] = modules.config.metadata_created_by

        generation_params_text = ", ".join(
            [k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])

        positive_prompt = data.get('full_prompt', [])
        negative_prompt = data.get('full_negative_prompt', [])
        positive_prompt_resolved = ', '.join(positive_prompt) if isinstance(positive_prompt, list) else str(positive_prompt)
        negative_prompt_resolved = ', '.join(negative_prompt) if isinstance(negative_prompt, list) else str(negative_prompt)
        negative_prompt_text = f"\nNegative prompt: {negative_prompt_resolved}" if negative_prompt_resolved else ""
        return f"{positive_prompt_resolved}{negative_prompt_text}\n{generation_params_text}".strip()

    def get_exif(self, metadata: str, metadata_scheme: str):
        exif = Image.Exif()
        exif[0x9286] = metadata
        exif[0x0131] = 'Fooocus v' + fooocus_version.version
        exif[0x927C] = metadata_scheme
        return exif

    def parse_from_preset(self, preset_content: dict) -> dict:
        assert isinstance(preset_content, dict)
        preset_prepared = {}
        items = preset_content

        for settings_key, meta_key in modules.config.possible_preset_keys.items():
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

        return preset_prepared

    def load_parameters(self, raw_metadata, is_generating: bool, inpaint_mode: str) -> list:
        if isinstance(raw_metadata, str):
            try:
                loaded_parameter_dict = json.loads(raw_metadata)
            except Exception:
                loaded_parameter_dict = {}
        elif isinstance(raw_metadata, dict):
            loaded_parameter_dict = raw_metadata
        else:
            loaded_parameter_dict = {}

        parsed = self.parse_raw(loaded_parameter_dict, source=MetadataSource.UNKNOWN)

        results = [len(loaded_parameter_dict) > 0]

        results.append(self._get_image_number_result(parsed))
        results.append(parsed.get('prompt', ''))
        results.append(parsed.get('negative_prompt', ''))
        results.append(parsed.get('styles', []))

        performance_val = parsed.get('performance', Performance.SPEED.value)
        results.append(performance_val)

        results.append(self._get_steps_result(parsed))
        results.append(parsed.get('overwrite_switch', -1))

        resolution_results = self._get_resolution_results(parsed)
        results.extend(resolution_results)

        results.append(parsed.get('guidance_scale', 7.0))
        results.append(parsed.get('sharpness', 2.0))
        results.extend(self._get_adm_guidance_results(parsed))
        results.append(parsed.get('refiner_swap_method', 'joint'))
        results.append(parsed.get('adaptive_cfg', 7.0))
        results.append(int(parsed.get('clip_skip', 2)))
        results.append(parsed.get('base_model', 'None'))
        results.append(parsed.get('refiner_model', 'None'))
        results.append(parsed.get('refiner_switch', 0.8))
        results.append(parsed.get('sampler', 'dpmpp_2m_sde_gpu'))
        results.append(parsed.get('scheduler', 'karras'))
        results.append(parsed.get('vae', 'Default (model)'))

        seed_results = self._get_seed_results(parsed)
        results.extend(seed_results)

        inpaint_engine_results = self._get_inpaint_engine_results(parsed, inpaint_mode)
        results.extend(inpaint_engine_results)

        inpaint_method_results = self._get_inpaint_method_results(parsed)
        results.extend(inpaint_method_results)

        if is_generating:
            results.append(gr.update())
        else:
            results.append(gr.update(visible=True))

        results.append(gr.update(visible=False))

        freeu_results = self._get_freeu_results(parsed)
        results.extend(freeu_results)

        performance_filename = None
        if performance_val is not None and performance_val in Performance.values():
            try:
                perf = Performance(performance_val)
                performance_filename = perf.lora_filename()
            except ValueError:
                pass

        for i in range(modules.config.default_max_lora_number):
            lora_result = self._get_single_lora_result(i + 1, loaded_parameter_dict, performance_filename)
            results.extend(lora_result)

        return results

    def _get_image_number_result(self, parsed: ParsedMetadata) -> int:
        val = parsed.get('image_number', 1)
        try:
            return min(int(val), modules.config.default_max_image_number)
        except (ValueError, TypeError):
            return 1

    def _get_steps_result(self, parsed: ParsedMetadata) -> int:
        steps = parsed.get('steps')
        if steps is None:
            return -1
        try:
            steps_int = int(steps)
            performance_name = parsed.get('performance', '')
            performance_name = str(performance_name).replace(' ', '_').replace('-', '_').casefold()
            performance_candidates = [
                key for key in Steps.keys()
                if key.casefold() == performance_name and Steps[key] == steps_int
            ]
            if len(performance_candidates) == 0:
                return steps_int
            return -1
        except (ValueError, TypeError):
            return -1

    def _get_resolution_results(self, parsed: ParsedMetadata) -> list:
        resolution = parsed.get('resolution')
        if resolution is None:
            return [gr.update(), gr.update(), gr.update()]
        try:
            if isinstance(resolution, str):
                width, height = self._safe_eval_resolution(resolution)
            else:
                width, height = int(resolution[0]), int(resolution[1])

            formatted = modules.config.add_ratio(f'{width}*{height}')
            if formatted in modules.config.available_aspect_ratios_labels:
                return [formatted, -1, -1]
            else:
                return [gr.update(), width, height]
        except Exception:
            return [gr.update(), gr.update(), gr.update()]

    def _get_adm_guidance_results(self, parsed: ParsedMetadata) -> list:
        adm = parsed.get('adm_guidance')
        if adm is None:
            return [gr.update(), gr.update(), gr.update()]
        try:
            if isinstance(adm, str):
                p, n, e = eval(adm)
            else:
                p, n, e = adm
            return [float(p), float(n), float(e)]
        except Exception:
            return [gr.update(), gr.update(), gr.update()]

    def _get_seed_results(self, parsed: ParsedMetadata) -> list:
        seed = parsed.get('seed')
        if seed is None:
            return [gr.update(), gr.update()]
        try:
            return [False, int(seed)]
        except (ValueError, TypeError):
            return [gr.update(), gr.update()]

    def _get_inpaint_engine_results(self, parsed: ParsedMetadata, inpaint_mode: str) -> list:
        val = parsed.get('inpaint_engine_version')
        if val is None or val not in modules.flags.inpaint_engine_versions:
            return [gr.update(), 'empty']
        if inpaint_mode != modules.flags.inpaint_option_detail:
            return [val, val]
        else:
            return [gr.update(), val]

    def _get_inpaint_method_results(self, parsed: ParsedMetadata) -> list:
        val = parsed.get('inpaint_method')
        if val is None or val not in modules.flags.inpaint_options:
            return [gr.update()] + [gr.update()] * modules.config.default_enhance_tabs
        return [val] + [val] * modules.config.default_enhance_tabs

    def _get_freeu_results(self, parsed: ParsedMetadata) -> list:
        freeu = parsed.get('freeu')
        if freeu is None:
            return [False, gr.update(), gr.update(), gr.update(), gr.update()]
        try:
            if isinstance(freeu, str):
                b1, b2, s1, s2 = eval(freeu)
            else:
                b1, b2, s1, s2 = freeu
            return [True, float(b1), float(b2), float(s1), float(s2)]
        except Exception:
            return [False, gr.update(), gr.update(), gr.update(), gr.update()]

    def _get_single_lora_result(self, index: int, data: dict, performance_filename: Optional[str]) -> list:
        lora = self.get_lora_field(index, data, performance_filename)
        return [lora['enabled'], lora['name'], lora['weight']]


def get_metadata_service() -> MetadataService:
    return MetadataService()
