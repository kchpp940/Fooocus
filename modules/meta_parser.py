import json
import re
import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, List, Optional

import gradio as gr
from PIL import Image

import fooocus_version
import modules.config
import modules.sdxl_styles
from modules.flags import MetadataScheme, Performance, Steps
from modules.flags import SAMPLERS, CIVITAI_NO_KARRAS
from modules.hash_cache import sha256_from_cache
from modules.util import quote, unquote, extract_styles_from_prompt, is_json, get_file_from_folder_list

re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])*"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code, re.DOTALL)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")
re_resolution_tuple = re.compile(r"[\(\[]\s*(\d+)\s*[,\*xX]\s*(\d+)\s*[\)\]]")
re_number = re.compile(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")


class LoadStatus:
    SUCCESS = 'success'
    GENERATING_BLOCKED = 'generating_blocked'
    UNRECOGNIZED_FORMAT = 'unrecognized_format'
    FIELD_DAMAGED = 'field_damaged'
    PARTIAL_SUCCESS = 'partial_success'


class LoadResult:
    def __init__(self):
        self.status = LoadStatus.SUCCESS
        self.missing_fields = []
        self.damaged_fields = []
        self.loaded_fields = []
        self.error_details = []
        self.metadata_source = None

    def add_missing(self, field_name: str):
        self.missing_fields.append(field_name)

    def add_loaded(self, field_name: str):
        self.loaded_fields.append(field_name)

    def add_damaged(self, field_name: str, error_msg: str):
        self.damaged_fields.append((field_name, error_msg))
        self.error_details.append((field_name, error_msg))
        if self.status == LoadStatus.SUCCESS:
            self.status = LoadStatus.PARTIAL_SUCCESS

    def add_error(self, field_name: str, error_msg: str):
        self.damaged_fields.append((field_name, error_msg))
        self.error_details.append((field_name, error_msg))
        self.status = LoadStatus.FIELD_DAMAGED

    def set_status(self, status: str):
        self.status = status

    def set_source(self, source: str):
        self.metadata_source = source

    def has_issues(self) -> bool:
        return self.status not in [LoadStatus.SUCCESS, LoadStatus.GENERATING_BLOCKED]

    def has_damage(self) -> bool:
        return len(self.damaged_fields) > 0

    def to_html(self) -> str:
        if self.status == LoadStatus.GENERATING_BLOCKED:
            return (
                '<div style="padding: 8px 12px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; color: #856404; margin: 8px 0;">'
                '⚠️  <strong>参数加载已跳过</strong>：生成正在进行中，请等待生成完成后再加载参数。'
                '</div>'
            )

        if self.status == LoadStatus.UNRECOGNIZED_FORMAT:
            source_info = f'（来自{self.metadata_source}）' if self.metadata_source else ''
            err_detail = ''
            if len(self.error_details) > 0:
                err_detail = f'<br/>错误原因：{self.error_details[0][1]}'
            return (
                '<div style="padding: 8px 12px; background: #f8d7da; border: 1px solid #dc3545; border-radius: 4px; color: #721c24; margin: 8px 0;">'
                f'❌  <strong>参数加载失败</strong>{source_info}：无法识别参数格式。'
                f'{err_detail}'
                '<br/>请确认内容是 Fooocus JSON 或 A1111 风格的参数文本。'
                '</div>'
            )

        total_loaded = len(self.loaded_fields)
        total_damaged = len(self.damaged_fields)
        source_info = f'（来自{self.metadata_source}）' if self.metadata_source else ''

        if total_loaded > 0 and total_damaged == 0:
            return (
                f'<div style="padding: 8px 12px; background: #d4edda; border: 1px solid #28a745; border-radius: 4px; color: #155724; margin: 8px 0;">'
                f'✅  <strong>参数加载成功</strong>{source_info}：共加载 {total_loaded} 个字段。'
                f'</div>'
            )

        if total_loaded > 0 and total_damaged > 0:
            damaged_html = ''
            if total_damaged > 0:
                items = []
                for field, reason in self.damaged_fields[:8]:
                    items.append(f'<li><code>{field}</code>: {reason}</li>')
                more = f'<li>... 还有 {total_damaged - 8} 个字段损坏</li>' if total_damaged > 8 else ''
                damaged_html = f'<br/><strong>损坏的字段（已跳过）：</strong><ul style="margin: 4px 0; padding-left: 20px;">{"".join(items)}{more}</ul>'

            status_icon = '⚠️' if self.status == LoadStatus.PARTIAL_SUCCESS else '❌'
            status_title = '部分参数加载' if self.status == LoadStatus.PARTIAL_SUCCESS else '存在字段损坏'
            status_color = '#856404' if self.status == LoadStatus.PARTIAL_SUCCESS else '#721c24'
            border_color = '#ffc107' if self.status == LoadStatus.PARTIAL_SUCCESS else '#dc3545'
            bg_color = '#fff3cd' if self.status == LoadStatus.PARTIAL_SUCCESS else '#f8d7da'

            return (
                f'<div style="padding: 8px 12px; background: {bg_color}; border: 1px solid {border_color}; border-radius: 4px; color: {status_color}; margin: 8px 0;">'
                f'{status_icon}  <strong>{status_title}</strong>{source_info}：成功加载 {total_loaded} 个字段，{total_damaged} 个字段损坏已跳过。'
                f'{damaged_html}'
                f'</div>'
            )

        if total_loaded == 0 and total_damaged == 0 and self.status == LoadStatus.SUCCESS:
            return ''

        if total_loaded == 0 and total_damaged > 0:
            damaged_html = ''
            items = []
            for field, reason in self.damaged_fields[:8]:
                items.append(f'<li><code>{field}</code>: {reason}</li>')
            more = f'<li>... 还有 {total_damaged - 8} 个字段</li>' if total_damaged > 8 else ''
            damaged_html = f'<br/><strong>损坏的字段：</strong><ul style="margin: 4px 0; padding-left: 20px;">{"".join(items)}{more}</ul>'
            return (
                f'<div style="padding: 8px 12px; background: #f8d7da; border: 1px solid #dc3545; border-radius: 4px; color: #721c24; margin: 8px 0;">'
                f'❌  <strong>加载失败</strong>{source_info}：{total_damaged} 个字段损坏，未能加载任何有效参数。'
                f'{damaged_html}'
                f'</div>'
            )

        return ''


def safe_parse_list(value: Any) -> Optional[List[Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
    return None


def safe_parse_resolution(value: Any) -> Optional[Tuple[int, int]]:
    if isinstance(value, tuple) and len(value) == 2:
        try:
            w, h = int(value[0]), int(value[1])
            if w > 0 and h > 0:
                return (w, h)
        except (ValueError, TypeError):
            pass
    if isinstance(value, list) and len(value) == 2:
        try:
            w, h = int(value[0]), int(value[1])
            if w > 0 and h > 0:
                return (w, h)
        except (ValueError, TypeError):
            pass
    if isinstance(value, str):
        value = value.strip()
        m = re_imagesize.match(value)
        if m is not None:
            return (int(m.group(1)), int(m.group(2)))
        m = re_resolution_tuple.match(value)
        if m is not None:
            return (int(m.group(1)), int(m.group(2)))
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
                w, h = int(parsed[0]), int(parsed[1])
                if w > 0 and h > 0:
                    return (w, h)
        except (ValueError, SyntaxError):
            pass
    return None


def safe_parse_float_tuple(value: Any, expected_len: int) -> Optional[Tuple[float, ...]]:
    if isinstance(value, (tuple, list)) and len(value) == expected_len:
        try:
            return tuple(float(x) for x in value)
        except (ValueError, TypeError):
            pass
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (tuple, list)) and len(parsed) == expected_len:
                return tuple(float(x) for x in parsed)
        except (ValueError, SyntaxError):
            pass
        nums = re.findall(re_number, value)
        if len(nums) == expected_len:
            try:
                return tuple(float(x) for x in nums)
            except (ValueError, TypeError):
                pass
    return None


def safe_parse_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        try:
            return float(value)
        except ValueError:
            pass
    return None


def safe_parse_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            try:
                f = float(value)
                if f.is_integer():
                    return int(f)
            except ValueError:
                pass
    return None


def load_parameter_button_click(raw_metadata: dict | str, is_generating: bool, inpaint_mode: str,
                                source: str = '手动粘贴',
                                parser_damaged_fields: list[tuple[str, str]] | None = None):
    load_result = LoadResult()
    load_result.set_source(source)

    parser_fatal_error = None
    if parser_damaged_fields:
        for field_name, reason in parser_damaged_fields:
            if field_name == 'metadata':
                parser_fatal_error = reason
            else:
                load_result.add_damaged(field_name, reason)

    def get_result_count():
        n = 1
        n += 1
        n += 2
        n += 1
        n += 1
        n += 1
        n += 1
        n += 3
        n += 1
        n += 1
        n += 3
        n += 1
        n += 1
        n += 1
        n += 1
        n += 1
        n += 1
        n += 1
        n += 1
        n += 1
        n += 1
        n += 1
        n += 2
        n += 1
        n += modules.config.default_enhance_tabs
        n += 1
        n += 1
        n += 5
        n += modules.config.default_max_lora_number * 3
        n += 1
        return n

    TOTAL_CONTROLS = get_result_count()

    try:
        loaded_parameter_dict = raw_metadata
        if isinstance(raw_metadata, str):
            loaded_parameter_dict = json.loads(raw_metadata)
        assert isinstance(loaded_parameter_dict, dict)
    except Exception as e:
        print(f"[Load Parameters] Failed to parse metadata: {e}")
        load_result.set_status(LoadStatus.UNRECOGNIZED_FORMAT)
        load_result.add_error('metadata', f'解析失败: {e}')
        loaded_parameter_dict = {}

    if is_generating:
        print("[Load Parameters] Skipping parameter load during generation")
        load_result.set_status(LoadStatus.GENERATING_BLOCKED)
        controls = [gr.update()] * TOTAL_CONTROLS
        controls[-1] = load_result.to_html()
        return controls

    if len(loaded_parameter_dict) == 0:
        if not load_result.has_damage() and parser_fatal_error is None:
            load_result.set_status(LoadStatus.UNRECOGNIZED_FORMAT)
        if parser_fatal_error is not None:
            load_result.set_status(LoadStatus.UNRECOGNIZED_FORMAT)
            load_result.add_error('metadata', parser_fatal_error)
        controls = [gr.update()] * TOTAL_CONTROLS
        controls[-1] = load_result.to_html()
        return controls

    if parser_fatal_error is not None:
        load_result.add_damaged('metadata', parser_fatal_error)

    results = [len(loaded_parameter_dict) > 0]

    get_image_number('image_number', 'Image Number', loaded_parameter_dict, results, load_result)
    get_str('prompt', 'Prompt', loaded_parameter_dict, results, load_result)
    get_str('negative_prompt', 'Negative Prompt', loaded_parameter_dict, results, load_result)
    get_list('styles', 'Styles', loaded_parameter_dict, results, load_result)

    performance = get_str('performance', 'Performance', loaded_parameter_dict, results, load_result)

    get_steps('steps', 'Steps', loaded_parameter_dict, results, load_result)
    get_number('overwrite_switch', 'Overwrite Switch', loaded_parameter_dict, results, load_result)
    get_resolution('resolution', 'Resolution', loaded_parameter_dict, results, load_result)
    get_number('guidance_scale', 'Guidance Scale', loaded_parameter_dict, results, load_result)
    get_number('sharpness', 'Sharpness', loaded_parameter_dict, results, load_result)
    get_adm_guidance('adm_guidance', 'ADM Guidance', loaded_parameter_dict, results, load_result)
    get_str('refiner_swap_method', 'Refiner Swap Method', loaded_parameter_dict, results, load_result)
    get_number('adaptive_cfg', 'CFG Mimicking from TSNR', loaded_parameter_dict, results, load_result)
    get_number('clip_skip', 'CLIP Skip', loaded_parameter_dict, results, load_result, cast_type=int)
    get_str('base_model', 'Base Model', loaded_parameter_dict, results, load_result)
    get_str('refiner_model', 'Refiner Model', loaded_parameter_dict, results, load_result)
    get_number('refiner_switch', 'Refiner Switch', loaded_parameter_dict, results, load_result)
    get_str('sampler', 'Sampler', loaded_parameter_dict, results, load_result)
    get_str('scheduler', 'Scheduler', loaded_parameter_dict, results, load_result)
    get_str('vae', 'VAE', loaded_parameter_dict, results, load_result)
    get_seed('seed', 'Seed', loaded_parameter_dict, results, load_result)
    get_inpaint_engine_version('inpaint_engine_version', 'Inpaint Engine Version', loaded_parameter_dict, results, inpaint_mode, load_result)
    get_inpaint_method('inpaint_method', 'Inpaint Mode', loaded_parameter_dict, results, load_result)

    results.append(gr.update(visible=True))
    results.append(gr.update(visible=False))

    get_freeu('freeu', 'FreeU', loaded_parameter_dict, results, load_result)

    performance_filename = None
    try:
        if performance is not None and performance in Performance.values():
            perf = Performance(performance)
            performance_filename = perf.lora_filename()
    except Exception as e:
        print(f"[Load Parameters] Failed to resolve performance LoRA: {e}")

    for i in range(modules.config.default_max_lora_number):
        lora_key = f'lora_combined_{i + 1}'
        lora_label = f'LoRA {i + 1}'
        get_lora(lora_key, lora_label, loaded_parameter_dict, results, performance_filename, load_result)

    results.append(load_result.to_html())
    return results


def _check_field_exists(key: str, fallback: str | None, source_dict: dict, default) -> tuple[bool, Any]:
    if key in source_dict:
        return True, source_dict[key]
    if fallback is not None and fallback in source_dict:
        return True, source_dict[fallback]
    if default is not None:
        return True, default
    return False, None


def get_str(key: str, fallback: str | None, source_dict: dict, results: list, load_result: LoadResult, default=None) -> str | None:
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(gr.update())
        return None
    try:
        if not isinstance(raw_value, str):
            raise ValueError(f'期望字符串，实际类型: {type(raw_value).__name__}')
        load_result.add_loaded(key)
        results.append(raw_value)
        return raw_value
    except Exception as e:
        load_result.add_damaged(key, f'字符串类型校验失败: {e}')
        results.append(gr.update())
        return None


def get_list(key: str, fallback: str | None, source_dict: dict, results: list, load_result: LoadResult, default=None):
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(gr.update())
        return
    try:
        parsed = safe_parse_list(raw_value)
        if parsed is None:
            raise ValueError(f'无法解析为列表，原始值: {repr(raw_value)[:50]}')
        if not isinstance(parsed, list):
            raise ValueError(f'解析结果类型错误: {type(parsed).__name__}')
        load_result.add_loaded(key)
        results.append(parsed)
    except Exception as e:
        load_result.add_damaged(key, f'列表格式解析失败: {e}')
        results.append(gr.update())


def get_number(key: str, fallback: str | None, source_dict: dict, results: list, load_result: LoadResult, default=None, cast_type=float):
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(gr.update())
        return
    try:
        if raw_value is None:
            raise ValueError('值为 None')
        parsed = cast_type(raw_value)
        load_result.add_loaded(key)
        results.append(parsed)
    except Exception as e:
        load_result.add_damaged(key, f'数值解析失败: {e}')
        results.append(gr.update())


def get_image_number(key: str, fallback: str | None, source_dict: dict, results: list, load_result: LoadResult, default=None):
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(1)
        return
    try:
        if raw_value is None:
            raise ValueError('值为 None')
        parsed = int(raw_value)
        parsed = min(parsed, modules.config.default_max_image_number)
        load_result.add_loaded(key)
        results.append(parsed)
    except Exception as e:
        load_result.add_damaged(key, f'图片数量解析失败: {e}')
        results.append(1)


def get_steps(key: str, fallback: str | None, source_dict: dict, results: list, load_result: LoadResult, default=None):
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(-1)
        return
    try:
        if raw_value is None:
            raise ValueError('值为 None')
        h = int(raw_value)
        performance_name = source_dict.get('performance', '').replace(' ', '_').replace('-', '_').casefold()
        performance_candidates = [k for k in Steps.keys() if k.casefold() == performance_name and Steps[k] == h]
        load_result.add_loaded(key)
        if len(performance_candidates) == 0:
            results.append(h)
            return
        results.append(-1)
    except Exception as e:
        load_result.add_damaged(key, f'步数解析失败: {e}')
        results.append(-1)


def get_resolution(key: str, fallback: str | None, source_dict: dict, results: list, load_result: LoadResult, default=None):
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        return
    try:
        resolution = safe_parse_resolution(raw_value)
        if resolution is None:
            raise ValueError(f'无法解析分辨率，原始值: {repr(raw_value)[:50]}')
        width, height = resolution
        formatted = modules.config.add_ratio(f'{width}*{height}')
        load_result.add_loaded(key)
        if formatted in modules.config.available_aspect_ratios_labels:
            results.append(formatted)
            results.append(-1)
            results.append(-1)
        else:
            results.append(gr.update())
            results.append(int(width))
            results.append(int(height))
    except Exception as e:
        load_result.add_damaged(key, f'分辨率解析失败: {e}')
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_seed(key: str, fallback: str | None, source_dict: dict, results: list, load_result: LoadResult, default=None):
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(gr.update())
        results.append(gr.update())
        return
    try:
        if raw_value is None:
            raise ValueError('值为 None')
        h = int(raw_value)
        load_result.add_loaded(key)
        results.append(False)
        results.append(h)
    except Exception as e:
        load_result.add_damaged(key, f'种子解析失败: {e}')
        results.append(gr.update())
        results.append(gr.update())


def get_inpaint_engine_version(key: str, fallback: str | None, source_dict: dict, results: list, inpaint_mode: str, load_result: LoadResult, default=None) -> str | None:
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(gr.update())
        results.append('empty')
        return None
    try:
        if not isinstance(raw_value, str):
            raise ValueError(f'期望字符串，实际类型: {type(raw_value).__name__}')
        if raw_value not in modules.flags.inpaint_engine_versions:
            raise ValueError(f'值不在有效范围内: {repr(raw_value)}')
        load_result.add_loaded(key)
        if inpaint_mode != modules.flags.inpaint_option_detail:
            results.append(raw_value)
        else:
            results.append(gr.update())
        results.append(raw_value)
        return raw_value
    except Exception as e:
        load_result.add_damaged(key, f'修复引擎版本解析失败: {e}')
        results.append(gr.update())
        results.append('empty')
        return None


def get_inpaint_method(key: str, fallback: str | None, source_dict: dict, results: list, load_result: LoadResult, default=None) -> str | None:
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(gr.update())
        for i in range(modules.config.default_enhance_tabs):
            results.append(gr.update())
        return None
    try:
        if not isinstance(raw_value, str):
            raise ValueError(f'期望字符串，实际类型: {type(raw_value).__name__}')
        if raw_value not in modules.flags.inpaint_options:
            raise ValueError(f'值不在有效范围内: {repr(raw_value)}')
        load_result.add_loaded(key)
        results.append(raw_value)
        for i in range(modules.config.default_enhance_tabs):
            results.append(raw_value)
        return raw_value
    except Exception as e:
        load_result.add_damaged(key, f'修复方法解析失败: {e}')
        results.append(gr.update())
        for i in range(modules.config.default_enhance_tabs):
            results.append(gr.update())
        return None


def get_adm_guidance(key: str, fallback: str | None, source_dict: dict, results: list, load_result: LoadResult, default=None):
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        return
    try:
        parsed = safe_parse_float_tuple(raw_value, 3)
        if parsed is None:
            raise ValueError(f'无法解析为三元组，原始值: {repr(raw_value)[:50]}')
        p, n, e = parsed
        load_result.add_loaded(key)
        results.append(float(p))
        results.append(float(n))
        results.append(float(e))
    except Exception as e:
        load_result.add_damaged(key, f'ADM Guidance 解析失败: {e}')
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_freeu(key: str, fallback: str | None, source_dict: dict, results: list, load_result: LoadResult, default=None):
    exists, raw_value = _check_field_exists(key, fallback, source_dict, default)
    if not exists:
        load_result.add_missing(key)
        results.append(False)
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        return
    try:
        parsed = safe_parse_float_tuple(raw_value, 4)
        if parsed is None:
            raise ValueError(f'无法解析为四元组，原始值: {repr(raw_value)[:50]}')
        b1, b2, s1, s2 = parsed
        load_result.add_loaded(key)
        results.append(True)
        results.append(float(b1))
        results.append(float(b2))
        results.append(float(s1))
        results.append(float(s2))
    except Exception as e:
        load_result.add_damaged(key, f'FreeU 解析失败: {e}')
        results.append(False)
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_lora(key: str, fallback: str | None, source_dict: dict, results: list, performance_filename: str | None, load_result: LoadResult):
    exists, raw_value = _check_field_exists(key, fallback, source_dict, None)
    if not exists:
        load_result.add_missing(key)
        results.append(True)
        results.append('None')
        results.append(1)
        return
    try:
        raw_str = str(raw_value).strip()
        if not raw_str or raw_str == 'None':
            raise ValueError(f"空值")

        split_data = [s.strip() for s in raw_str.split(' : ')]

        enabled = True
        name = ''
        weight = 1.0

        if len(split_data) == 2:
            name = split_data[0]
            weight_str = split_data[1]
        elif len(split_data) == 3:
            enabled_str = split_data[0]
            if enabled_str in ['True', 'true', '1']:
                enabled = True
            elif enabled_str in ['False', 'false', '0']:
                enabled = False
            else:
                enabled = bool(enabled_str)
            name = split_data[1]
            weight_str = split_data[2]
        else:
            raise ValueError(f"格式错误: 期望 'name : weight' 或 'enabled : name : weight'")

        if name == performance_filename or (performance_filename is not None and name == Path(performance_filename).stem):
            load_result.add_missing(key)
            results.append(True)
            results.append('None')
            results.append(1)
            return

        if not name or name == 'None':
            raise ValueError(f"LoRA 名称无效: {name}")

        parsed_weight = safe_parse_float(weight_str)
        if parsed_weight is None:
            raise ValueError(f"权重解析失败: {weight_str}")
        weight = parsed_weight

        load_result.add_loaded(key)
        results.append(enabled)
        results.append(name)
        results.append(weight)
    except Exception as e:
        load_result.add_damaged(key, str(e))
        results.append(True)
        results.append('None')
        results.append(1)


def parse_meta_from_preset(preset_content):
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


class MetadataParser(ABC):
    def __init__(self):
        self.raw_prompt: str = ''
        self.full_prompt: str = ''
        self.raw_negative_prompt: str = ''
        self.full_negative_prompt: str = ''
        self.steps: int = Steps.SPEED.value
        self.base_model_name: str = ''
        self.base_model_hash: str = ''
        self.refiner_model_name: str = ''
        self.refiner_model_hash: str = ''
        self.loras: list = []
        self.vae_name: str = ''
        self.damaged_fields: list[tuple[str, str]] = []

    def add_damaged(self, field_name: str, reason: str):
        self.damaged_fields.append((field_name, reason))

    def get_and_clear_damaged(self) -> list[tuple[str, str]]:
        result = list(self.damaged_fields)
        self.damaged_fields.clear()
        return result

    @abstractmethod
    def get_scheme(self) -> MetadataScheme:
        raise NotImplementedError

    @abstractmethod
    def to_json(self, metadata: dict | str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def to_string(self, metadata: dict) -> str:
        raise NotImplementedError

    def set_data(self, raw_prompt, full_prompt, raw_negative_prompt, full_negative_prompt, steps, base_model_name,
                 refiner_model_name, loras, vae_name):
        self.raw_prompt = raw_prompt
        self.full_prompt = full_prompt
        self.raw_negative_prompt = raw_negative_prompt
        self.full_negative_prompt = full_negative_prompt
        self.steps = steps
        self.base_model_name = Path(base_model_name).stem

        base_model_path = get_file_from_folder_list(base_model_name, modules.config.paths_checkpoints)
        self.base_model_hash = sha256_from_cache(base_model_path)

        if refiner_model_name not in ['', 'None']:
            self.refiner_model_name = Path(refiner_model_name).stem
            refiner_model_path = get_file_from_folder_list(refiner_model_name, modules.config.paths_checkpoints)
            self.refiner_model_hash = sha256_from_cache(refiner_model_path)

        self.loras = []
        for (lora_name, lora_weight) in loras:
            if lora_name != 'None':
                lora_path = get_file_from_folder_list(lora_name, modules.config.paths_loras)
                lora_hash = sha256_from_cache(lora_path)
                self.loras.append((Path(lora_name).stem, lora_weight, lora_hash))
        self.vae_name = Path(vae_name).stem


class A1111MetadataParser(MetadataParser):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.A1111

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

    def to_json(self, metadata: str) -> dict:
        data = {}
        try:
            metadata_str = str(metadata).strip()
            if not metadata_str:
                return data
        except Exception as e:
            print(f"[A1111 to_json] Invalid metadata input: {e}")
            return data

        metadata_prompt = ''
        metadata_negative_prompt = ''
        done_with_prompt = False

        try:
            all_lines = metadata_str.split("\n")
            if len(all_lines) == 0:
                return data
            if len(all_lines) == 1:
                lines = []
                lastline = all_lines[0]
            else:
                *lines, lastline = all_lines

            if len(re_param.findall(lastline)) < 3:
                lines.append(lastline)
                lastline = ''
        except Exception as e:
            print(f"[A1111 to_json] Failed to split metadata lines: {e}")
            lines = []
            lastline = ''

        for line in lines:
            try:
                line = line.strip() if isinstance(line, str) else ''
                if not line:
                    continue
                neg_label = self.fooocus_to_a1111['negative_prompt']
                if line.startswith(f"{neg_label}:"):
                    done_with_prompt = True
                    line = line[len(neg_label):].lstrip(':').strip()
                if done_with_prompt:
                    metadata_negative_prompt += ('' if metadata_negative_prompt == '' else "\n") + line
                else:
                    metadata_prompt += ('' if metadata_prompt == '' else "\n") + line
            except Exception as e:
                print(f"[A1111 to_json] Skipping prompt line: {e}")

        try:
            found_styles, prompt, negative_prompt = extract_styles_from_prompt(metadata_prompt, metadata_negative_prompt)
        except Exception as e:
            print(f"[A1111 to_json] Failed to extract styles from prompt: {e}")
            found_styles = []
            prompt = metadata_prompt
            negative_prompt = metadata_negative_prompt

        data['prompt'] = prompt
        data['negative_prompt'] = negative_prompt

        for k, v in re_param.findall(lastline):
            try:
                fooocus_key = self.a1111_to_fooocus.get(k)
                if fooocus_key is None:
                    continue

                v_clean = v
                if isinstance(v_clean, str) and v_clean != '' and v_clean[0] == '"' and v_clean[-1] == '"':
                    v_clean = unquote(v_clean)

                if fooocus_key == 'resolution':
                    m = re_imagesize.match(v_clean.strip())
                    if m is not None:
                        w = int(m.group(1))
                        h = int(m.group(2))
                        data['resolution'] = [w, h]
                    else:
                        self.add_damaged(fooocus_key, f'A1111 Size 无法解析: {repr(v_clean)[:50]}')
                    continue

                if fooocus_key in ['freeu']:
                    parsed = safe_parse_float_tuple(v_clean, 4)
                    if parsed is not None:
                        data[fooocus_key] = list(parsed)
                    else:
                        self.add_damaged(fooocus_key, f'A1111 FreeU 无法解析: {repr(v_clean)[:50]}')
                    continue

                if fooocus_key in ['adm_guidance']:
                    parsed = safe_parse_float_tuple(v_clean, 3)
                    if parsed is not None:
                        data[fooocus_key] = list(parsed)
                    else:
                        self.add_damaged(fooocus_key, f'A1111 ADM Guidance 无法解析: {repr(v_clean)[:50]}')
                    continue

                if fooocus_key in ['styles']:
                    parsed = safe_parse_list(v_clean)
                    if parsed is not None:
                        data[fooocus_key] = parsed
                    else:
                        self.add_damaged(fooocus_key, f'A1111 Styles 无法解析: {repr(v_clean)[:50]}')
                    continue

                if fooocus_key == 'seed':
                    parsed = safe_parse_int(v_clean)
                    if parsed is not None:
                        data[fooocus_key] = parsed
                    else:
                        self.add_damaged(fooocus_key, f'A1111 Seed 无法解析: {repr(v_clean)[:50]}')
                    continue

                if fooocus_key in ['steps', 'clip_skip']:
                    parsed = safe_parse_int(v_clean)
                    if parsed is not None:
                        data[fooocus_key] = parsed
                    else:
                        data[fooocus_key] = v_clean
                    continue

                if fooocus_key in ['guidance_scale', 'sharpness', 'adaptive_cfg', 'refiner_switch', 'overwrite_switch']:
                    parsed = safe_parse_float(v_clean)
                    if parsed is not None:
                        data[fooocus_key] = parsed
                    else:
                        data[fooocus_key] = v_clean
                    continue

                data[fooocus_key] = v_clean

            except Exception as e:
                print(f"[A1111 to_json] Error parsing \"{k}: {v}\": {e}")
                fooocus_key_err = self.a1111_to_fooocus.get(k, k)
                self.add_damaged(fooocus_key_err, f'A1111 解析异常: {e}')
                continue

        try:
            if 'raw_prompt' in data and isinstance(data['raw_prompt'], str) and data['raw_prompt'] != '':
                data['prompt'] = data['raw_prompt']
                raw_prompt_compact = data['raw_prompt'].replace("\n", ', ')
                if metadata_prompt != raw_prompt_compact and modules.sdxl_styles.fooocus_expansion not in found_styles:
                    found_styles.append(modules.sdxl_styles.fooocus_expansion)
        except Exception as e:
            print(f"[A1111 to_json] Failed to apply raw_prompt workaround: {e}")

        try:
            if 'raw_negative_prompt' in data and isinstance(data['raw_negative_prompt'], str) and data['raw_negative_prompt'] != '':
                data['negative_prompt'] = data['raw_negative_prompt']
        except Exception as e:
            print(f"[A1111 to_json] Failed to apply raw_negative_prompt workaround: {e}")

        data['styles'] = found_styles

        try:
            if 'steps' in data and data.get('performance') in [None, '']:
                steps_val = safe_parse_int(data['steps'])
                if steps_val is not None:
                    data['performance'] = Performance.by_steps(steps_val).value
        except (ValueError, KeyError, Exception) as e:
            print(f"[A1111 to_json] Failed to infer performance: {e}")

        try:
            if 'sampler' in data and isinstance(data['sampler'], str):
                sampler_str = data['sampler'].replace(' Karras', '').strip()
                scheduler_inferred = None
                if 'scheduler' not in data or data.get('scheduler') in [None, '']:
                    if data['sampler'].endswith(' Karras'):
                        scheduler_inferred = 'karras'
                for sk, sv in SAMPLERS.items():
                    if sv == sampler_str:
                        data['sampler'] = sk
                        break
                if scheduler_inferred is not None and 'scheduler' not in data:
                    data['scheduler'] = scheduler_inferred
        except Exception as e:
            print(f"[A1111 to_json] Failed to normalize sampler: {e}")

        for key in ['base_model', 'refiner_model', 'vae']:
            try:
                if key in data and data[key] not in [None, '', 'None']:
                    filenames = modules.config.vae_filenames if key == 'vae' else modules.config.model_filenames
                    self.add_extension_to_filename(data, filenames, key)
            except Exception as e:
                print(f"[A1111 to_json] Failed to resolve {key}: {e}")
                continue

        lora_data = ''
        try:
            lw = data.get('lora_weights')
            lh = data.get('lora_hashes')
            if isinstance(lw, str) and lw != '':
                lora_data = lw
            elif isinstance(lh, str) and lh != '' and lh.split(', ')[0].count(':') == 2:
                lora_data = lh
        except Exception as e:
            print(f"[A1111 to_json] Failed to extract lora data: {e}")
            lora_data = ''

        if isinstance(lora_data, str) and lora_data != '':
            for li, lora in enumerate(lora_data.split(', ')):
                try:
                    lora_items = [item.strip() for item in lora.split(':') if item.strip() != '']
                    if len(lora_items) < 2:
                        self.add_damaged(f'lora_combined_{li + 1}', f'A1111 LoRA 格式错误，期望 name:weight，原始值: {repr(lora)[:50]}')
                        continue
                    lora_name = lora_items[0]
                    lora_weight = lora_items[-1]
                    matched_filename = None
                    for filename in modules.config.lora_filenames:
                        path = Path(filename)
                        if lora_name == path.stem:
                            matched_filename = filename
                            break
                    if matched_filename is None:
                        self.add_damaged(f'lora_combined_{li + 1}', f'A1111 LoRA 未在本地找到: {repr(lora_name)}')
                        continue
                    parsed_w = safe_parse_float(lora_weight)
                    if parsed_w is None:
                        self.add_damaged(f'lora_combined_{li + 1}', f'A1111 LoRA 权重解析失败: {repr(lora_weight)}')
                        continue
                    data[f'lora_combined_{li + 1}'] = f'{matched_filename} : {parsed_w}'
                except Exception as e:
                    print(f"[A1111 to_json] Skipping LoRA entry #{li}: {e}")
                    self.add_damaged(f'lora_combined_{li + 1}', f'A1111 LoRA 解析异常: {e}')
                    continue

        return data

    def to_string(self, metadata: list) -> str:
        try:
            data = {k: v for _, k, v in metadata}
        except Exception as e:
            print(f"[A1111 to_string] Failed to build metadata dict: {e}")
            data = {}

        width = 1024
        height = 1024
        try:
            resolution = safe_parse_resolution(data.get('resolution'))
            if resolution is not None:
                width, height = resolution
        except Exception as e:
            print(f"[A1111 to_string] Invalid resolution, using default 1024x1024: {e}")

        sampler = data.get('sampler', '')
        scheduler = data.get('scheduler', '')

        try:
            if isinstance(sampler, str) and sampler in SAMPLERS and SAMPLERS[sampler] != '':
                sampler = SAMPLERS[sampler]
                if sampler not in CIVITAI_NO_KARRAS and scheduler == 'karras' and not sampler.endswith(' Karras'):
                    sampler += f' Karras'
        except Exception as e:
            print(f"[A1111 to_string] Failed to process sampler/scheduler: {e}")

        generation_params = {}
        try:
            generation_params[self.fooocus_to_a1111['steps']] = self.steps
            generation_params[self.fooocus_to_a1111['sampler']] = sampler
            generation_params[self.fooocus_to_a1111['seed']] = data.get('seed', 0)
            generation_params[self.fooocus_to_a1111['resolution']] = f'{width}x{height}'
            generation_params[self.fooocus_to_a1111['guidance_scale']] = data.get('guidance_scale', 7.0)
            generation_params[self.fooocus_to_a1111['sharpness']] = data.get('sharpness', 2.0)
            generation_params[self.fooocus_to_a1111['adm_guidance']] = str(data.get('adm_guidance', ''))
            generation_params[self.fooocus_to_a1111['base_model']] = Path(str(data.get('base_model', ''))).stem
            generation_params[self.fooocus_to_a1111['base_model_hash']] = self.base_model_hash

            generation_params[self.fooocus_to_a1111['performance']] = data.get('performance', '')
            generation_params[self.fooocus_to_a1111['scheduler']] = scheduler
            generation_params[self.fooocus_to_a1111['vae']] = Path(str(data.get('vae', ''))).stem
            generation_params[self.fooocus_to_a1111['raw_prompt']] = self.raw_prompt
            generation_params[self.fooocus_to_a1111['raw_negative_prompt']] = self.raw_negative_prompt
        except Exception as e:
            print(f"[A1111 to_string] Error building base generation_params: {e}")

        try:
            if self.refiner_model_name not in ['', 'None']:
                generation_params[self.fooocus_to_a1111['refiner_model']] = self.refiner_model_name
                generation_params[self.fooocus_to_a1111['refiner_model_hash']] = self.refiner_model_hash
        except Exception as e:
            print(f"[A1111 to_string] Failed to add refiner: {e}")

        for key in ['adaptive_cfg', 'clip_skip', 'overwrite_switch', 'refiner_swap_method', 'freeu']:
            try:
                if key in data:
                    generation_params[self.fooocus_to_a1111[key]] = data[key]
            except Exception as e:
                print(f"[A1111 to_string] Skipping optional field {key}: {e}")
                continue

        try:
            if len(self.loras) > 0:
                lora_hashes = []
                lora_weights = []
                for index, lora_entry in enumerate(self.loras):
                    try:
                        if len(lora_entry) >= 3:
                            lora_name, lora_weight, lora_hash = lora_entry[0], lora_entry[1], lora_entry[2]
                            lora_hashes.append(f'{lora_name}: {lora_hash}')
                            lora_weights.append(f'{lora_name}: {lora_weight}')
                    except Exception as e:
                        print(f"[A1111 to_string] Skipping LoRA #{index}: {e}")
                        continue
                lora_hashes_string = ', '.join(lora_hashes)
                lora_weights_string = ', '.join(lora_weights)
                generation_params[self.fooocus_to_a1111['lora_hashes']] = lora_hashes_string
                generation_params[self.fooocus_to_a1111['lora_weights']] = lora_weights_string
        except Exception as e:
            print(f"[A1111 to_string] Failed to process LoRA list: {e}")

        try:
            generation_params[self.fooocus_to_a1111['version']] = data.get('version', fooocus_version.version)
        except Exception as e:
            print(f"[A1111 to_string] Failed to add version: {e}")

        try:
            if modules.config.metadata_created_by != '':
                generation_params[self.fooocus_to_a1111['created_by']] = modules.config.metadata_created_by
        except Exception as e:
            print(f"[A1111 to_string] Failed to add created_by: {e}")

        try:
            gen_items = []
            for k, v in generation_params.items():
                if v is None:
                    continue
                if k == v:
                    gen_items.append(str(k))
                else:
                    try:
                        gen_items.append(f'{k}: {quote(v)}')
                    except Exception:
                        gen_items.append(f'{k}: {quote(str(v))}')
            generation_params_text = ", ".join(gen_items)
        except Exception as e:
            print(f"[A1111 to_string] Failed to serialize generation_params: {e}")
            generation_params_text = ''

        try:
            positive_prompt_resolved = ', '.join(self.full_prompt) if isinstance(self.full_prompt, list) else str(self.full_prompt)
            negative_prompt_resolved = ', '.join(self.full_negative_prompt) if isinstance(self.full_negative_prompt, list) else str(self.full_negative_prompt)
        except Exception as e:
            print(f"[A1111 to_string] Failed to join prompts: {e}")
            positive_prompt_resolved = str(self.full_prompt)
            negative_prompt_resolved = str(self.full_negative_prompt)

        negative_prompt_text = ''
        try:
            if negative_prompt_resolved:
                negative_prompt_text = f"\nNegative prompt: {negative_prompt_resolved}"
        except Exception as e:
            print(f"[A1111 to_string] Failed to build negative prompt text: {e}")

        result = f"{positive_prompt_resolved}{negative_prompt_text}\n{generation_params_text}".strip()
        return result

    @staticmethod
    def add_extension_to_filename(data, filenames, key):
        try:
            current_val = data.get(key)
            if current_val is None:
                return
            current_str = str(current_val)
            for filename in filenames:
                path = Path(filename)
                if current_str == path.stem:
                    data[key] = filename
                    return
        except Exception as e:
            print(f"[add_extension_to_filename] Failed for {key}: {e}")


class FooocusMetadataParser(MetadataParser):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.FOOOCUS

    def to_json(self, metadata: dict) -> dict:
        result = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str) and value in ['', 'None']:
                continue
            try:
                if key in ['base_model', 'refiner_model']:
                    replaced = self.replace_value_with_filename(key, value, modules.config.model_filenames)
                    if replaced is None:
                        self.add_damaged(key, f'Fooocus 模型文件未找到: {repr(str(value)[:50])}')
                    result[key] = replaced if replaced is not None else value
                elif key.startswith('lora_combined_'):
                    replaced = self.replace_value_with_filename(key, value, modules.config.lora_filenames)
                    if replaced is None:
                        self.add_damaged(key, f'Fooocus LoRA 文件未找到: {repr(str(value)[:50])}')
                    result[key] = replaced if replaced is not None else value
                elif key == 'vae':
                    replaced = self.replace_value_with_filename(key, value, modules.config.vae_filenames)
                    if replaced is None:
                        self.add_damaged(key, f'Fooocus VAE 文件未找到: {repr(str(value)[:50])}')
                    result[key] = replaced if replaced is not None else value
                else:
                    result[key] = value
            except Exception as e:
                print(f"[Fooocus to_json] Skipping field {key} due to error: {e}")
                self.add_damaged(key, f'Fooocus 字段解析异常: {e}')
                if key not in result and not (isinstance(value, str) and value in ['', 'None']) and value is not None:
                    result[key] = value

        return result

    def to_string(self, metadata: list) -> str:
        res = {}
        for label, key, value in metadata:
            try:
                if key.startswith('lora_combined_'):
                    value_str = str(value) if not isinstance(value, str) else value
                    if ' : ' in value_str:
                        parts = [p.strip() for p in value_str.split(' : ')]
                        if len(parts) >= 2:
                            lora_name = parts[0] if len(parts) == 2 else parts[1]
                            lora_weight = parts[1] if len(parts) == 2 else parts[2]
                            lora_name = Path(lora_name).stem
                            if len(parts) == 3:
                                res[key] = f'{parts[0]} : {lora_name} : {lora_weight}'
                            else:
                                res[key] = f'{lora_name} : {lora_weight}'
                            continue
                    res[key] = value
                    continue

                if key == 'resolution':
                    parsed = safe_parse_resolution(value)
                    if parsed is not None:
                        res[key] = [parsed[0], parsed[1]]
                        continue

                if key == 'styles':
                    parsed = safe_parse_list(value)
                    if parsed is not None:
                        res[key] = parsed
                        continue

                if key == 'freeu':
                    parsed = safe_parse_float_tuple(value, 4)
                    if parsed is not None:
                        res[key] = [parsed[0], parsed[1], parsed[2], parsed[3]]
                        continue

                if key == 'adm_guidance':
                    parsed = safe_parse_float_tuple(value, 3)
                    if parsed is not None:
                        res[key] = [parsed[0], parsed[1], parsed[2]]
                        continue

                res[key] = value
            except Exception as e:
                print(f"[Fooocus to_string] Skipping field {key} ({label}) due to error: {e}")

        res['full_prompt'] = self.full_prompt
        res['full_negative_prompt'] = self.full_negative_prompt
        res['steps'] = self.steps
        res['base_model'] = self.base_model_name
        res['base_model_hash'] = self.base_model_hash

        if self.refiner_model_name not in ['', 'None']:
            res['refiner_model'] = self.refiner_model_name
            res['refiner_model_hash'] = self.refiner_model_hash

        res['vae'] = self.vae_name
        res['loras'] = self.loras

        if modules.config.metadata_created_by != '':
            res['created_by'] = modules.config.metadata_created_by

        return json.dumps(dict(sorted(res.items())), ensure_ascii=False)

    @staticmethod
    def replace_value_with_filename(key, value, filenames):
        try:
            value_str = str(value) if not isinstance(value, str) else value
            for filename in filenames:
                path = Path(filename)
                if key.startswith('lora_combined_'):
                    if ' : ' in value_str:
                        parts = [p.strip() for p in value_str.split(' : ')]
                        if len(parts) >= 2:
                            name = parts[0] if len(parts) == 2 else parts[1]
                            weight = parts[1] if len(parts) == 2 else parts[2]
                            if name == path.stem:
                                if len(parts) == 3:
                                    return f'{parts[0]} : {filename} : {weight}'
                                return f'{filename} : {weight}'
                elif value_str == path.stem:
                    return filename
        except Exception as e:
            print(f"[replace_value_with_filename] Error for {key}: {e}")
        return None


def get_metadata_parser(metadata_scheme: MetadataScheme) -> MetadataParser:
    match metadata_scheme:
        case MetadataScheme.FOOOCUS:
            return FooocusMetadataParser()
        case MetadataScheme.A1111:
            return A1111MetadataParser()
        case _:
            raise NotImplementedError


def _decode_exif_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            if value.startswith(b'ASCII\x00\x00\x00'):
                value = value[8:]
            elif value.startswith(b'UNICODE\x00'):
                value = value[8:]
            return value.decode('utf-8', errors='replace')
        except Exception:
            try:
                return value.decode('utf-16', errors='replace')
            except Exception:
                return None
    if isinstance(value, str):
        return value
    return str(value)


def read_info_from_image(file) -> tuple[str | dict | None, MetadataScheme | None]:
    items = (file.info or {}).copy()

    parameters = items.pop('parameters', None)
    metadata_scheme = items.pop('fooocus_scheme', None)
    exif = items.pop('exif', None)

    parameters = _decode_exif_value(parameters)
    metadata_scheme = _decode_exif_value(metadata_scheme)

    if parameters is not None and is_json(parameters):
        parameters = json.loads(parameters)
    elif exif is not None:
        try:
            exif = file.getexif()
            parameters = exif.get(0x9286, None)
            metadata_scheme = exif.get(0x927C, None)
            
            parameters = _decode_exif_value(parameters)
            metadata_scheme = _decode_exif_value(metadata_scheme)

            if parameters is not None and is_json(parameters):
                parameters = json.loads(parameters)
        except Exception as e:
            print(f"[Read Metadata] Failed to read EXIF: {e}")

    try:
        if metadata_scheme is not None:
            metadata_scheme = MetadataScheme(metadata_scheme)
    except ValueError:
        metadata_scheme = None

        if isinstance(parameters, dict):
            metadata_scheme = MetadataScheme.FOOOCUS

        if isinstance(parameters, str):
            metadata_scheme = MetadataScheme.A1111

    return parameters, metadata_scheme


def get_exif(metadata: str | None, metadata_scheme: str):
    exif = Image.Exif()
    # tags see https://github.com/python-pillow/Pillow/blob/9.2.x/src/PIL/ExifTags.py
    # 0x9286 = UserComment
    if metadata is not None:
        exif[0x9286] = metadata.encode('utf-8')
    # 0x0131 = Software
    exif[0x0131] = 'Fooocus v' + fooocus_version.version
    # 0x927C = MakerNote
    if metadata_scheme is not None:
        exif[0x927C] = metadata_scheme.encode('utf-8')
    return exif
