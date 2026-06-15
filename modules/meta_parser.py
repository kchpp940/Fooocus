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


def load_parameter_button_click(raw_metadata: dict | str, is_generating: bool, inpaint_mode: str):
    try:
        loaded_parameter_dict = raw_metadata
        if isinstance(raw_metadata, str):
            loaded_parameter_dict = json.loads(raw_metadata)
        assert isinstance(loaded_parameter_dict, dict)
    except Exception as e:
        print(f"[Load Parameters] Failed to parse metadata: {e}")
        loaded_parameter_dict = {}

    if is_generating:
        print("[Load Parameters] Skipping parameter load during generation")
        result_count = 1
        result_count += 1
        result_count += 2
        result_count += 1
        result_count += 1
        result_count += 1
        result_count += 1
        result_count += 3
        result_count += 1
        result_count += 1
        result_count += 3
        result_count += 1
        result_count += 1
        result_count += 1
        result_count += 1
        result_count += 1
        result_count += 1
        result_count += 1
        result_count += 1
        result_count += 1
        result_count += 1
        result_count += 2
        result_count += 1
        result_count += modules.config.default_enhance_tabs
        result_count += 1
        result_count += 1
        result_count += 5
        result_count += modules.config.default_max_lora_number * 3
        return [gr.update()] * result_count

    results = [len(loaded_parameter_dict) > 0]

    try:
        get_image_number('image_number', 'Image Number', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load image_number: {e}")
        results.append(1)

    try:
        get_str('prompt', 'Prompt', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load prompt: {e}")
        results.append(gr.update())

    try:
        get_str('negative_prompt', 'Negative Prompt', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load negative_prompt: {e}")
        results.append(gr.update())

    try:
        get_list('styles', 'Styles', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load styles: {e}")
        results.append(gr.update())

    performance = None
    try:
        performance = get_str('performance', 'Performance', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load performance: {e}")
        results.append(gr.update())

    try:
        get_steps('steps', 'Steps', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load steps: {e}")
        results.append(-1)

    try:
        get_number('overwrite_switch', 'Overwrite Switch', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load overwrite_switch: {e}")
        results.append(gr.update())

    try:
        get_resolution('resolution', 'Resolution', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load resolution: {e}")
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

    try:
        get_number('guidance_scale', 'Guidance Scale', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load guidance_scale: {e}")
        results.append(gr.update())

    try:
        get_number('sharpness', 'Sharpness', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load sharpness: {e}")
        results.append(gr.update())

    try:
        get_adm_guidance('adm_guidance', 'ADM Guidance', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load adm_guidance: {e}")
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

    try:
        get_str('refiner_swap_method', 'Refiner Swap Method', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load refiner_swap_method: {e}")
        results.append(gr.update())

    try:
        get_number('adaptive_cfg', 'CFG Mimicking from TSNR', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load adaptive_cfg: {e}")
        results.append(gr.update())

    try:
        get_number('clip_skip', 'CLIP Skip', loaded_parameter_dict, results, cast_type=int)
    except Exception as e:
        print(f"[Load Parameters] Failed to load clip_skip: {e}")
        results.append(gr.update())

    try:
        get_str('base_model', 'Base Model', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load base_model: {e}")
        results.append(gr.update())

    try:
        get_str('refiner_model', 'Refiner Model', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load refiner_model: {e}")
        results.append(gr.update())

    try:
        get_number('refiner_switch', 'Refiner Switch', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load refiner_switch: {e}")
        results.append(gr.update())

    try:
        get_str('sampler', 'Sampler', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load sampler: {e}")
        results.append(gr.update())

    try:
        get_str('scheduler', 'Scheduler', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load scheduler: {e}")
        results.append(gr.update())

    try:
        get_str('vae', 'VAE', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load vae: {e}")
        results.append(gr.update())

    try:
        get_seed('seed', 'Seed', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load seed: {e}")
        results.append(gr.update())
        results.append(gr.update())

    try:
        get_inpaint_engine_version('inpaint_engine_version', 'Inpaint Engine Version', loaded_parameter_dict, results, inpaint_mode)
    except Exception as e:
        print(f"[Load Parameters] Failed to load inpaint_engine_version: {e}")
        results.append(gr.update())
        results.append('empty')

    try:
        get_inpaint_method('inpaint_method', 'Inpaint Mode', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load inpaint_method: {e}")
        results.append(gr.update())
        for i in range(modules.config.default_enhance_tabs):
            results.append(gr.update())

    results.append(gr.update(visible=True))
    results.append(gr.update(visible=False))

    try:
        get_freeu('freeu', 'FreeU', loaded_parameter_dict, results)
    except Exception as e:
        print(f"[Load Parameters] Failed to load freeu: {e}")
        results.append(False)
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

    performance_filename = None
    try:
        if performance is not None and performance in Performance.values():
            perf = Performance(performance)
            performance_filename = perf.lora_filename()
    except Exception as e:
        print(f"[Load Parameters] Failed to resolve performance LoRA: {e}")

    for i in range(modules.config.default_max_lora_number):
        try:
            get_lora(f'lora_combined_{i + 1}', f'LoRA {i + 1}', loaded_parameter_dict, results, performance_filename)
        except Exception as e:
            print(f"[Load Parameters] Failed to load lora_combined_{i + 1}: {e}")
            results.append(True)
            results.append('None')
            results.append(1)

    return results


def get_str(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str)
        results.append(h)
        return h
    except:
        results.append(gr.update())
        return None


def get_list(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        h = safe_parse_list(h)
        assert h is not None and isinstance(h, list)
        results.append(h)
    except:
        results.append(gr.update())


def get_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None, cast_type=float):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = cast_type(h)
        results.append(h)
    except:
        results.append(gr.update())


def get_image_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        h = min(h, modules.config.default_max_image_number)
        results.append(h)
    except:
        results.append(1)


def get_steps(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        # if not in steps or in steps and performance is not the same
        performance_name = source_dict.get('performance', '').replace(' ', '_').replace('-', '_').casefold()
        performance_candidates = [key for key in Steps.keys() if key.casefold() == performance_name and Steps[key] == h]
        if len(performance_candidates) == 0:
            results.append(h)
            return
        results.append(-1)
    except:
        results.append(-1)


def get_resolution(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        resolution = safe_parse_resolution(h)
        assert resolution is not None
        width, height = resolution
        formatted = modules.config.add_ratio(f'{width}*{height}')
        if formatted in modules.config.available_aspect_ratios_labels:
            results.append(formatted)
            results.append(-1)
            results.append(-1)
        else:
            results.append(gr.update())
            results.append(int(width))
            results.append(int(height))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_seed(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        results.append(False)
        results.append(h)
    except:
        results.append(gr.update())
        results.append(gr.update())


def get_inpaint_engine_version(key: str, fallback: str | None, source_dict: dict, results: list, inpaint_mode: str, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str) and h in modules.flags.inpaint_engine_versions
        if inpaint_mode != modules.flags.inpaint_option_detail:
            results.append(h)
        else:
            results.append(gr.update())
        results.append(h)
        return h
    except:
        results.append(gr.update())
        results.append('empty')
        return None


def get_inpaint_method(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str) and h in modules.flags.inpaint_options
        results.append(h)
        for i in range(modules.config.default_enhance_tabs):
            results.append(h)
        return h
    except:
        results.append(gr.update())
        for i in range(modules.config.default_enhance_tabs):
            results.append(gr.update())


def get_adm_guidance(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        parsed = safe_parse_float_tuple(h, 3)
        assert parsed is not None
        p, n, e = parsed
        results.append(float(p))
        results.append(float(n))
        results.append(float(e))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_freeu(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        parsed = safe_parse_float_tuple(h, 4)
        assert parsed is not None
        b1, b2, s1, s2 = parsed
        results.append(True)
        results.append(float(b1))
        results.append(float(b2))
        results.append(float(s1))
        results.append(float(s2))
    except:
        results.append(False)
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_lora(key: str, fallback: str | None, source_dict: dict, results: list, performance_filename: str | None):
    try:
        raw_value = source_dict.get(key, source_dict.get(fallback))
        if raw_value is None:
            raise ValueError(f"Missing value for {key}")
        
        raw_str = str(raw_value).strip()
        if not raw_str or raw_str == 'None':
            raise ValueError(f"Empty value for {key}")
        
        split_data = [s.strip() for s in raw_str.split(' : ')]
        
        enabled = True
        name = ''
        weight = 1.0

        if len(split_data) == 2:
            name = split_data[0]
            weight_str = split_data[1]
        elif len(split_data) == 3:
            enabled_str = split_data[0]
            if enabled_str == 'True' or enabled_str == 'true' or enabled_str == '1':
                enabled = True
            elif enabled_str == 'False' or enabled_str == 'false' or enabled_str == '0':
                enabled = False
            else:
                enabled = bool(enabled_str)
            name = split_data[1]
            weight_str = split_data[2]
        else:
            raise ValueError(f"Invalid LoRA format: {raw_str}")

        if name == performance_filename or (performance_filename is not None and name == Path(performance_filename).stem):
            raise Exception("Skipping performance LoRA")

        if not name or name == 'None':
            raise ValueError(f"Invalid LoRA name: {name}")

        parsed_weight = safe_parse_float(weight_str)
        if parsed_weight is None:
            raise ValueError(f"Invalid LoRA weight: {weight_str}")
        weight = parsed_weight

        results.append(enabled)
        results.append(name)
        results.append(weight)
    except:
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
                    continue

                if fooocus_key in ['freeu']:
                    parsed = safe_parse_float_tuple(v_clean, 4)
                    if parsed is not None:
                        data[fooocus_key] = list(parsed)
                    continue

                if fooocus_key in ['adm_guidance']:
                    parsed = safe_parse_float_tuple(v_clean, 3)
                    if parsed is not None:
                        data[fooocus_key] = list(parsed)
                    continue

                if fooocus_key in ['styles']:
                    parsed = safe_parse_list(v_clean)
                    if parsed is not None:
                        data[fooocus_key] = parsed
                    continue

                if fooocus_key == 'seed':
                    parsed = safe_parse_int(v_clean)
                    if parsed is not None:
                        data[fooocus_key] = parsed
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
                        continue
                    lora_name = lora_items[0]
                    lora_weight = lora_items[-1]
                    matched_filename = None
                    for filename in modules.config.lora_filenames:
                        path = Path(filename)
                        if lora_name == path.stem:
                            matched_filename = filename
                            break
                    if matched_filename is not None:
                        parsed_w = safe_parse_float(lora_weight)
                        final_w = parsed_w if parsed_w is not None else lora_weight
                        data[f'lora_combined_{li + 1}'] = f'{matched_filename} : {final_w}'
                except Exception as e:
                    print(f"[A1111 to_json] Skipping LoRA entry #{li}: {e}")
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
                    result[key] = replaced if replaced is not None else value
                elif key.startswith('lora_combined_'):
                    replaced = self.replace_value_with_filename(key, value, modules.config.lora_filenames)
                    result[key] = replaced if replaced is not None else value
                elif key == 'vae':
                    replaced = self.replace_value_with_filename(key, value, modules.config.vae_filenames)
                    result[key] = replaced if replaced is not None else value
                else:
                    result[key] = value
            except Exception as e:
                print(f"[Fooocus to_json] Skipping field {key} due to error: {e}")
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
