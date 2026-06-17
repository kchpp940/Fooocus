from pathlib import Path

import numpy as np
import datetime
import random
import math
import os
import cv2
import re
from typing import List, Tuple, AnyStr, NamedTuple

import json
import hashlib

from PIL import Image

import modules.config
import modules.sdxl_styles
from modules.flags import Performance

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

# Regexp compiled once. Matches entries with the following pattern:
# <lora:some_lora:1>
# <lora:aNotherLora:-1.6>
LORAS_PROMPT_PATTERN = re.compile(r"(<lora:([^:]+):([+-]?(?:\d+(?:\.\d*)?|\.\d+))>)", re.X)

HASH_SHA256_LENGTH = 10


def erode_or_dilate(x, k):
    k = int(k)
    if k > 0:
        return cv2.dilate(x, kernel=np.ones(shape=(3, 3), dtype=np.uint8), iterations=k)
    if k < 0:
        return cv2.erode(x, kernel=np.ones(shape=(3, 3), dtype=np.uint8), iterations=-k)
    return x


def resample_image(im, width, height):
    im = Image.fromarray(im)
    im = im.resize((int(width), int(height)), resample=LANCZOS)
    return np.array(im)


def resize_image(im, width, height, resize_mode=1):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
    """

    im = Image.fromarray(im)

    def resize(im, w, h):
        return im.resize((w, h), resample=LANCZOS)

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return np.array(res)


def get_shape_ceil(h, w):
    return math.ceil(((h * w) ** 0.5) / 64.0) * 64.0


def get_image_shape_ceil(im):
    H, W = im.shape[:2]
    return get_shape_ceil(H, W)


def set_image_shape_ceil(im, shape_ceil):
    shape_ceil = float(shape_ceil)

    H_origin, W_origin, _ = im.shape
    H, W = H_origin, W_origin
    
    for _ in range(256):
        current_shape_ceil = get_shape_ceil(H, W)
        if abs(current_shape_ceil - shape_ceil) < 0.1:
            break
        k = shape_ceil / current_shape_ceil
        H = int(round(float(H) * k / 64.0) * 64)
        W = int(round(float(W) * k / 64.0) * 64)

    if H == H_origin and W == W_origin:
        return im

    return resample_image(im, width=W, height=H)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def remove_empty_str(items, default=None):
    items = [x for x in items if x != ""]
    if len(items) == 0 and default is not None:
        return [default]
    return items


def join_prompts(*args, **kwargs):
    prompts = [str(x) for x in args if str(x) != ""]
    if len(prompts) == 0:
        return ""
    if len(prompts) == 1:
        return prompts[0]
    return ', '.join(prompts)


def generate_temp_filename(folder='./outputs/', extension='png'):
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    random_number = random.randint(1000, 9999)
    filename = f"{time_string}_{random_number}.{extension}"
    result = os.path.join(folder, date_string, filename)
    return date_string, os.path.abspath(result), filename


def sha256(filename, use_addnet_hash=False, length=HASH_SHA256_LENGTH):
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256(filename)

    return sha256_value[:length] if length is not None else sha256_value


def addnet_hash_safetensors(b):
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def calculate_sha256(filename) -> str:
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)


def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text

    try:
        return json.loads(text)
    except Exception:
        return text


def unwrap_style_text_from_prompt(style_text, prompt):
    """
    Checks the prompt to see if the style text is wrapped around it. If so,
    returns True plus the prompt text without the style text. Otherwise, returns
    False with the original prompt.

    Note that the "cleaned" version of the style text is only used for matching
    purposes here. It isn't returned; the original style text is not modified.
    """
    stripped_prompt = prompt
    stripped_style_text = style_text
    if "{prompt}" in stripped_style_text:
        # Work out whether the prompt is wrapped in the style text. If so, we
        # return True and the "inner" prompt text that isn't part of the style.
        try:
            left, right = stripped_style_text.split("{prompt}", 2)
        except ValueError as e:
            # If the style text has multple "{prompt}"s, we can't split it into
            # two parts. This is an error, but we can't do anything about it.
            print(f"Unable to compare style text to prompt:\n{style_text}")
            print(f"Error: {e}")
            return False, prompt, ''

        left_pos = stripped_prompt.find(left)
        right_pos = stripped_prompt.find(right)
        if 0 <= left_pos < right_pos:
            real_prompt = stripped_prompt[left_pos + len(left):right_pos]
            prompt = stripped_prompt.replace(left + real_prompt + right, '', 1)
            if prompt.startswith(", "):
                prompt = prompt[2:]
            if prompt.endswith(", "):
                prompt = prompt[:-2]
            return True, prompt, real_prompt
    else:
        # Work out whether the given prompt starts with the style text. If so, we
        # return True and the prompt text up to where the style text starts.
        if stripped_prompt.endswith(stripped_style_text):
            prompt = stripped_prompt[: len(stripped_prompt) - len(stripped_style_text)]
            if prompt.endswith(", "):
                prompt = prompt[:-2]
            return True, prompt, prompt

    return False, prompt, ''


def extract_original_prompts(style, prompt, negative_prompt):
    """
    Takes a style and compares it to the prompt and negative prompt. If the style
    matches, returns True plus the prompt and negative prompt with the style text
    removed. Otherwise, returns False with the original prompt and negative prompt.
    """
    if not style.prompt and not style.negative_prompt:
        return False, prompt, negative_prompt

    match_positive, extracted_positive, real_prompt = unwrap_style_text_from_prompt(
        style.prompt, prompt
    )
    if not match_positive:
        return False, prompt, negative_prompt, ''

    match_negative, extracted_negative, _ = unwrap_style_text_from_prompt(
        style.negative_prompt, negative_prompt
    )
    if not match_negative:
        return False, prompt, negative_prompt, ''

    return True, extracted_positive, extracted_negative, real_prompt


def extract_styles_from_prompt(prompt, negative_prompt):
    extracted = []
    applicable_styles = []

    for style_name, (style_prompt, style_negative_prompt) in modules.sdxl_styles.styles.items():
        applicable_styles.append(PromptStyle(name=style_name, prompt=style_prompt, negative_prompt=style_negative_prompt))

    real_prompt = ''

    while True:
        found_style = None

        for style in applicable_styles:
            is_match, new_prompt, new_neg_prompt, new_real_prompt = extract_original_prompts(
                style, prompt, negative_prompt
            )
            if is_match:
                found_style = style
                prompt = new_prompt
                negative_prompt = new_neg_prompt
                if real_prompt == '' and new_real_prompt != '' and new_real_prompt != prompt:
                    real_prompt = new_real_prompt
                break

        if not found_style:
            break

        applicable_styles.remove(found_style)
        extracted.append(found_style.name)

    # add prompt expansion if not all styles could be resolved
    if prompt != '':
        if real_prompt != '':
            extracted.append(modules.sdxl_styles.fooocus_expansion)
        else:
            # find real_prompt when only prompt expansion is selected
            first_word = prompt.split(', ')[0]
            first_word_positions = [i for i in range(len(prompt)) if prompt.startswith(first_word, i)]
            if len(first_word_positions) > 1:
                real_prompt = prompt[:first_word_positions[-1]]
                extracted.append(modules.sdxl_styles.fooocus_expansion)
                if real_prompt.endswith(', '):
                    real_prompt = real_prompt[:-2]

    return list(reversed(extracted)), real_prompt, negative_prompt


class PromptStyle(NamedTuple):
    name: str
    prompt: str
    negative_prompt: str


def is_json(data: str) -> bool:
    try:
        loaded_json = json.loads(data)
        assert isinstance(loaded_json, dict)
    except (ValueError, AssertionError):
        return False
    return True


def get_filname_by_stem(lora_name, filenames: List[str]) -> str | None:
    for filename in filenames:
        path = Path(filename)
        if lora_name == path.stem:
            return filename
    return None


def get_file_from_folder_list(name, folders):
    if not isinstance(folders, list):
        folders = [folders]

    for folder in folders:
        filename = os.path.abspath(os.path.realpath(os.path.join(folder, name)))
        if os.path.isfile(filename):
            return filename

    return os.path.abspath(os.path.realpath(os.path.join(folders[0], name)))


def get_enabled_loras(loras: list, remove_none=True) -> list:
    return [(lora[1], lora[2]) for lora in loras if lora[0] and (lora[1] != 'None' if remove_none else True)]


def parse_lora_references_from_prompt(prompt: str, loras: List[Tuple[AnyStr, float]], loras_limit: int = 5,
                                      skip_file_check=False, prompt_cleanup=True, deduplicate_loras=True,
                                      lora_filenames=None) -> tuple[List[Tuple[AnyStr, float]], str]:
    # prevent unintended side effects when returning without detection
    loras = loras.copy()

    if lora_filenames is None:
        lora_filenames = []

    found_loras = []
    prompt_without_loras = ''
    cleaned_prompt = ''

    for token in prompt.split(','):
        matches = LORAS_PROMPT_PATTERN.findall(token)

        if len(matches) == 0:
            prompt_without_loras += token + ', '
            continue
        for match in matches:
            lora_name = match[1] + '.safetensors'
            if not skip_file_check:
                lora_name = get_filname_by_stem(match[1], lora_filenames)
            if lora_name is not None:
                found_loras.append((lora_name, float(match[2])))
            token = token.replace(match[0], '')
        prompt_without_loras += token + ', '

    if prompt_without_loras != '':
        cleaned_prompt = prompt_without_loras[:-2]

    if prompt_cleanup:
        cleaned_prompt = cleanup_prompt(prompt_without_loras)

    new_loras = []
    lora_names = [lora[0] for lora in loras]
    for found_lora in found_loras:
        if deduplicate_loras and (found_lora[0] in lora_names or found_lora in new_loras):
            continue
        new_loras.append(found_lora)

    if len(new_loras) == 0:
        return loras, cleaned_prompt

    updated_loras = []
    for lora in loras + new_loras:
        if lora[0] != "None":
            updated_loras.append(lora)

    return updated_loras[:loras_limit], cleaned_prompt


def remove_performance_lora(filenames: list, performance: Performance | None):
    loras_without_performance = filenames.copy()

    if performance is None:
        return loras_without_performance

    performance_lora = performance.lora_filename()

    for filename in filenames:
        path = Path(filename)
        if performance_lora == path.name:
            loras_without_performance.remove(filename)

    return loras_without_performance


def cleanup_prompt(prompt):
    prompt = re.sub(' +', ' ', prompt)
    prompt = re.sub(',+', ',', prompt)
    cleaned_prompt = ''
    for token in prompt.split(','):
        token = token.strip()
        if token == '':
            continue
        cleaned_prompt += token + ', '
    return cleaned_prompt[:-2]


def apply_wildcards(wildcard_text, rng, i, read_wildcards_in_order) -> str:
    for _ in range(modules.config.wildcards_max_bfs_depth):
        placeholders = re.findall(r'__([\w-]+)__', wildcard_text)
        if len(placeholders) == 0:
            return wildcard_text

        print(f'[Wildcards] processing: {wildcard_text}')
        for placeholder in placeholders:
            try:
                matches = [x for x in modules.config.wildcard_filenames if os.path.splitext(os.path.basename(x))[0] == placeholder]
                words = open(os.path.join(modules.config.path_wildcards, matches[0]), encoding='utf-8').read().splitlines()
                words = [x for x in words if x != '']
                assert len(words) > 0
                if read_wildcards_in_order:
                    wildcard_text = wildcard_text.replace(f'__{placeholder}__', words[i % len(words)], 1)
                else:
                    wildcard_text = wildcard_text.replace(f'__{placeholder}__', rng.choice(words), 1)
            except:
                print(f'[Wildcards] Warning: {placeholder}.txt missing or empty. '
                      f'Using "{placeholder}" as a normal word.')
                wildcard_text = wildcard_text.replace(f'__{placeholder}__', placeholder)
            print(f'[Wildcards] {wildcard_text}')

    print(f'[Wildcards] BFS stack overflow. Current text: {wildcard_text}')
    return wildcard_text


def get_image_size_info(image: np.ndarray, aspect_ratios: list) -> str:
    try:
        image = Image.fromarray(np.uint8(image))
        width, height = image.size
        ratio = round(width / height, 2)
        gcd = math.gcd(width, height)
        lcm_ratio = f'{width // gcd}:{height // gcd}'
        size_info = f'Image Size: {width} x {height}, Ratio: {ratio}, {lcm_ratio}'

        closest_ratio = min(aspect_ratios, key=lambda x: abs(ratio - float(x.split('*')[0]) / float(x.split('*')[1])))
        recommended_width, recommended_height = map(int, closest_ratio.split('*'))
        recommended_ratio = round(recommended_width / recommended_height, 2)
        recommended_gcd = math.gcd(recommended_width, recommended_height)
        recommended_lcm_ratio = f'{recommended_width // recommended_gcd}:{recommended_height // recommended_gcd}'

        size_info = f'{width} x {height}, {ratio}, {lcm_ratio}'
        size_info += f'\n{recommended_width} x {recommended_height}, {recommended_ratio}, {recommended_lcm_ratio}'

        return size_info
    except Exception as e:
        return f'Error reading image: {e}'


SUPPORTED_PARAM_BINDINGS = {
    'seed': {'type': int, 'display': 'Seed'},
    'cfg': {'type': float, 'alias': ['cfg_scale'], 'display': 'Guidance Scale'},
    'cfg_scale': {'type': float, 'display': 'Guidance Scale'},
    'sampler': {'type': str, 'display': 'Sampler'},
    'scheduler': {'type': str, 'display': 'Scheduler'},
    'steps': {'type': int, 'display': 'Steps'},
    'style': {'type': str, 'is_multi': True, 'alias': ['styles'], 'display': 'Styles'},
    'styles': {'type': str, 'is_multi': True, 'display': 'Styles'},
    'width': {'type': int, 'display': 'Width'},
    'height': {'type': int, 'display': 'Height'},
    'sharpness': {'type': float, 'display': 'Sharpness'},
    'refiner_switch': {'type': float, 'display': 'Refiner Switch'},
}


def parse_prompt_matrix_config(matrix_config_text: str) -> list:
    if not matrix_config_text or not matrix_config_text.strip():
        return []

    try:
        parsed = json.loads(matrix_config_text)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict) and 'name' in item and 'values' in item:
                    name = str(item['name']).strip()
                    values = item['values']
                    if isinstance(values, str):
                        values = [v.strip() for v in values.split(',') if v.strip()]
                    elif isinstance(values, list):
                        values = [str(v).strip() for v in values if str(v).strip() != '']
                    if name and len(values) > 0:
                        entry = {'name': name, 'values': values}
                        if item.get('is_param_override') or name.startswith('@'):
                            _mark_param_binding(entry)
                        result.append(entry)
            return result
    except json.JSONDecodeError:
        pass

    result = []
    for line in matrix_config_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            name_part, values_part = line.split(':', 1)
            name = name_part.strip()
            values = [v.strip() for v in values_part.split(',') if v.strip()]
            if name and len(values) > 0:
                entry = {'name': name, 'values': values}
                if name.startswith('@'):
                    _mark_param_binding(entry)
                result.append(entry)
    return result


def _mark_param_binding(entry: dict):
    raw_name = entry['name']
    target_name = raw_name[1:].strip()
    entry['is_param_override'] = True
    spec = _resolve_param_binding_spec(target_name)
    entry['param_target'] = spec['canonical']
    entry['param_spec'] = {'type': spec['type'], 'display': spec['display'],
                           'is_multi': spec.get('is_multi', False),
                           'is_lora_weight': spec.get('is_lora_weight', False),
                           'lora_index': spec.get('lora_index')}
    if spec.get('type'):
        typed_values = []
        for v in entry['values']:
            try:
                if spec['type'] == int:
                    typed_values.append(int(float(v)))
                elif spec['type'] == float:
                    typed_values.append(float(v))
                else:
                    typed_values.append(v)
            except (ValueError, TypeError):
                typed_values.append(v)
        entry['values'] = typed_values


def _resolve_param_binding_spec(target_name: str) -> dict:
    tn = target_name.lower()
    for canonical, spec in SUPPORTED_PARAM_BINDINGS.items():
        aliases = [canonical] + spec.get('alias', [])
        if tn in aliases:
            return {'canonical': canonical, 'type': spec.get('type'),
                    'display': spec.get('display', canonical),
                    'is_multi': spec.get('is_multi', False)}
    if tn.startswith('lora_weight_'):
        try:
            idx = int(tn.split('_')[-1])
            return {'canonical': f'lora_weight_{idx}', 'type': float,
                    'display': f'LoRA {idx} Weight',
                    'is_lora_weight': True, 'lora_index': idx}
        except (ValueError, IndexError):
            pass
    if tn.startswith('lora_') and '_weight_' in tn:
        try:
            idx = int(tn.split('_weight_')[-1])
            return {'canonical': f'lora_weight_{idx}', 'type': float,
                    'display': f'LoRA {idx} Weight',
                    'is_lora_weight': True, 'lora_index': idx}
        except (ValueError, IndexError):
            pass
    return {'canonical': target_name, 'type': None, 'display': target_name}


def get_matrix_combinations(matrix_config: list) -> list:
    if not matrix_config:
        return [{}]

    def _expand(index, current):
        if index >= len(matrix_config):
            return [current.copy()]
        results = []
        var = matrix_config[index]
        for value in var['values']:
            current[var['name']] = value
            results.extend(_expand(index + 1, current))
            del current[var['name']]
        return results

    return _expand(0, {})


def apply_prompt_variables(text: str, variables: dict) -> str:
    if not text or not variables:
        return text
    result = text
    for name, value in variables.items():
        if name.startswith('@'):
            continue
        placeholder = '{' + name + '}'
        result = result.replace(placeholder, str(value))
    return result


def extract_param_overrides(matrix_config: list, variables: dict) -> dict:
    if not matrix_config or not variables:
        return {}
    overrides = {}
    config_by_name = {c['name']: c for c in matrix_config}
    for var_name, var_value in variables.items():
        if var_name not in config_by_name:
            continue
        cfg = config_by_name[var_name]
        if not cfg.get('is_param_override'):
            continue
        target = cfg.get('param_target', var_name.lstrip('@'))
        spec = cfg.get('param_spec', {})
        overrides[target] = {
            'value': var_value,
            'spec': spec,
            'display_name': spec.get('display', target),
            'raw_name': var_name
        }
    return overrides


def get_matrix_combination_count(matrix_config: list) -> int:
    if not matrix_config:
        return 1
    count = 1
    for var in matrix_config:
        count *= len(var.get('values', []))
    return count


def validate_matrix_config(matrix_config: list, max_combinations: int = 512,
                           max_variables: int = 20, image_number: int = 1) -> dict:
    result = {'valid': True, 'errors': [], 'warnings': [],
              'total_combinations': 0, 'total_tasks': 0}

    if not matrix_config:
        result['total_combinations'] = 1
        result['total_tasks'] = image_number
        return result

    result['total_combinations'] = get_matrix_combination_count(matrix_config)
    result['total_tasks'] = result['total_combinations'] * image_number

    if len(matrix_config) > max_variables:
        result['errors'].append(
            f'Too many variables: {len(matrix_config)} (max {max_variables})')
        result['valid'] = False

    if result['total_combinations'] > max_combinations:
        result['errors'].append(
            f'Too many matrix combinations: {result["total_combinations"]} '
            f'(max {max_combinations}). Reduce variables or values.')
        result['valid'] = False

    for var in matrix_config:
        name = var.get('name', '')
        values = var.get('values', [])

        if not name or not name.strip():
            result['errors'].append('Variable with empty name found')
            result['valid'] = False
            continue

        if not values or len(values) == 0:
            result['errors'].append(f'Variable "{name}" has no values')
            result['valid'] = False
            continue

        if any(v is None or (isinstance(v, str) and not v.strip()) for v in values):
            result['errors'].append(f'Variable "{name}" contains empty values')
            result['valid'] = False

        if len(values) > 100:
            result['warnings'].append(
                f'Variable "{name}" has {len(values)} values, may take very long')

        if var.get('is_param_override'):
            _validate_param_binding(var, result)

    if result['errors']:
        result['valid'] = False

    return result


def _validate_param_binding(var: dict, result: dict):
    name = var['name']
    target = var.get('param_target', name.lstrip('@'))
    spec = var.get('param_spec', {})
    values = var.get('values', [])

    if target in ('seed',):
        for v in values:
            try:
                iv = int(float(v))
                if iv < 0 or iv > 2**32 - 1:
                    result['errors'].append(
                        f'{name}: seed value "{v}" out of valid range [0, 2^32-1]')
                    result['valid'] = False
            except (ValueError, TypeError):
                result['errors'].append(f'{name}: invalid seed value "{v}" (must be integer)')
                result['valid'] = False

    elif target in ('cfg', 'cfg_scale'):
        for v in values:
            try:
                fv = float(v)
                if fv < 0.0 or fv > 50.0:
                    result['warnings'].append(
                        f'{name}: cfg value "{v}" is outside typical range [0, 50]')
            except (ValueError, TypeError):
                result['errors'].append(f'{name}: invalid cfg value "{v}" (must be number)')
                result['valid'] = False

    elif target == 'steps':
        for v in values:
            try:
                iv = int(float(v))
                if iv < 1 or iv > 200:
                    result['warnings'].append(
                        f'{name}: steps value "{v}" is outside typical range [1, 200]')
            except (ValueError, TypeError):
                result['errors'].append(f'{name}: invalid steps value "{v}" (must be integer)')
                result['valid'] = False

    elif target in ('width', 'height'):
        for v in values:
            try:
                iv = int(float(v))
                if iv < 128 or iv > 4096:
                    result['warnings'].append(
                        f'{name}: {target} value "{v}" is outside typical range [128, 4096]')
                if iv % 8 != 0:
                    result['warnings'].append(
                        f'{name}: {target} value "{v}" is not a multiple of 8, may cause issues')
            except (ValueError, TypeError):
                result['errors'].append(f'{name}: invalid {target} value "{v}" (must be integer)')
                result['valid'] = False

    elif target == 'sharpness':
        for v in values:
            try:
                fv = float(v)
                if fv < 0.0 or fv > 10.0:
                    result['warnings'].append(
                        f'{name}: sharpness value "{v}" is outside typical range [0, 10]')
            except (ValueError, TypeError):
                result['errors'].append(f'{name}: invalid sharpness value "{v}" (must be number)')
                result['valid'] = False

    elif target == 'refiner_switch':
        for v in values:
            try:
                fv = float(v)
                if fv < 0.0 or fv > 1.0:
                    result['warnings'].append(
                        f'{name}: refiner_switch value "{v}" is outside valid range [0, 1]')
            except (ValueError, TypeError):
                result['errors'].append(f'{name}: invalid refiner_switch value "{v}" (must be number)')
                result['valid'] = False

    elif target == 'sampler':
        try:
            from modules.flags import sampler_list
            valid_samplers = set(sampler_list)
            valid_lower = {s.lower(): s for s in sampler_list}
            for v in values:
                vs = str(v)
                if vs not in valid_samplers and vs.lower() not in valid_lower:
                    result['errors'].append(
                        f'{name}: unknown sampler "{vs}". Valid samplers: {", ".join(sampler_list[:8])}...')
                    result['valid'] = False
        except Exception as e:
            result['warnings'].append(f'Cannot validate sampler list: {e}')

    elif target == 'scheduler':
        try:
            from modules.flags import scheduler_list
            valid_schedulers = set(scheduler_list)
            valid_lower = {s.lower(): s for s in scheduler_list}
            for v in values:
                vs = str(v)
                if vs not in valid_schedulers and vs.lower() not in valid_lower:
                    result['errors'].append(
                        f'{name}: unknown scheduler "{vs}". Valid schedulers: {", ".join(scheduler_list)}')
                    result['valid'] = False
        except Exception as e:
            result['warnings'].append(f'Cannot validate scheduler list: {e}')

    elif target in ('style', 'styles'):
        try:
            from modules.sdxl_styles import styles as sdxl_styles
            valid_styles = set(sdxl_styles.keys())
            valid_lower = {s.lower(): s for s in sdxl_styles.keys()}
            for v in values:
                vs = str(v)
                style_names = [s.strip() for s in vs.split('|') if s.strip()]
                if not style_names:
                    result['errors'].append(f'{name}: empty style value "{vs}"')
                    result['valid'] = False
                    continue
                for sn in style_names:
                    if sn not in valid_styles and sn.lower() not in valid_lower:
                        result['errors'].append(
                            f'{name}: unknown style "{sn}". '
                            f'Use style names from the Style dropdown.')
                        result['valid'] = False
        except Exception as e:
            result['warnings'].append(f'Cannot validate style list: {e}')

    elif spec.get('is_lora_weight'):
        lora_idx = spec.get('lora_index')
        if lora_idx is None or lora_idx < 1:
            result['errors'].append(f'{name}: invalid LoRA index')
            result['valid'] = False
        try:
            from modules.config import default_max_lora_number
            if lora_idx is not None and lora_idx > default_max_lora_number:
                result['errors'].append(
                    f'{name}: LoRA index {lora_idx} exceeds max {default_max_lora_number}')
                result['valid'] = False
        except Exception:
            pass
        for v in values:
            try:
                fv = float(v)
                if fv < -2.0 or fv > 2.0:
                    result['warnings'].append(
                        f'{name}: LoRA weight "{v}" is outside typical range [-2, 2]')
            except (ValueError, TypeError):
                result['errors'].append(
                    f'{name}: invalid LoRA weight value "{v}" (must be number)')
                result['valid'] = False

