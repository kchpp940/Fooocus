import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import gradio as gr
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


def load_parameter_button_click(raw_metadata: dict | str, is_generating: bool, inpaint_mode: str):
    loaded_parameter_dict = raw_metadata
    if isinstance(raw_metadata, str):
        loaded_parameter_dict = json.loads(raw_metadata)
    assert isinstance(loaded_parameter_dict, dict)

    results = [len(loaded_parameter_dict) > 0]

    get_image_number('image_number', 'Image Number', loaded_parameter_dict, results)
    get_str('prompt', 'Prompt', loaded_parameter_dict, results)
    get_str('negative_prompt', 'Negative Prompt', loaded_parameter_dict, results)
    get_list('styles', 'Styles', loaded_parameter_dict, results)
    performance = get_str('performance', 'Performance', loaded_parameter_dict, results)
    get_steps('steps', 'Steps', loaded_parameter_dict, results)
    get_number('overwrite_switch', 'Overwrite Switch', loaded_parameter_dict, results)
    get_resolution('resolution', 'Resolution', loaded_parameter_dict, results)
    get_number('guidance_scale', 'Guidance Scale', loaded_parameter_dict, results)
    get_number('sharpness', 'Sharpness', loaded_parameter_dict, results)
    get_adm_guidance('adm_guidance', 'ADM Guidance', loaded_parameter_dict, results)
    get_str('refiner_swap_method', 'Refiner Swap Method', loaded_parameter_dict, results)
    get_number('adaptive_cfg', 'CFG Mimicking from TSNR', loaded_parameter_dict, results)
    get_number('clip_skip', 'CLIP Skip', loaded_parameter_dict, results, cast_type=int)
    get_str('base_model', 'Base Model', loaded_parameter_dict, results)
    get_str('refiner_model', 'Refiner Model', loaded_parameter_dict, results)
    get_number('refiner_switch', 'Refiner Switch', loaded_parameter_dict, results)
    get_str('sampler', 'Sampler', loaded_parameter_dict, results)
    get_str('scheduler', 'Scheduler', loaded_parameter_dict, results)
    get_str('vae', 'VAE', loaded_parameter_dict, results)
    get_seed('seed', 'Seed', loaded_parameter_dict, results)
    get_inpaint_engine_version('inpaint_engine_version', 'Inpaint Engine Version', loaded_parameter_dict, results, inpaint_mode)
    get_inpaint_method('inpaint_method', 'Inpaint Mode', loaded_parameter_dict, results)

    if is_generating:
        results.append(gr.update())
    else:
        results.append(gr.update(visible=True))

    results.append(gr.update(visible=False))

    get_freeu('freeu', 'FreeU', loaded_parameter_dict, results)

    # prevent performance LoRAs to be added twice, by performance and by lora
    performance_filename = None
    if performance is not None and performance in Performance.values():
        performance = Performance(performance)
        performance_filename = performance.lora_filename()

    for i in range(modules.config.default_max_lora_number):
        get_lora(f'lora_combined_{i + 1}', f'LoRA {i + 1}', loaded_parameter_dict, results, performance_filename)

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
        h = eval(h)
        assert isinstance(h, list)
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
        p, n, e = eval(h)
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
        b1, b2, s1, s2 = eval(h)
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
        split_data = source_dict.get(key, source_dict.get(fallback)).split(' : ')
        enabled = True
        name = split_data[0]
        weight = split_data[1]

        if len(split_data) == 3:
            enabled = split_data[0] == 'True'
            name = split_data[1]
            weight = split_data[2]

        if name == performance_filename:
            raise Exception

        weight = float(weight)
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

    def to_json(self, metadata: str) -> dict:
        metadata_prompt = ''
        metadata_negative_prompt = ''

        done_with_prompt = False

        *lines, lastline = metadata.strip().split("\n")
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

        data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt
        }

        for k, v in re_param.findall(lastline):
            try:
                if v != '' and v[0] == '"' and v[-1] == '"':
                    v = unquote(v)

                m = re_imagesize.match(v)
                if m is not None:
                    data['resolution'] = str((m.group(1), m.group(2)))
                else:
                    data[list(self.fooocus_to_a1111.keys())[list(self.fooocus_to_a1111.values()).index(k)]] = v
            except Exception:
                print(f"Error parsing \"{k}: {v}\"")

        # workaround for multiline prompts
        if 'raw_prompt' in data:
            data['prompt'] = data['raw_prompt']
            raw_prompt = data['raw_prompt'].replace("\n", ', ')
            if metadata_prompt != raw_prompt and modules.sdxl_styles.fooocus_expansion not in found_styles:
                found_styles.append(modules.sdxl_styles.fooocus_expansion)

        if 'raw_negative_prompt' in data:
            data['negative_prompt'] = data['raw_negative_prompt']

        data['styles'] = str(found_styles)

        # try to load performance based on steps, fallback for direct A1111 imports
        if 'steps' in data and 'performance' in data is None:
            try:
                data['performance'] = Performance.by_steps(data['steps']).value
            except ValueError | KeyError:
                pass

        if 'sampler' in data:
            data['sampler'] = data['sampler'].replace(' Karras', '')
            # get key
            for k, v in SAMPLERS.items():
                if v == data['sampler']:
                    data['sampler'] = k
                    break

        for key in ['base_model', 'refiner_model', 'vae']:
            if key in data:
                if key == 'vae':
                    self.add_extension_to_filename(data, modules.config.vae_filenames, 'vae')
                else:
                    self.add_extension_to_filename(data, modules.config.model_filenames, key)

        lora_data = ''
        if 'lora_weights' in data and data['lora_weights'] != '':
            lora_data = data['lora_weights']
        elif 'lora_hashes' in data and data['lora_hashes'] != '' and data['lora_hashes'].split(', ')[0].count(':') == 2:
            lora_data = data['lora_hashes']

        if lora_data != '':
            for li, lora in enumerate(lora_data.split(', ')):
                lora_split = lora.split(': ')
                lora_name = lora_split[0]
                lora_weight = lora_split[2] if len(lora_split) == 3 else lora_split[1]
                for filename in modules.config.lora_filenames:
                    path = Path(filename)
                    if lora_name == path.stem:
                        data[f'lora_combined_{li + 1}'] = f'{filename} : {lora_weight}'
                        break

        return data

    def to_string(self, metadata: dict) -> str:
        data = {k: v for _, k, v in metadata}

        width, height = eval(data['resolution'])

        sampler = data['sampler']
        scheduler = data['scheduler']

        if sampler in SAMPLERS and SAMPLERS[sampler] != '':
            sampler = SAMPLERS[sampler]
            if sampler not in CIVITAI_NO_KARRAS and scheduler == 'karras':
                sampler += f' Karras'

        generation_params = {
            self.fooocus_to_a1111['steps']: self.steps,
            self.fooocus_to_a1111['sampler']: sampler,
            self.fooocus_to_a1111['seed']: data['seed'],
            self.fooocus_to_a1111['resolution']: f'{width}x{height}',
            self.fooocus_to_a1111['guidance_scale']: data['guidance_scale'],
            self.fooocus_to_a1111['sharpness']: data['sharpness'],
            self.fooocus_to_a1111['adm_guidance']: data['adm_guidance'],
            self.fooocus_to_a1111['base_model']: Path(data['base_model']).stem,
            self.fooocus_to_a1111['base_model_hash']: self.base_model_hash,

            self.fooocus_to_a1111['performance']: data['performance'],
            self.fooocus_to_a1111['scheduler']: scheduler,
            self.fooocus_to_a1111['vae']: Path(data['vae']).stem,
            # workaround for multiline prompts
            self.fooocus_to_a1111['raw_prompt']: self.raw_prompt,
            self.fooocus_to_a1111['raw_negative_prompt']: self.raw_negative_prompt,
        }

        if self.refiner_model_name not in ['', 'None']:
            generation_params |= {
                self.fooocus_to_a1111['refiner_model']: self.refiner_model_name,
                self.fooocus_to_a1111['refiner_model_hash']: self.refiner_model_hash
            }

        for key in ['adaptive_cfg', 'clip_skip', 'overwrite_switch', 'refiner_swap_method', 'freeu']:
            if key in data:
                generation_params[self.fooocus_to_a1111[key]] = data[key]

        if len(self.loras) > 0:
            lora_hashes = []
            lora_weights = []
            for index, (lora_name, lora_weight, lora_hash) in enumerate(self.loras):
                # workaround for Fooocus not knowing LoRA name in LoRA metadata
                lora_hashes.append(f'{lora_name}: {lora_hash}')
                lora_weights.append(f'{lora_name}: {lora_weight}')
            lora_hashes_string = ', '.join(lora_hashes)
            lora_weights_string = ', '.join(lora_weights)
            generation_params[self.fooocus_to_a1111['lora_hashes']] = lora_hashes_string
            generation_params[self.fooocus_to_a1111['lora_weights']] = lora_weights_string

        generation_params[self.fooocus_to_a1111['version']] = data['version']

        if modules.config.metadata_created_by != '':
            generation_params[self.fooocus_to_a1111['created_by']] = modules.config.metadata_created_by

        generation_params_text = ", ".join(
            [k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if
             v is not None])
        positive_prompt_resolved = ', '.join(self.full_prompt)
        negative_prompt_resolved = ', '.join(self.full_negative_prompt)
        negative_prompt_text = f"\nNegative prompt: {negative_prompt_resolved}" if negative_prompt_resolved else ""
        return f"{positive_prompt_resolved}{negative_prompt_text}\n{generation_params_text}".strip()

    @staticmethod
    def add_extension_to_filename(data, filenames, key):
        for filename in filenames:
            path = Path(filename)
            if data[key] == path.stem:
                data[key] = filename
                break


class FooocusMetadataParser(MetadataParser):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.FOOOCUS

    def to_json(self, metadata: dict) -> dict:
        for key, value in metadata.items():
            if value in ['', 'None']:
                continue
            if key in ['base_model', 'refiner_model']:
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.model_filenames)
            elif key.startswith('lora_combined_'):
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.lora_filenames)
            elif key == 'vae':
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.vae_filenames)
            else:
                continue

        return metadata

    def to_string(self, metadata: list) -> str:
        for li, (label, key, value) in enumerate(metadata):
            # remove model folder paths from metadata
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                name = Path(name).stem
                value = f'{name} : {weight}'
                metadata[li] = (label, key, value)

        res = {k: v for _, k, v in metadata}

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

        return json.dumps(dict(sorted(res.items())))

    @staticmethod
    def replace_value_with_filename(key, value, filenames):
        for filename in filenames:
            path = Path(filename)
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                if name == path.stem:
                    return f'{filename} : {weight}'
            elif value == path.stem:
                return filename

        return None


def get_metadata_parser(metadata_scheme: MetadataScheme) -> MetadataParser:
    match metadata_scheme:
        case MetadataScheme.FOOOCUS:
            return FooocusMetadataParser()
        case MetadataScheme.A1111:
            return A1111MetadataParser()
        case _:
            raise NotImplementedError


def read_info_from_image(file) -> tuple[str | None, MetadataScheme | None]:
    items = (file.info or {}).copy()

    parameters = items.pop('parameters', None)
    metadata_scheme = items.pop('fooocus_scheme', None)
    exif = items.pop('exif', None)

    if parameters is not None and is_json(parameters):
        parameters = json.loads(parameters)
    elif exif is not None:
        exif = file.getexif()
        # 0x9286 = UserComment
        parameters = exif.get(0x9286, None)
        # 0x927C = MakerNote
        metadata_scheme = exif.get(0x927C, None)

        if is_json(parameters):
            parameters = json.loads(parameters)

    try:
        metadata_scheme = MetadataScheme(metadata_scheme)
    except ValueError:
        metadata_scheme = None

        # broad fallback
        if isinstance(parameters, dict):
            metadata_scheme = MetadataScheme.FOOOCUS

        if isinstance(parameters, str):
            metadata_scheme = MetadataScheme.A1111

    return parameters, metadata_scheme


def get_exif(metadata: str | None, metadata_scheme: str):
    exif = Image.Exif()
    # tags see see https://github.com/python-pillow/Pillow/blob/9.2.x/src/PIL/ExifTags.py
    # 0x9286 = UserComment
    exif[0x9286] = metadata
    # 0x0131 = Software
    exif[0x0131] = 'Fooocus v' + fooocus_version.version
    # 0x927C = MakerNote
    exif[0x927C] = metadata_scheme
    return exif


COMPARISON_FIELDS = [
    ('prompt', 'Prompt', str),
    ('negative_prompt', 'Negative Prompt', str),
    ('styles', 'Styles', list),
    ('performance', 'Performance', str),
    ('steps', 'Steps', int),
    ('base_model', 'Base Model', str),
    ('refiner_model', 'Refiner Model', str),
    ('refiner_switch', 'Refiner Switch', float),
    ('sampler', 'Sampler', str),
    ('scheduler', 'Scheduler', str),
    ('vae', 'VAE', str),
    ('guidance_scale', 'CFG Scale', float),
    ('sharpness', 'Sharpness', float),
    ('adaptive_cfg', 'Adaptive CFG', float),
    ('clip_skip', 'CLIP Skip', int),
    ('seed', 'Seed', int),
    ('resolution', 'Resolution', tuple),
    ('inpaint_engine_version', 'Inpaint Engine', str),
    ('inpaint_method', 'Inpaint Method', str),
    ('adm_guidance', 'ADM Guidance', tuple),
    ('freeu', 'FreeU', tuple),
]


STANDARD_APPLY_KEYS = [
    'image_number',
    'prompt',
    'negative_prompt',
    'styles',
    'performance',
    'steps',
    'overwrite_switch',
    'resolution',
    'guidance_scale',
    'sharpness',
    'adm_guidance',
    'refiner_swap_method',
    'adaptive_cfg',
    'clip_skip',
    'base_model',
    'refiner_model',
    'refiner_switch',
    'sampler',
    'scheduler',
    'vae',
    'seed',
    'inpaint_engine_version',
    'inpaint_method',
    'freeu',
]


def _get_standard_lora_keys():
    try:
        n = modules.config.default_max_lora_number
    except Exception:
        n = 5
    return [f'lora_combined_{i + 1}' for i in range(n)]


def _normalize_model_filename(value, filenames_list):
    try:
        if value in (None, '', 'None'):
            return value
        for filename in filenames_list:
            path = Path(filename)
            if value == filename or value == str(path):
                return filename
            if value == path.stem or value == path.name:
                return filename
        return value
    except Exception:
        return value


def _normalize_lora_filename(value, filenames_list):
    try:
        if value in (None, '', 'None'):
            return value
        if isinstance(value, str) and ' : ' in value:
            parts = value.split(' : ')
            if len(parts) == 2:
                name, weight = parts
                for filename in filenames_list:
                    path = Path(filename)
                    if name == filename or name == str(path) or name == path.stem or name == path.name:
                        return f'{filename} : {weight}'
            elif len(parts) == 3:
                enabled, name, weight = parts
                for filename in filenames_list:
                    path = Path(filename)
                    if name == filename or name == str(path) or name == path.stem or name == path.name:
                        return f'{enabled} : {filename} : {weight}'
        return value
    except Exception:
        return value


def _has_valid_field(data, key):
    if not isinstance(data, dict):
        return False
    if key not in data:
        return False
    val = data[key]
    return val is not None and val != '' and val != 'None'


def safe_extract_field(data: dict, key: str, fallback: str, field_type):
    try:
        if not isinstance(data, dict):
            return None, False
        raw = data.get(key, data.get(fallback))
        if raw is None or raw == '' or raw == 'None':
            return None, False
        if field_type == str:
            return str(raw), True
        elif field_type == int:
            return int(raw), True
        elif field_type == float:
            return float(raw), True
        elif field_type == list:
            if isinstance(raw, str):
                parsed = eval(raw)
                if isinstance(parsed, list):
                    return parsed, True
            elif isinstance(raw, list):
                return raw, True
            return None, False
        elif field_type == tuple:
            if isinstance(raw, str):
                parsed = eval(raw)
                if isinstance(parsed, (tuple, list)):
                    return tuple(parsed), True
            elif isinstance(raw, (tuple, list)):
                return tuple(raw), True
            return None, False
        return None, False
    except Exception:
        return None, False


def safe_extract_loras(data: dict) -> list:
    result = []
    try:
        if not isinstance(data, dict):
            return result
        max_loras = modules.config.default_max_lora_number
        for i in range(max_loras):
            key = f'lora_combined_{i + 1}'
            fallback = f'LoRA {i + 1}'
            try:
                raw = data.get(key, data.get(fallback))
                if raw is None or raw == '' or raw == 'None':
                    continue
                if isinstance(raw, str):
                    split_data = raw.split(' : ')
                    if len(split_data) == 2:
                        name, weight = split_data
                        weight = float(weight)
                        if name != 'None':
                            result.append({'enabled': True, 'name': Path(name).stem, 'weight': weight, 'raw': raw})
                    elif len(split_data) == 3:
                        enabled = split_data[0] == 'True'
                        name = split_data[1]
                        weight = float(split_data[2])
                        if name != 'None':
                            result.append({'enabled': enabled, 'name': Path(name).stem, 'weight': weight, 'raw': raw})
            except Exception:
                continue
    except Exception:
        pass
    return result


def build_standard_apply_metadata(image_path: str | None, embedded_parsed: dict | None,
                                  log_parsed: dict | None, metadata_scheme=None) -> tuple[dict, dict, list]:
    """
    构建标准回填 metadata，返回三元组：
      (apply_metadata: dict, apply_sources: dict, dropped_keys: list)

    - apply_metadata: 仅含白名单 key 的可回填参数字典
    - apply_sources: 每个 key 的来源: embedded / log / default
    - dropped_keys: 输入中有但不在白名单里、被丢弃的 key 列表
    """
    apply_data = {}
    apply_sources = {}
    dropped_keys = []

    lora_keys = _get_standard_lora_keys()
    all_apply_keys = set(list(STANDARD_APPLY_KEYS) + lora_keys)

    try:
        model_filenames = modules.config.model_filenames
    except Exception:
        model_filenames = []
    try:
        vae_filenames = modules.config.vae_filenames
    except Exception:
        vae_filenames = []
    try:
        lora_filenames = modules.config.lora_filenames
    except Exception:
        lora_filenames = []

    def _normalize_value(key, val):
        try:
            if key in ('base_model', 'refiner_model'):
                return _normalize_model_filename(val, model_filenames)
            elif key == 'vae':
                return _normalize_model_filename(val, vae_filenames)
            elif key.startswith('lora_combined_'):
                return _normalize_lora_filename(val, lora_filenames)
        except Exception:
            pass
        return val

    if embedded_parsed is not None and isinstance(embedded_parsed, dict):
        for key, val in embedded_parsed.items():
            if key in all_apply_keys:
                if _has_valid_field(embedded_parsed, key):
                    norm_val = _normalize_value(key, val)
                    apply_data[key] = norm_val
                    apply_sources[key] = 'embedded'
            else:
                dropped_keys.append(key)

    if log_parsed is not None and isinstance(log_parsed, dict):
        for key, val in log_parsed.items():
            if key in all_apply_keys:
                if _has_valid_field(apply_data, key):
                    continue
                if not _has_valid_field(log_parsed, key):
                    continue
                norm_val = _normalize_value(key, val)
                apply_data[key] = norm_val
                apply_sources[key] = 'log'
            else:
                if key not in dropped_keys:
                    dropped_keys.append(key)

    if not _has_valid_field(apply_data, 'image_number'):
        apply_data['image_number'] = 1
        apply_sources['image_number'] = 'default'

    return apply_data, apply_sources, dropped_keys


def extract_comparison_data(file) -> dict:
    result = {
        'fields': {},
        'loras': [],
        'raw_metadata': None,
        'log_metadata': None,
        'merged_metadata': None,
        'apply_metadata': {},
        'apply_sources': {},
        'apply_dropped_keys': [],
        'apply_source_summary': {'embedded': 0, 'log': 0, 'default': 0},
        'metadata_scheme': None,
        'metadata_source': 'none',
        'image_path': None,
        'error': None
    }
    try:
        if file is None:
            result['error'] = 'No image provided'
            return result
        image_path = None
        if isinstance(file, str):
            result['image_path'] = file
            image_path = file
            with Image.open(file) as img:
                parameters, metadata_scheme = read_info_from_image(img)
        else:
            parameters, metadata_scheme = read_info_from_image(file)
        result['metadata_scheme'] = metadata_scheme.value if metadata_scheme else None
        embedded_parsed = None
        has_embedded = False
        if parameters is not None and metadata_scheme is not None:
            parser = get_metadata_parser(metadata_scheme)
            embedded_parsed = parser.to_json(parameters) if isinstance(parameters, (dict, str)) else parameters
            if isinstance(embedded_parsed, dict) and len(embedded_parsed) > 0:
                has_embedded = True
                result['raw_metadata'] = embedded_parsed
        elif isinstance(parameters, dict) and len(parameters) > 0:
            embedded_parsed = parameters
            has_embedded = True
            result['raw_metadata'] = embedded_parsed
        log_parsed = None
        has_log = False
        if image_path is not None:
            try:
                import modules.private_logger
                log_data = modules.private_logger.lookup_metadata_from_log(image_path)
                if log_data is not None and isinstance(log_data, dict) and len(log_data) > 0:
                    log_parsed = log_data
                    has_log = True
                    result['log_metadata'] = log_data
            except Exception:
                pass
        if has_embedded and has_log:
            result['metadata_source'] = 'merged'
        elif has_embedded:
            result['metadata_source'] = 'embedded'
        elif has_log:
            result['metadata_source'] = 'log'
        merged = {}
        if has_embedded:
            merged.update(embedded_parsed)
        if has_log:
            for k, v in log_parsed.items():
                if k not in merged or merged.get(k) in (None, '', 'None'):
                    merged[k] = v
        result['merged_metadata'] = merged if merged else None
        try:
            apply_data, apply_src, dropped = build_standard_apply_metadata(
                image_path, embedded_parsed, log_parsed, metadata_scheme
            )
            result['apply_metadata'] = apply_data
            result['apply_sources'] = apply_src
            result['apply_dropped_keys'] = dropped
            source_counts = {'embedded': 0, 'log': 0, 'default': 0}
            for k, s in apply_src.items():
                if s in source_counts:
                    source_counts[s] += 1
            result['apply_source_summary'] = source_counts
        except Exception:
            result['apply_metadata'] = {}
            result['apply_sources'] = {}
            result['apply_dropped_keys'] = []
            result['apply_source_summary'] = {'embedded': 0, 'log': 0, 'default': 0}
        for key, label, ftype in COMPARISON_FIELDS:
            value, ok = safe_extract_field(merged, key, label, ftype)
            if ok:
                field_source = 'embedded'
                if has_embedded and has_log:
                    emb_val, emb_ok = safe_extract_field(embedded_parsed, key, label, ftype)
                    if not emb_ok:
                        field_source = 'log'
                    elif has_log:
                        log_val, log_ok = safe_extract_field(log_parsed, key, label, ftype)
                        if log_ok and emb_val == log_val:
                            field_source = 'both'
                        else:
                            field_source = 'embedded'
                elif has_log and not has_embedded:
                    field_source = 'log'
                result['fields'][key] = {'label': label, 'value': value, 'type': ftype.__name__, 'source': field_source}
        result['loras'] = safe_extract_loras(merged)
    except Exception as e:
        result['error'] = f'Error extracting metadata: {str(e)}'
    return result


def format_field_value(key: str, value) -> str:
    if value is None:
        return '(not set)'
    if key == 'styles' and isinstance(value, list):
        return ', '.join(value) if value else '(none)'
    if key == 'resolution' and isinstance(value, tuple) and len(value) >= 2:
        return f'{value[0]} × {value[1]}'
    if key == 'adm_guidance' and isinstance(value, tuple) and len(value) >= 3:
        return f'P:{value[0]}, N:{value[1]}, E:{value[2]}'
    if key == 'freeu' and isinstance(value, tuple) and len(value) >= 4:
        return f'b1:{value[0]}, b2:{value[1]}, s1:{value[2]}, s2:{value[3]}'
    if isinstance(value, float):
        return f'{value:.4g}'
    return str(value)


def compare_metadata(data_a: dict, data_b: dict) -> dict:
    result = {
        'summary': {
            'total_fields': len(COMPARISON_FIELDS),
            'different': 0,
            'same': 0,
            'missing_a': 0,
            'missing_b': 0,
            'lora_different': False
        },
        'field_comparison': [],
        'lora_comparison': {
            'only_in_a': [],
            'only_in_b': [],
            'both_different_weight': [],
            'both_same': []
        },
        'errors': {
            'a': data_a.get('error'),
            'b': data_b.get('error')
        },
        'source_a': data_a.get('metadata_source', 'none'),
        'source_b': data_b.get('metadata_source', 'none'),
        'apply_source_summary_a': data_a.get('apply_source_summary', {}),
        'apply_source_summary_b': data_b.get('apply_source_summary', {}),
        'apply_dropped_keys_a': data_a.get('apply_dropped_keys', []),
        'apply_dropped_keys_b': data_b.get('apply_dropped_keys', [])
    }
    fields_a = data_a.get('fields', {})
    fields_b = data_b.get('fields', {})
    for key, label, ftype in COMPARISON_FIELDS:
        in_a = key in fields_a
        in_b = key in fields_b
        val_a = fields_a[key]['value'] if in_a else None
        val_b = fields_b[key]['value'] if in_b else None
        src_a = fields_a[key].get('source', 'unknown') if in_a else None
        src_b = fields_b[key].get('source', 'unknown') if in_b else None
        entry = {
            'key': key,
            'label': label,
            'value_a': val_a,
            'value_b': val_b,
            'formatted_a': format_field_value(key, val_a),
            'formatted_b': format_field_value(key, val_b),
            'source_a': src_a,
            'source_b': src_b,
            'in_a': in_a,
            'in_b': in_b,
            'is_different': False,
            'status': 'same'
        }
        if in_a and in_b:
            if val_a != val_b:
                entry['is_different'] = True
                entry['status'] = 'different'
                result['summary']['different'] += 1
            else:
                result['summary']['same'] += 1
        elif in_a and not in_b:
            entry['status'] = 'only_a'
            result['summary']['missing_b'] += 1
            result['summary']['different'] += 1
            entry['is_different'] = True
        elif not in_a and in_b:
            entry['status'] = 'only_b'
            result['summary']['missing_a'] += 1
            result['summary']['different'] += 1
            entry['is_different'] = True
        result['field_comparison'].append(entry)
    loras_a = data_a.get('loras', [])
    loras_b = data_b.get('loras', [])
    names_a = {l['name']: l for l in loras_a}
    names_b = {l['name']: l for l in loras_b}
    all_names = set(names_a.keys()) | set(names_b.keys())
    for name in all_names:
        in_a = name in names_a
        in_b = name in names_b
        if in_a and in_b:
            la, lb = names_a[name], names_b[name]
            if la['weight'] != lb['weight'] or la['enabled'] != lb['enabled']:
                result['lora_comparison']['both_different_weight'].append({
                    'name': name,
                    'a': {'enabled': la['enabled'], 'weight': la['weight']},
                    'b': {'enabled': lb['enabled'], 'weight': lb['weight']}
                })
                result['summary']['lora_different'] = True
            else:
                result['lora_comparison']['both_same'].append({
                    'name': name, 'weight': la['weight'], 'enabled': la['enabled']
                })
        elif in_a:
            la = names_a[name]
            result['lora_comparison']['only_in_a'].append({
                'name': name, 'enabled': la['enabled'], 'weight': la['weight']
            })
            result['summary']['lora_different'] = True
        elif in_b:
            lb = names_b[name]
            result['lora_comparison']['only_in_b'].append({
                'name': name, 'enabled': lb['enabled'], 'weight': lb['weight']
            })
            result['summary']['lora_different'] = True
    return result


def _source_tag(source: str | None) -> str:
    if source is None:
        return ''
    labels = {
        'embedded': ('📷', 'Embedded metadata', '#4d8066'),
        'log': ('📝', 'Private log', '#66664d'),
        'both': ('📷📝', 'Both (verified)', '#4d6680'),
        'merged': ('🔗', 'Merged (embedded + log)', '#665080'),
        'none': ('❌', 'No metadata', '#804040'),
    }
    emoji, title, color = labels.get(source, ('❓', source, '#666'))
    return f'<span style="font-size:10px;color:{color};cursor:help;" title="{title}">{emoji}</span>'


def _source_summary_label(source: str) -> str:
    labels = {
        'embedded': '📷 Embedded only',
        'log': '📝 Log only',
        'merged': '🔗 Merged (embedded + log)',
        'both': '📷📝 Both',
        'none': '❌ None',
    }
    return labels.get(source, source)


def render_comparison_html(comparison: dict, data_a: dict, data_b: dict) -> str:
    summary = comparison['summary']
    errors = comparison['errors']
    source_a = comparison.get('source_a', 'none')
    source_b = comparison.get('source_b', 'none')
    css = """
    <style>
    .cmp-container { font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #e0e0e0; }
    .cmp-summary { background: #1e1e1e; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; }
    .cmp-summary-row { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; align-items: center; }
    .cmp-badge { padding: 3px 10px; border-radius: 12px; font-weight: 600; font-size: 12px; }
    .cmp-badge-diff { background: #5c2d2d; color: #ff9999; }
    .cmp-badge-same { background: #1f4d2e; color: #90ee90; }
    .cmp-badge-miss { background: #4d4020; color: #ffd700; }
    .cmp-badge-src { background: #2a2a3a; color: #b0b0ff; font-size: 11px; }
    .cmp-error { background: #4d1f1f; color: #ffb3b3; padding: 10px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; }
    .cmp-table { width: 100%; border-collapse: collapse; font-size: 13px; background: #1a1a1a; border-radius: 8px; overflow: hidden; }
    .cmp-table th { background: #2a2a2a; padding: 10px 12px; text-align: left; font-weight: 600; color: #bbb; border-bottom: 1px solid #333; }
    .cmp-table td { padding: 8px 12px; border-bottom: 1px solid #252525; vertical-align: top; }
    .cmp-table tr.cmp-diff td { background: #2a1f1f; }
    .cmp-table tr.cmp-only-a td { background: #1f2a2a; }
    .cmp-table tr.cmp-only-b td { background: #2a2a1f; }
    .cmp-field { font-weight: 600; color: #9ecbff; }
    .cmp-val-a { color: #90ee90; }
    .cmp-val-b { color: #87ceeb; }
    .cmp-val-miss { color: #888; font-style: italic; }
    .cmp-lora-section { margin-top: 16px; background: #1a1a1a; border-radius: 8px; padding: 12px; }
    .cmp-lora-title { font-weight: 600; color: #bbb; margin-bottom: 8px; font-size: 14px; }
    .cmp-lora-item { padding: 6px 10px; border-radius: 4px; margin: 4px 0; font-size: 12px; }
    .cmp-lora-a { background: #1f3a2a; color: #90ee90; }
    .cmp-lora-b { background: #1f2e3a; color: #87ceeb; }
    .cmp-lora-diff { background: #3a2a1f; color: #ffd08a; }
    .cmp-lora-same { background: #2a2a2a; color: #aaa; }
    .cmp-images { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
    .cmp-img-box { background: #1a1a1a; border-radius: 8px; padding: 10px; text-align: center; }
    .cmp-img-label { font-size: 12px; color: #888; margin-top: 6px; word-break: break-all; }
    .cmp-header-row { display: grid; grid-template-columns: 18% 41% 41%; }
    .cmp-header { font-weight: 600; padding: 10px 12px; background: #2a2a2a; color: #bbb; border-bottom: 1px solid #333; }
    .cmp-src-indicator { font-size: 10px; margin-left: 4px; opacity: 0.7; }
    .cmp-legend { background: #1e1e1e; padding: 8px 16px; border-radius: 6px; margin-bottom: 12px; font-size: 11px; color: #aaa; }
    .cmp-legend-item { display: inline-block; margin-right: 14px; }
    .cmp-apply-summary { background: #1a1a2a; padding: 10px 16px; border-radius: 6px; margin-bottom: 14px; font-size: 12px; border-left: 3px solid #4a6baf; }
    .cmp-apply-title { font-weight: 600; color: #b0c4ff; margin-bottom: 6px; font-size: 13px; }
    .cmp-apply-row { display: grid; grid-template-columns: 18% 41% 41%; gap: 8px; }
    .cmp-apply-col { padding: 4px 8px; }
    .cmp-apply-badge { display: inline-block; padding: 2px 8px; border-radius: 8px; font-size: 11px; margin-right: 6px; margin-bottom: 3px; }
    .cmp-apply-badge-emb { background: #1f3a2a; color: #90ee90; }
    .cmp-apply-badge-log { background: #3a3520; color: #ffd700; }
    .cmp-apply-badge-def { background: #2a2a2a; color: #999; }
    .cmp-apply-badge-drop { background: #3a1f1f; color: #ff9999; }
    </style>
    """
    parts = [css, '<div class="cmp-container">']
    parts.append('<div class="cmp-legend">')
    parts.append('<span class="cmp-legend-item">📷 = Embedded metadata</span>')
    parts.append('<span class="cmp-legend-item">📝 = Private log</span>')
    parts.append('<span class="cmp-legend-item">📷📝 = Both (verified match)</span>')
    parts.append('<span class="cmp-legend-item">🔗 = Merged (embedded + log)</span>')
    parts.append('</div>')
    if errors.get('a') or errors.get('b'):
        if errors.get('a'):
            parts.append(f'<div class="cmp-error">Image A error: {errors["a"]}</div>')
        if errors.get('b'):
            parts.append(f'<div class="cmp-error">Image B error: {errors["b"]}</div>')
    parts.append('<div class="cmp-summary">')
    parts.append('<div class="cmp-summary-row">')
    parts.append(f'<span class="cmp-badge cmp-badge-diff">Different: {summary["different"]}</span>')
    parts.append(f'<span class="cmp-badge cmp-badge-same">Same: {summary["same"]}</span>')
    parts.append(f'<span class="cmp-badge cmp-badge-miss">Missing in A: {summary["missing_a"]}</span>')
    parts.append(f'<span class="cmp-badge cmp-badge-miss">Missing in B: {summary["missing_b"]}</span>')
    if summary['lora_different']:
        parts.append(f'<span class="cmp-badge cmp-badge-diff">LoRA differ</span>')
    parts.append(f'<span class="cmp-badge cmp-badge-src">A: {_source_summary_label(source_a)}</span>')
    parts.append(f'<span class="cmp-badge cmp-badge-src">B: {_source_summary_label(source_b)}</span>')
    parts.append('</div></div>')
    apply_sum_a = comparison.get('apply_source_summary_a', {})
    apply_sum_b = comparison.get('apply_source_summary_b', {})
    dropped_a = comparison.get('apply_dropped_keys_a', [])
    dropped_b = comparison.get('apply_dropped_keys_b', [])
    parts.append('<div class="cmp-apply-summary">')
    parts.append('<div class="cmp-apply-title">📋 Apply parameter sources (what will actually be applied)</div>')
    parts.append('<div class="cmp-apply-row">')
    parts.append('<div class="cmp-apply-col"><b>Image A</b></div>')
    parts.append('<div class="cmp-apply-col">')
    parts.append(f'<span class="cmp-apply-badge cmp-apply-badge-emb">📷 Embedded: {apply_sum_a.get("embedded", 0)}</span>')
    parts.append(f'<span class="cmp-apply-badge cmp-apply-badge-log">📝 Log: {apply_sum_a.get("log", 0)}</span>')
    parts.append(f'<span class="cmp-apply-badge cmp-apply-badge-def">Default: {apply_sum_a.get("default", 0)}</span>')
    if dropped_a:
        parts.append(f'<br><span class="cmp-apply-badge cmp-apply-badge-drop" title="Unknown keys not used for apply">Dropped keys: {len(dropped_a)}</span>')
        parts.append(f'<span style="font-size:11px;color:#ff9999;opacity:0.7;">{", ".join(dropped_a[:8])}</span>')
    parts.append('</div>')
    parts.append('<div class="cmp-apply-col">')
    parts.append(f'<span class="cmp-apply-badge cmp-apply-badge-emb">📷 Embedded: {apply_sum_b.get("embedded", 0)}</span>')
    parts.append(f'<span class="cmp-apply-badge cmp-apply-badge-log">📝 Log: {apply_sum_b.get("log", 0)}</span>')
    parts.append(f'<span class="cmp-apply-badge cmp-apply-badge-def">Default: {apply_sum_b.get("default", 0)}</span>')
    if dropped_b:
        parts.append(f'<br><span class="cmp-apply-badge cmp-apply-badge-drop" title="Unknown keys not used for apply">Dropped keys: {len(dropped_b)}</span>')
        parts.append(f'<span style="font-size:11px;color:#ff9999;opacity:0.7;">{", ".join(dropped_b[:8])}</span>')
    parts.append('</div>')
    parts.append('</div></div>')
    parts.append('<table class="cmp-table">')
    parts.append('<thead><tr class="cmp-header-row">')
    parts.append('<th class="cmp-header">Parameter</th>')
    parts.append('<th class="cmp-header">Image A</th>')
    parts.append('<th class="cmp-header">Image B</th>')
    parts.append('</tr></thead><tbody>')
    for entry in comparison['field_comparison']:
        row_class = ''
        if entry['status'] == 'different':
            row_class = ' class="cmp-diff"'
        elif entry['status'] == 'only_a':
            row_class = ' class="cmp-only-a"'
        elif entry['status'] == 'only_b':
            row_class = ' class="cmp-only-b"'
        src_tag_a = _source_tag(entry.get('source_a'))
        src_tag_b = _source_tag(entry.get('source_b'))
        va = f'<span class="cmp-val-a">{entry["formatted_a"]}</span> {src_tag_a}' if entry['in_a'] else '<span class="cmp-val-miss">(not set)</span>'
        vb = f'<span class="cmp-val-b">{entry["formatted_b"]}</span> {src_tag_b}' if entry['in_b'] else '<span class="cmp-val-miss">(not set)</span>'
        parts.append(f'<tr{row_class}><td class="cmp-field">{entry["label"]}</td><td>{va}</td><td>{vb}</td></tr>')
    parts.append('</tbody></table>')
    lc = comparison['lora_comparison']
    has_lora = any(v for v in lc.values())
    if has_lora:
        parts.append('<div class="cmp-lora-section">')
        parts.append(f'<div class="cmp-lora-title">LoRA Comparison</div>')
        if lc['only_in_a']:
            for item in lc['only_in_a']:
                w = f"{item['weight']:.3g}"
                on = 'ON' if item['enabled'] else 'off'
                parts.append(f'<div class="cmp-lora-item cmp-lora-a">🅰 Only in A: <b>{item["name"]}</b> (weight={w}, {on})</div>')
        if lc['only_in_b']:
            for item in lc['only_in_b']:
                w = f"{item['weight']:.3g}"
                on = 'ON' if item['enabled'] else 'off'
                parts.append(f'<div class="cmp-lora-item cmp-lora-b">🅱 Only in B: <b>{item["name"]}</b> (weight={w}, {on})</div>')
        if lc['both_different_weight']:
            for item in lc['both_different_weight']:
                wa = f"{item['a']['weight']:.3g}"
                wb = f"{item['b']['weight']:.3g}"
                oa = 'ON' if item['a']['enabled'] else 'off'
                ob = 'ON' if item['b']['enabled'] else 'off'
                parts.append(f'<div class="cmp-lora-item cmp-lora-diff">⚡ <b>{item["name"]}</b> — A: weight={wa} ({oa}) | B: weight={wb} ({ob})</div>')
        if lc['both_same']:
            for item in lc['both_same']:
                w = f"{item['weight']:.3g}"
                on = 'ON' if item['enabled'] else 'off'
                parts.append(f'<div class="cmp-lora-item cmp-lora-same">✓ Same: <b>{item["name"]}</b> (weight={w}, {on})</div>')
        parts.append('</div>')
    parts.append('</div>')
    return ''.join(parts)
