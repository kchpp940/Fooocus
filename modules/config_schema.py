import os
import json
import numbers
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ast import literal_eval

from modules.flags import (
    Performance, OutputFormat, MetadataScheme,
    sampler_list, scheduler_list, sdxl_aspect_ratios,
    inpaint_engine_versions, inpaint_options, inpaint_mask_models,
    inpaint_mask_cloth_category, inpaint_mask_sam_model,
    ip_list, default_ip, default_parameters, uov_list,
    input_image_tab_ids, default_input_image_tab, default_vae,
    enhancement_uov_processing_order, enhancement_uov_before,
    enhancement_uov_prompt_types, enhancement_uov_prompt_type_original,
    describe_types, describe_type_photo, clip_skip_max, disabled
)
from modules.extra_utils import try_eval_env_var, makedirs_with_log


class ConfigSource(Enum):
    DEFAULT = "default"
    PRESET = "preset"
    CONFIG_FILE = "config_file"
    DEPRECATED_CONFIG = "deprecated_config"
    ENVIRONMENT = "environment"
    CLI_ARG = "cli_arg"
    UI_CONTROL = "ui_control"


class ConfigIssueType(Enum):
    UNKNOWN_KEY = "unknown_key"
    TYPE_ERROR = "type_error"
    VALIDATION_ERROR = "validation_error"
    PATH_NOT_FOUND = "path_not_found"
    DEPRECATED = "deprecated"
    MISSING_REQUIRED = "missing_required"


@dataclass
class ConfigIssue:
    key: str
    issue_type: ConfigIssueType
    message: str
    source: Optional[ConfigSource] = None
    value: Any = None
    suggestion: Optional[str] = None


@dataclass
class ConfigItem:
    key: str
    default_value: Any
    value_type: type
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None
    converter: Optional[Callable[[Any], Any]] = None
    is_path: bool = False
    is_path_array: bool = False
    make_directory: bool = False
    deprecated: bool = False
    deprecated_reason: str = ""
    replaced_by: Optional[str] = None
    ui_default: bool = True
    save_to_config: bool = True
    category: str = "general"


@dataclass
class ConfigValue:
    value: Any
    source: ConfigSource
    source_detail: str = ""


@dataclass
class ConfigSchema:
    items: Dict[str, ConfigItem] = field(default_factory=dict)
    deprecated_aliases: Dict[str, str] = field(default_factory=dict)

    def register(self, item: ConfigItem) -> None:
        self.items[item.key] = item

    def get(self, key: str) -> Optional[ConfigItem]:
        return self.items.get(key)

    def has(self, key: str) -> bool:
        return key in self.items

    def all_keys(self) -> List[str]:
        return list(self.items.keys())

    def add_deprecated_alias(self, old_key: str, new_key: str) -> None:
        self.deprecated_aliases[old_key] = new_key

    def resolve_alias(self, key: str) -> Optional[str]:
        return self.deprecated_aliases.get(key)


def create_default_schema() -> ConfigSchema:
    schema = ConfigSchema()

    schema.add_deprecated_alias('modelfile_path', 'path_checkpoints')
    schema.add_deprecated_alias('lorafile_path', 'path_loras')
    schema.add_deprecated_alias('embeddings_path', 'path_embeddings')
    schema.add_deprecated_alias('vae_approx_path', 'path_vae_approx')
    schema.add_deprecated_alias('upscale_models_path', 'path_upscale_models')
    schema.add_deprecated_alias('inpaint_models_path', 'path_inpaint')
    schema.add_deprecated_alias('controlnet_models_path', 'path_controlnet')
    schema.add_deprecated_alias('clip_vision_models_path', 'path_clip_vision')
    schema.add_deprecated_alias('fooocus_expansion_path', 'path_fooocus_expansion')
    schema.add_deprecated_alias('temp_outputs_path', 'path_outputs')

    path_items = [
        ('path_checkpoints', ['../models/checkpoints/'], True, 'Checkpoint models directory'),
        ('path_loras', ['../models/loras/'], True, 'LoRA models directory'),
        ('path_embeddings', '../models/embeddings/', False, 'Embeddings directory'),
        ('path_vae_approx', '../models/vae_approx/', False, 'VAE approx directory'),
        ('path_vae', '../models/vae/', False, 'VAE models directory'),
        ('path_upscale_models', '../models/upscale_models/', False, 'Upscale models directory'),
        ('path_inpaint', '../models/inpaint/', False, 'Inpaint models directory'),
        ('path_controlnet', '../models/controlnet/', False, 'ControlNet models directory'),
        ('path_clip_vision', '../models/clip_vision/', False, 'CLIP vision models directory'),
        ('path_fooocus_expansion', '../models/prompt_expansion/fooocus_expansion', False, 'Fooocus expansion directory'),
        ('path_wildcards', '../wildcards/', False, 'Wildcards directory'),
        ('path_safety_checker', '../models/safety_checker/', False, 'Safety checker directory'),
        ('path_sam', '../models/sam/', False, 'SAM models directory'),
        ('path_outputs', '../outputs/', False, 'Outputs directory'),
    ]

    for key, default, is_array, desc in path_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=list if is_array else str,
            description=desc,
            is_path=True,
            is_path_array=is_array,
            make_directory=True,
            category='paths'
        ))

    model_items = [
        ('default_model', 'model.safetensors', str, 'Default base model',
         lambda x: isinstance(x, str)),
        ('default_refiner', 'None', str, 'Default refiner model',
         lambda x: isinstance(x, str)),
        ('default_refiner_switch', 0.8, float, 'Refiner switch value',
         lambda x: isinstance(x, numbers.Number) and 0 <= x <= 1),
        ('previous_default_models', [], list, 'List of previous default models',
         lambda x: isinstance(x, list) and all(isinstance(k, str) for k in x)),
        ('default_vae', default_vae, str, 'Default VAE',
         lambda x: isinstance(x, str)),
    ]

    for key, default, vtype, desc, validator in model_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=vtype,
            description=desc,
            validator=validator,
            category='models'
        ))

    lora_items = [
        ('default_loras_min_weight', -2, float, 'Minimum LoRA weight',
         lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10),
        ('default_loras_max_weight', 2, float, 'Maximum LoRA weight',
         lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10),
        ('default_max_lora_number', 5, int, 'Maximum number of LoRA slots',
         lambda x: isinstance(x, int) and x >= 1),
    ]

    for key, default, vtype, desc, validator in lora_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=vtype,
            description=desc,
            validator=validator,
            category='loras'
        ))

    default_loras = [
        [True, "None", 1.0],
        [True, "None", 1.0],
        [True, "None", 1.0],
        [True, "None", 1.0],
        [True, "None", 1.0]
    ]

    schema.register(ConfigItem(
        key='default_loras',
        default_value=default_loras,
        value_type=list,
        description='Default LoRA configurations',
        validator=lambda x: isinstance(x, list) and all(
            len(y) == 3 and isinstance(y[0], bool) and isinstance(y[1], str) and isinstance(y[2], numbers.Number)
            or len(y) == 2 and isinstance(y[0], str) and isinstance(y[1], numbers.Number)
            for y in x),
        converter=convert_loras,
        category='loras'
    ))

    sampling_items = [
        ('default_cfg_scale', 7.0, float, 'CFG scale',
         lambda x: isinstance(x, numbers.Number)),
        ('default_sample_sharpness', 2.0, float, 'Sample sharpness',
         lambda x: isinstance(x, numbers.Number)),
        ('default_cfg_tsnr', 7.0, float, 'Adaptive CFG (TSNR)',
         lambda x: isinstance(x, numbers.Number)),
        ('default_clip_skip', 2, int, 'CLIP skip layers',
         lambda x: isinstance(x, int) and 1 <= x <= clip_skip_max),
        ('default_sampler', 'dpmpp_2m_sde_gpu', str, 'Default sampler',
         lambda x: x in sampler_list),
        ('default_scheduler', 'karras', str, 'Default scheduler',
         lambda x: x in scheduler_list),
        ('default_overwrite_step', -1, int, 'Overwrite steps (-1 for auto)',
         lambda x: isinstance(x, int)),
        ('default_overwrite_switch', -1, int, 'Overwrite switch (-1 for auto)',
         lambda x: isinstance(x, int)),
        ('default_overwrite_upscale', -1, float, 'Overwrite upscale (-1 for auto)',
         lambda x: isinstance(x, numbers.Number)),
    ]

    for key, default, vtype, desc, validator in sampling_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=vtype,
            description=desc,
            validator=validator,
            category='sampling'
        ))

    style_items = [
        ('default_styles', ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"], list, 'Default styles',
         None),
        ('default_prompt', '', str, 'Default prompt',
         lambda x: isinstance(x, str)),
        ('default_prompt_negative', '', str, 'Default negative prompt',
         lambda x: isinstance(x, str)),
    ]

    for key, default, vtype, desc, validator in style_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=vtype,
            description=desc,
            validator=validator,
            category='prompts'
        ))

    performance_items = [
        ('default_performance', Performance.SPEED.value, str, 'Default performance mode',
         lambda x: x in Performance.values()),
        ('default_aspect_ratio', '1152*896', str, 'Default aspect ratio',
         lambda x: x in sdxl_aspect_ratios),
        ('available_aspect_ratios', sdxl_aspect_ratios, list, 'Available aspect ratios',
         lambda x: isinstance(x, list) and all('*' in v for v in x) and len(x) > 1),
    ]

    for key, default, vtype, desc, validator in performance_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=vtype,
            description=desc,
            validator=validator,
            category='performance'
        ))

    ui_items = [
        ('default_image_prompt_checkbox', False, bool, 'Image prompt checkbox default',
         lambda x: isinstance(x, bool)),
        ('default_enhance_checkbox', False, bool, 'Enhance checkbox default',
         lambda x: isinstance(x, bool)),
        ('default_advanced_checkbox', False, bool, 'Advanced checkbox default',
         lambda x: isinstance(x, bool)),
        ('default_developer_debug_mode_checkbox', False, bool, 'Developer debug mode default',
         lambda x: isinstance(x, bool)),
        ('default_image_prompt_advanced_checkbox', False, bool, 'Image prompt advanced checkbox default',
         lambda x: isinstance(x, bool)),
        ('default_max_image_number', 32, int, 'Maximum image number',
         lambda x: isinstance(x, int) and x >= 1),
        ('default_image_number', 2, int, 'Default image number',
         lambda x: isinstance(x, int) and 1 <= x <= 32),
        ('default_output_format', 'png', str, 'Default output format',
         lambda x: x in OutputFormat.list()),
    ]

    for key, default, vtype, desc, validator in ui_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=vtype,
            description=desc,
            validator=validator,
            category='ui'
        ))

    inpaint_items = [
        ('default_inpaint_engine_version', 'v2.6', str, 'Inpaint engine version',
         lambda x: x in inpaint_engine_versions),
        ('default_inpaint_advanced_masking_checkbox', False, bool, 'Advanced masking checkbox default',
         lambda x: isinstance(x, bool)),
        ('default_inpaint_method', inpaint_options[0], str, 'Inpaint method',
         lambda x: x in inpaint_options),
        ('default_invert_mask_checkbox', False, bool, 'Invert mask checkbox default',
         lambda x: isinstance(x, bool)),
        ('default_inpaint_mask_model', 'isnet-general-use', str, 'Inpaint mask model',
         lambda x: x in inpaint_mask_models),
        ('default_enhance_inpaint_mask_model', 'sam', str, 'Enhance inpaint mask model',
         lambda x: x in inpaint_mask_models),
        ('default_inpaint_mask_cloth_category', 'full', str, 'Inpaint mask cloth category',
         lambda x: x in inpaint_mask_cloth_category),
        ('default_inpaint_mask_sam_model', 'vit_b', str, 'Inpaint SAM model',
         lambda x: x in inpaint_mask_sam_model),
        ('default_sam_max_detections', 0, int, 'SAM max detections (0 for all)',
         lambda x: isinstance(x, int) and 0 <= x <= 10),
    ]

    for key, default, vtype, desc, validator in inpaint_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=vtype,
            description=desc,
            validator=validator,
            category='inpaint'
        ))

    ip_items = [
        ('default_selected_image_input_tab_id', default_input_image_tab, str, 'Default image input tab',
         lambda x: x in input_image_tab_ids),
        ('default_uov_method', disabled, str, 'Default UOV method',
         lambda x: x in uov_list),
        ('default_controlnet_image_count', 4, int, 'Number of controlnet images',
         lambda x: isinstance(x, int) and x > 0),
    ]

    for key, default, vtype, desc, validator in ip_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=vtype,
            description=desc,
            validator=validator,
            category='image_prompt'
        ))

    for i in range(1, 5):
        schema.register(ConfigItem(
            key=f'default_ip_image_{i}',
            default_value='None',
            value_type=str,
            description=f'Default IP image {i}',
            validator=lambda x: x == 'None' or (isinstance(x, str) and os.path.exists(x)),
            category='image_prompt',
            ui_default=False
        ))
        schema.register(ConfigItem(
            key=f'default_ip_type_{i}',
            default_value=default_ip,
            value_type=str,
            description=f'Default IP type {i}',
            validator=lambda x: x in ip_list,
            category='image_prompt',
            ui_default=False
        ))
        schema.register(ConfigItem(
            key=f'default_ip_stop_at_{i}',
            default_value=default_parameters[default_ip][0],
            value_type=float,
            description=f'Default IP stop at {i}',
            validator=lambda x: isinstance(x, float) and 0 <= x <= 1,
            category='image_prompt',
            ui_default=False
        ))
        schema.register(ConfigItem(
            key=f'default_ip_weight_{i}',
            default_value=default_parameters[default_ip][1],
            value_type=float,
            description=f'Default IP weight {i}',
            validator=lambda x: isinstance(x, float) and 0 <= x <= 2,
            category='image_prompt',
            ui_default=False
        ))

    enhance_items = [
        ('example_inpaint_prompts',
         ['highly detailed face', 'detailed girl face', 'detailed man face', 'detailed hand', 'beautiful eyes'],
         list, 'Example inpaint prompts',
         lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x)),
        ('example_enhance_detection_prompts',
         ['face', 'eye', 'mouth', 'hair', 'hand', 'body'],
         list, 'Example enhance detection prompts',
         lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x)),
        ('default_enhance_tabs', 3, int, 'Number of enhance tabs',
         lambda x: isinstance(x, int) and 1 <= x <= 5),
        ('default_enhance_uov_method', disabled, str, 'Default enhance UOV method',
         lambda x: x in uov_list),
        ('default_enhance_uov_processing_order', enhancement_uov_before, str, 'Enhance UOV processing order',
         lambda x: x in enhancement_uov_processing_order),
        ('default_enhance_uov_prompt_type', enhancement_uov_prompt_type_original, str, 'Enhance UOV prompt type',
         lambda x: x in enhancement_uov_prompt_types),
    ]

    for key, default, vtype, desc, validator in enhance_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=vtype,
            description=desc,
            validator=validator,
            category='enhance'
        ))

    describe_items = [
        ('default_describe_apply_prompts_checkbox', True, bool, 'Apply prompts checkbox for describe',
         lambda x: isinstance(x, bool)),
        ('default_describe_content_type', [describe_type_photo], list, 'Describe content type',
         lambda x: all(k in describe_types for k in x)),
    ]

    for key, default, vtype, desc, validator in describe_items:
        schema.register(ConfigItem(
            key=key,
            default_value=default,
            value_type=