from dataclasses import dataclass
from typing import Callable, List, Optional
import sys

import args_manager
from modules.ui.types import (
    TopCheckboxes, PromptAndButtons,
    ImageInputTabs, EnhancePanelComponents,
    AdvancedColumnComponents
)


@dataclass
class ProtocolSlot:
    name: str
    source: str
    description: str
    spread: bool = False
    condition: Optional[Callable[[], bool]] = None


@dataclass
class SlotResolution:
    name: str
    source: str
    description: str
    spread: bool
    count: int
    value: object


@dataclass
class ProtocolValidationResult:
    declaration_name: str
    slots: List[SlotResolution]
    total_length: int
    errors: List[str]


def _resolve_path(context: dict, path: str, slot_name: str) -> object:
    parts = path.split('.')
    try:
        obj = context[parts[0]]
    except KeyError:
        raise KeyError(
            f"Protocol slot '{slot_name}': context key '{parts[0]}' not found. "
            f"Available keys: {list(context.keys())}"
        )
    for i, part in enumerate(parts[1:], 1):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise AttributeError(
                f"Protocol slot '{slot_name}': path '{path}' failed at position {i} "
                f"('{part}'). Object type '{type(obj).__name__}' at '{'.'.join(parts[:i])}' "
                f"has no attribute '{part}'. Available: {[a for a in dir(obj) if not a.startswith('_')]}"
            )
    return obj


def _resolve_and_report(declaration: List[ProtocolSlot], context: dict,
                         declaration_name: str) -> ProtocolValidationResult:
    slots = []
    errors = []

    for slot in declaration:
        if slot.condition is not None:
            try:
                cond_result = slot.condition()
            except Exception as e:
                errors.append(
                    f"Slot '{slot.name}': condition raised {type(e).__name__}: {e}"
                )
                continue
            if not cond_result:
                slots.append(SlotResolution(
                    name=slot.name,
                    source=slot.source,
                    description=slot.description + " (EXCLUDED by condition)",
                    spread=slot.spread,
                    count=0,
                    value=None
                ))
                continue

        try:
            value = _resolve_path(context, slot.source, slot.name)
        except (KeyError, AttributeError) as e:
            errors.append(str(e))
            continue

        count = 1
        if slot.spread:
            try:
                items = list(value)
                count = len(items)
                value = items
            except TypeError:
                errors.append(
                    f"Slot '{slot.name}': marked spread=True but value "
                    f"of type '{type(value).__name__}' is not iterable."
                )
                continue

        slots.append(SlotResolution(
            name=slot.name,
            source=slot.source,
            description=slot.description,
            spread=slot.spread,
            count=count,
            value=value
        ))

    total_length = sum(s.count for s in slots)
    return ProtocolValidationResult(
        declaration_name=declaration_name,
        slots=slots,
        total_length=total_length,
        errors=errors
    )


def _format_validation_report(result: ProtocolValidationResult,
                               expected_length: Optional[int]) -> str:
    lines = []
    lines.append(f"\n=== Protocol Validation: {result.declaration_name} ===")
    lines.append(f"{'Slot':<38} {'Count':>6}  Description")
    lines.append("-" * 90)
    for s in result.slots:
        count_str = f"{s.count}{'*' if s.spread else ''}"
        name_str = s.name + (" (skip)" if s.count == 0 else "")
        lines.append(f"  {name_str:<36} {count_str:>6}  {s.description}")
    lines.append("-" * 90)
    lines.append(f"Total slots: {len(result.slots)}  |  Total components: {result.total_length}")
    if expected_length is not None:
        status = "OK" if result.total_length == expected_length else "MISMATCH"
        lines.append(f"Expected length: {expected_length}  |  {status}")
    if result.errors:
        lines.append("\nERRORS:")
        for err in result.errors:
            lines.append(f"  * {err}")
    lines.append("")
    return "\n".join(lines)


def resolve_slots(declaration: List[ProtocolSlot], context: dict) -> list:
    result = []
    for slot in declaration:
        if slot.condition is not None and not slot.condition():
            continue
        value = _resolve_path(context, slot.source, slot.name)
        if slot.spread:
            result.extend(value)
        else:
            result.append(value)
    return result


def validate_and_build(
    declaration: List[ProtocolSlot],
    context: dict,
    declaration_name: str,
    expected_length: Optional[int] = None,
) -> tuple:
    report = _resolve_and_report(declaration, context, declaration_name)
    sys.stdout.write(_format_validation_report(report, expected_length))
    sys.stdout.flush()

    if report.errors:
        raise ValueError(
            f"Protocol validation failed for '{declaration_name}' "
            f"with {len(report.errors)} error(s). See output above."
        )

    if expected_length is not None and report.total_length != expected_length:
        raise ValueError(
            f"Protocol '{declaration_name}' length mismatch: "
            f"expected {expected_length}, got {report.total_length}. "
            f"Did a dataclass field change or a spread list length change?"
        )

    result = []
    for s in report.slots:
        if s.count == 0:
            continue
        if s.spread:
            result.extend(s.value)
        else:
            result.append(s.value)
    return result, report


def slot_index(declaration: List[ProtocolSlot], name: str) -> int:
    idx = 0
    for slot in declaration:
        if slot.condition is not None and not slot.condition():
            continue
        if slot.spread:
            raise ValueError(
                f"Cannot compute stable index for spread slot '{slot.name}'; "
                f"use a non-spread slot or resolve the list first."
            )
        if slot.name == name:
            return idx
        idx += 1
    raise KeyError(f"Slot '{name}' not found in declaration (or was conditionally excluded)")


_enable_image_log = lambda: not args_manager.args.disable_image_log
_enable_metadata = lambda: not args_manager.args.disable_metadata


LOAD_DATA_OUTPUTS_DECLARATION = [
    ProtocolSlot("advanced_checkbox", "top_checkboxes.advanced_checkbox", "Toggle advanced settings column"),
    ProtocolSlot("image_number", "advanced.settings.image_number", "Number of images to generate"),
    ProtocolSlot("prompt", "prompt_buttons.prompt", "Positive prompt text"),
    ProtocolSlot("negative_prompt", "advanced.settings.negative_prompt", "Negative prompt text"),
    ProtocolSlot("style_selections", "advanced.styles.style_selections", "Selected style names"),
    ProtocolSlot("performance_selection", "advanced.settings.performance_selection", "Speed/Quality/Extreme Speed"),
    ProtocolSlot("overwrite_step", "advanced.advanced.debug_tools.overwrite_step", "Override sampling steps"),
    ProtocolSlot("overwrite_switch", "advanced.advanced.debug_tools.overwrite_switch", "Override refiner switch step"),
    ProtocolSlot("aspect_ratios_selection", "advanced.settings.aspect_ratios_selection", "Image aspect ratio"),
    ProtocolSlot("overwrite_width", "advanced.advanced.debug_tools.overwrite_width", "Override image width"),
    ProtocolSlot("overwrite_height", "advanced.advanced.debug_tools.overwrite_height", "Override image height"),
    ProtocolSlot("guidance_scale", "advanced.advanced.guidance_scale", "CFG / guidance scale"),
    ProtocolSlot("sharpness", "advanced.advanced.sharpness", "Image sharpness"),
    ProtocolSlot("adm_scaler_positive", "advanced.advanced.debug_tools.adm_scaler_positive", "ADM positive scaler"),
    ProtocolSlot("adm_scaler_negative", "advanced.advanced.debug_tools.adm_scaler_negative", "ADM negative scaler"),
    ProtocolSlot("adm_scaler_end", "advanced.advanced.debug_tools.adm_scaler_end", "ADM scaler end value"),
    ProtocolSlot("refiner_swap_method", "advanced.advanced.debug_tools.refiner_swap_method", "Refiner swap method"),
    ProtocolSlot("adaptive_cfg", "advanced.advanced.debug_tools.adaptive_cfg", "Adaptive CFG"),
    ProtocolSlot("clip_skip", "advanced.advanced.debug_tools.clip_skip", "CLIP skip count"),
    ProtocolSlot("base_model", "advanced.models.base_model", "Base SDXL model name"),
    ProtocolSlot("refiner_model", "advanced.models.refiner_model", "Refiner model name"),
    ProtocolSlot("refiner_switch", "advanced.models.refiner_switch", "Refiner switch slider"),
    ProtocolSlot("sampler_name", "advanced.advanced.debug_tools.sampler_name", "Sampler name"),
    ProtocolSlot("scheduler_name", "advanced.advanced.debug_tools.scheduler_name", "Scheduler name"),
    ProtocolSlot("vae_name", "advanced.advanced.debug_tools.vae_name", "VAE name"),
    ProtocolSlot("seed_random", "advanced.settings.seed_random", "Random seed checkbox"),
    ProtocolSlot("image_seed", "advanced.settings.image_seed", "Seed value"),
    ProtocolSlot("inpaint_engine", "advanced.advanced.inpaint_advanced.inpaint_engine", "Inpaint engine version"),
    ProtocolSlot("inpaint_engine_state", "inpaint_engine_state", "Inpaint engine state tracker"),
    ProtocolSlot("inpaint_mode", "image_input.inpaint.inpaint_mode", "Inpaint/outpaint method"),
    ProtocolSlot("enhance_inpaint_modes", "enhance_panel.enhance_inpaint_mode_ctrls", "Per-tab enhance inpaint modes", spread=True),
    ProtocolSlot("generate_button", "prompt_buttons.generate_button", "Generate button (visibility toggle)"),
    ProtocolSlot("load_parameter_button", "prompt_buttons.load_parameter_button", "Load Parameters button (visibility toggle)"),
    ProtocolSlot("freeu_ctrls", "advanced.advanced.freeu.freeu_ctrls", "FreeU enable/b1/b2/s1/s2", spread=True),
    ProtocolSlot("lora_ctrls", "advanced.models.lora_ctrls", "LoRA enabled/model/weight per slot", spread=True),
]

LOAD_DATA_OUTPUTS_EXPECTED_LENGTH = 57


CTRLS_DECLARATION = [
    ProtocolSlot("currentTask", "currentTask", "Async task state object"),
    ProtocolSlot("generate_image_grid", "advanced.advanced.debug_tools.generate_image_grid", "Image grid generation toggle"),

    ProtocolSlot("prompt", "prompt_buttons.prompt", "Positive prompt text"),
    ProtocolSlot("negative_prompt", "advanced.settings.negative_prompt", "Negative prompt text"),
    ProtocolSlot("style_selections", "advanced.styles.style_selections", "Selected style names"),
    ProtocolSlot("performance_selection", "advanced.settings.performance_selection", "Speed/Quality/Extreme Speed"),
    ProtocolSlot("aspect_ratios_selection", "advanced.settings.aspect_ratios_selection", "Image aspect ratio"),
    ProtocolSlot("image_number", "advanced.settings.image_number", "Number of images to generate"),
    ProtocolSlot("output_format", "advanced.settings.output_format", "Output image format"),
    ProtocolSlot("image_seed", "advanced.settings.image_seed", "Seed value"),
    ProtocolSlot("read_wildcards_in_order", "advanced.advanced.debug_tools.read_wildcards_in_order", "Read wildcards in order"),
    ProtocolSlot("sharpness", "advanced.advanced.sharpness", "Image sharpness"),
    ProtocolSlot("guidance_scale", "advanced.advanced.guidance_scale", "CFG / guidance scale"),

    ProtocolSlot("base_model", "advanced.models.base_model", "Base SDXL model name"),
    ProtocolSlot("refiner_model", "advanced.models.refiner_model", "Refiner model name"),
    ProtocolSlot("refiner_switch", "advanced.models.refiner_switch", "Refiner switch slider"),
    ProtocolSlot("lora_ctrls", "advanced.models.lora_ctrls", "LoRA enabled/model/weight per slot", spread=True),

    ProtocolSlot("input_image_checkbox", "top_checkboxes.input_image_checkbox", "Image input panel toggle"),
    ProtocolSlot("current_tab", "current_tab", "Active image input tab identifier"),

    ProtocolSlot("uov_method", "image_input.uov_method", "Upscale or variation method"),
    ProtocolSlot("uov_input_image", "image_input.uov_input_image", "Upscale/variation source image"),

    ProtocolSlot("outpaint_selections", "image_input.inpaint.outpaint_selections", "Outpaint direction checkboxes"),
    ProtocolSlot("inpaint_input_image", "image_input.inpaint.inpaint_input_image", "Inpaint source image"),
    ProtocolSlot("inpaint_additional_prompt", "image_input.inpaint.inpaint_additional_prompt", "Inpaint extra prompt"),
    ProtocolSlot("inpaint_mask_image", "image_input.inpaint.inpaint_mask_image", "Inpaint mask upload"),

    ProtocolSlot("disable_preview", "advanced.advanced.debug_tools.disable_preview", "Disable live preview"),
    ProtocolSlot("disable_intermediate_results", "advanced.advanced.debug_tools.disable_intermediate_results", "Disable intermediate results"),
    ProtocolSlot("disable_seed_increment", "advanced.advanced.debug_tools.disable_seed_increment", "Disable seed auto-increment"),
    ProtocolSlot("black_out_nsfw", "advanced.advanced.debug_tools.black_out_nsfw", "Black out NSFW content"),

    ProtocolSlot("adm_scaler_positive", "advanced.advanced.debug_tools.adm_scaler_positive", "ADM positive scaler"),
    ProtocolSlot("adm_scaler_negative", "advanced.advanced.debug_tools.adm_scaler_negative", "ADM negative scaler"),
    ProtocolSlot("adm_scaler_end", "advanced.advanced.debug_tools.adm_scaler_end", "ADM scaler end value"),
    ProtocolSlot("adaptive_cfg", "advanced.advanced.debug_tools.adaptive_cfg", "Adaptive CFG"),
    ProtocolSlot("clip_skip", "advanced.advanced.debug_tools.clip_skip", "CLIP skip count"),

    ProtocolSlot("sampler_name", "advanced.advanced.debug_tools.sampler_name", "Sampler name"),
    ProtocolSlot("scheduler_name", "advanced.advanced.debug_tools.scheduler_name", "Scheduler name"),
    ProtocolSlot("vae_name", "advanced.advanced.debug_tools.vae_name", "VAE name"),

    ProtocolSlot("overwrite_step", "advanced.advanced.debug_tools.overwrite_step", "Override sampling steps"),
    ProtocolSlot("overwrite_switch", "advanced.advanced.debug_tools.overwrite_switch", "Override refiner switch step"),
    ProtocolSlot("overwrite_width", "advanced.advanced.debug_tools.overwrite_width", "Override image width"),
    ProtocolSlot("overwrite_height", "advanced.advanced.debug_tools.overwrite_height", "Override image height"),
    ProtocolSlot("overwrite_vary_strength", "advanced.advanced.debug_tools.overwrite_vary_strength", "Vary upscale strength override"),

    ProtocolSlot("overwrite_upscale_strength", "advanced.advanced.debug_tools.overwrite_upscale_strength", "Upscale strength override"),
    ProtocolSlot("mixing_image_prompt_and_vary_upscale", "advanced.advanced.control.mixing_image_prompt_and_vary_upscale", "Mix IP + vary upscale"),
    ProtocolSlot("mixing_image_prompt_and_inpaint", "advanced.advanced.control.mixing_image_prompt_and_inpaint", "Mix IP + inpaint"),

    ProtocolSlot("debugging_cn_preprocessor", "advanced.advanced.control.debugging_cn_preprocessor", "Debug ControlNet preprocessor"),
    ProtocolSlot("skipping_cn_preprocessor", "advanced.advanced.control.skipping_cn_preprocessor", "Skip ControlNet preprocessor"),
    ProtocolSlot("canny_low_threshold", "advanced.advanced.control.canny_low_threshold", "Canny edge low threshold"),
    ProtocolSlot("canny_high_threshold", "advanced.advanced.control.canny_high_threshold", "Canny edge high threshold"),

    ProtocolSlot("refiner_swap_method", "advanced.advanced.debug_tools.refiner_swap_method", "Refiner swap method"),
    ProtocolSlot("controlnet_softness", "advanced.advanced.control.controlnet_softness", "ControlNet softness"),

    ProtocolSlot("freeu_ctrls", "advanced.advanced.freeu.freeu_ctrls", "FreeU enable/b1/b2/s1/s2", spread=True),

    ProtocolSlot("debugging_inpaint_preprocessor", "advanced.advanced.inpaint_advanced.debugging_inpaint_preprocessor", "Debug inpaint preprocessor"),
    ProtocolSlot("inpaint_disable_initial_latent", "advanced.advanced.inpaint_advanced.inpaint_disable_initial_latent", "Disable initial latent in inpaint"),
    ProtocolSlot("inpaint_engine", "advanced.advanced.inpaint_advanced.inpaint_engine", "Inpaint engine version"),
    ProtocolSlot("inpaint_strength", "advanced.advanced.inpaint_advanced.inpaint_strength", "Inpaint denoising strength"),
    ProtocolSlot("inpaint_respective_field", "advanced.advanced.inpaint_advanced.inpaint_respective_field", "Inpaint respective field"),
    ProtocolSlot("inpaint_advanced_masking_checkbox", "image_input.inpaint.inpaint_advanced_masking_checkbox", "Enable advanced masking"),
    ProtocolSlot("invert_mask_checkbox", "image_input.inpaint.invert_mask_checkbox", "Invert inpaint mask"),
    ProtocolSlot("inpaint_erode_or_dilate", "advanced.advanced.inpaint_advanced.inpaint_erode_or_dilate", "Inpaint mask erode/dilate"),

    ProtocolSlot("save_final_enhanced_image_only", "advanced.advanced.debug_tools.save_final_enhanced_image_only", "Save only final enhanced image", condition=_enable_image_log),

    ProtocolSlot("save_metadata_to_images", "advanced.advanced.debug_tools.save_metadata_to_images", "Embed metadata in output images", condition=_enable_metadata),
    ProtocolSlot("metadata_scheme", "advanced.advanced.debug_tools.metadata_scheme", "Metadata format scheme", condition=_enable_metadata),

    ProtocolSlot("ip_ctrls", "image_input.ip.ip_ctrls", "Image Prompt image/type/stop/weight per slot", spread=True),

    ProtocolSlot("debugging_dino", "advanced.advanced.inpaint_advanced.debugging_dino", "Debug DINO detection"),
    ProtocolSlot("dino_erode_or_dilate", "advanced.advanced.inpaint_advanced.dino_erode_or_dilate", "DINO mask erode/dilate"),
    ProtocolSlot("debugging_enhance_masks_checkbox", "advanced.advanced.inpaint_advanced.debugging_enhance_masks_checkbox", "Debug enhance masks"),
    ProtocolSlot("enhance_input_image", "image_input.enhance_input_image", "Enhance source image"),
    ProtocolSlot("enhance_checkbox", "top_checkboxes.enhance_checkbox", "Enhance panel toggle"),
    ProtocolSlot("enhance_uov_method", "enhance_panel.enhance_uov_method", "Enhance upscale/variation method"),
    ProtocolSlot("enhance_uov_processing_order", "enhance_panel.enhance_uov_processing_order", "Enhance processing order"),
    ProtocolSlot("enhance_uov_prompt_type", "enhance_panel.enhance_uov_prompt_type", "Enhance prompt type selection"),

    ProtocolSlot("enhance_ctrls", "enhance_panel.enhance_ctrls", "Per-tab enhance enabled/prompt/mask/inpaint params", spread=True),
]

CTRLS_EXPECTED_LENGTH = 170


def build_load_data_outputs(
    top_checkboxes: TopCheckboxes,
    prompt_buttons: PromptAndButtons,
    image_input: ImageInputTabs,
    enhance_panel: EnhancePanelComponents,
    advanced: AdvancedColumnComponents,
    inpaint_engine_state,
):
    context = {
        'top_checkboxes': top_checkboxes,
        'prompt_buttons': prompt_buttons,
        'image_input': image_input,
        'enhance_panel': enhance_panel,
        'advanced': advanced,
        'inpaint_engine_state': inpaint_engine_state,
    }
    result, _ = validate_and_build(
        LOAD_DATA_OUTPUTS_DECLARATION, context,
        "load_data_outputs",
        expected_length=LOAD_DATA_OUTPUTS_EXPECTED_LENGTH
    )
    return result


def build_ctrls(
    currentTask,
    prompt_buttons: PromptAndButtons,
    top_checkboxes: TopCheckboxes,
    image_input: ImageInputTabs,
    enhance_panel: EnhancePanelComponents,
    advanced: AdvancedColumnComponents,
    current_tab,
):
    context = {
        'currentTask': currentTask,
        'prompt_buttons': prompt_buttons,
        'top_checkboxes': top_checkboxes,
        'image_input': image_input,
        'enhance_panel': enhance_panel,
        'advanced': advanced,
        'current_tab': current_tab,
    }
    result, _ = validate_and_build(
        CTRLS_DECLARATION, context,
        "ctrls",
        expected_length=CTRLS_EXPECTED_LENGTH
    )
    return result
