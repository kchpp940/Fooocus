import os
import sys
import json
from typing import Any, Dict, List, Optional

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from modules.tools.common import ToolResult


FOOOCUS_METADATA_REQUIRED_KEYS = [
    "prompt",
    "negative_prompt",
    "base_model",
    "seed",
    "steps",
    "performance",
    "resolution",
    "guidance_scale",
    "sharpness",
    "sampler",
    "scheduler",
    "version",
]

FOOOCUS_METADATA_OPTIONAL_KEYS = [
    "styles",
    "refiner_model",
    "refiner_switch",
    "vae",
    "clip_skip",
    "adaptive_cfg",
    "adm_guidance",
    "overwrite_switch",
    "overwrite_step",
    "overwrite_upscale",
    "base_model_hash",
    "refiner_model_hash",
    "created_by",
    "full_prompt",
    "full_negative_prompt",
    "loras",
    "inpaint_engine_version",
    "inpaint_method",
]


def add_validate_metadata_args(parser) -> None:
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a PNG/JPEG/WebP image to extract and validate metadata from",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default=None,
        help="Path to a JSON file containing metadata to validate against schema",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["fooocus", "a1111", "auto"],
        default="auto",
        help="Metadata scheme to validate against (default: auto-detect)",
    )
    parser.add_argument(
        "--lax",
        action="store_true",
        help="Treat missing optional/non-critical fields as INFO instead of WARNING",
    )


def _detect_scheme(data: Any) -> Optional[str]:
    if isinstance(data, dict):
        return "fooocus"
    if isinstance(data, str):
        stripped = data.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                json.loads(stripped)
                return "fooocus"
            except Exception:
                pass
        return "a1111"
    return None


def _validate_fooocus_dict(result: ToolResult, meta: Dict[str, Any], lax: bool) -> None:
    for key in FOOOCUS_METADATA_REQUIRED_KEYS:
        if key not in meta or meta.get(key) in (None, "", "None"):
            if lax:
                result.add_info(
                    name=f"meta_missing_{key}",
                    message=f"Required-ish field '{key}' missing or empty (lax mode)",
                    detail={"key": key},
                )
            else:
                result.add_warning(
                    name=f"meta_missing_{key}",
                    message=f"Expected field '{key}' missing or empty in Fooocus metadata",
                    suggestion="Ensure the generator populates this field for reproducibility.",
                    detail={"key": key, "keys_present": sorted(list(meta.keys()))[:30]},
                )

    if "performance" in meta and meta["performance"]:
        try:
            import modules.flags as flags
            if meta["performance"] not in flags.Performance.values():
                result.add_warning(
                    name="meta_performance_value",
                    message=f"performance='{meta['performance']}' is not a known Performance value",
                    detail={"valid_values": list(flags.Performance.values())},
                )
        except Exception:
            pass

    if "sampler" in meta and meta["sampler"]:
        try:
            import modules.flags as flags
            if meta["sampler"] not in flags.sampler_list:
                result.add_warning(
                    name="meta_sampler_value",
                    message=f"sampler='{meta['sampler']}' not in registered sampler_list",
                )
        except Exception:
            pass

    if "scheduler" in meta and meta["scheduler"]:
        try:
            import modules.flags as flags
            if meta["scheduler"] not in flags.scheduler_list:
                result.add_warning(
                    name="meta_scheduler_value",
                    message=f"scheduler='{meta['scheduler']}' not in registered scheduler_list",
                )
        except Exception:
            pass

    if "resolution" in meta:
        res = meta["resolution"]
        try:
            if isinstance(res, str):
                w, h = eval(res)
                if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
                    raise ValueError(f"bad dims: {w}x{h}")
            elif isinstance(res, tuple) and len(res) == 2:
                w, h = res
            else:
                raise ValueError(f"unsupported type: {type(res)}")
            result.add_info(
                name="meta_resolution",
                message=f"Resolution valid: {w}x{h}",
                detail={"width": w, "height": h},
            )
        except Exception as e:
            result.add_error(
                name="meta_resolution",
                message=f"Resolution field is malformed ({type(res).__name__}): {e}",
                detail={"raw_value": str(res)[:200]},
            )

    all_keys = set(FOOOCUS_METADATA_REQUIRED_KEYS) | set(FOOOCUS_METADATA_OPTIONAL_KEYS)
    unknown = [k for k in meta if k not in all_keys and not k.startswith("lora_combined_")]
    if unknown:
        result.add_info(
            name="meta_extra_keys",
            message=f"{len(unknown)} extra non-standard key(s) present",
            detail={"extra_keys": unknown[:20]},
        )

    result.add_info(
        name="meta_schema_ok",
        message=f"Fooocus JSON metadata structure valid ({len(meta)} keys)",
        detail={"keys_count": len(meta)},
    )


def _validate_a1111_string(result: ToolResult, meta: str, lax: bool) -> None:
    lines = meta.strip().splitlines()
    if not lines:
        result.add_error(
            name="meta_a1111_empty",
            message="A1111 metadata string is empty",
        )
        return

    lastline = lines[-1]
    has_params = any(marker in lastline for marker in ["Steps:", "Sampler:", "CFG scale:", "Size:"])
    if not has_params:
        if lax:
            result.add_info(
                name="meta_a1111_params_missing",
                message="Last line doesn't look like A1111 params (lax mode)",
            )
        else:
            result.add_warning(
                name="meta_a1111_params_missing",
                message="Last line does not contain typical A1111 generation params (Steps, Sampler, ...)",
                suggestion="Ensure the params line is included, or validate as scheme=fooocus instead.",
            )

    if "Negative prompt:" in meta:
        result.add_info(
            name="meta_a1111_has_negative",
            message="A1111 metadata contains 'Negative prompt:' section",
        )

    result.add_info(
        name="meta_a1111_ok",
        message=f"A1111 text metadata structurally plausible ({len(lines)} lines)",
        detail={"lines_count": len(lines), "char_count": len(meta)},
    )


def _validate_from_image(result: ToolResult, image_path: str, scheme: str, lax: bool) -> None:
    if not os.path.exists(image_path):
        result.add_error(
            name="image_not_found",
            message=f"Image not found: {image_path}",
        )
        return

    try:
        from PIL import Image
        from modules.meta_parser import read_info_from_image, MetadataScheme
    except Exception as e:
        result.add_error(
            name="image_deps",
            message=f"Cannot import PIL/meta_parser to read image metadata: {e}",
            suggestion="Install Pillow: pip install Pillow",
        )
        return

    try:
        with Image.open(image_path) as img:
            parameters, detected_scheme = read_info_from_image(img)
    except Exception as e:
        result.add_error(
            name="image_open",
            message=f"Failed to open image '{image_path}': {type(e).__name__}: {e}",
        )
        return

    result.data["image_path"] = image_path
    result.data["image_size"] = None
    try:
        with Image.open(image_path) as img:
            result.data["image_size"] = {"width": img.width, "height": img.height, "mode": img.mode}
    except Exception:
        pass

    if detected_scheme is not None:
        detected_str = detected_scheme.value if hasattr(detected_scheme, "value") else str(detected_scheme)
        result.add_info(
            name="image_scheme_detected",
            message=f"Detected metadata scheme: {detected_str}",
            detail={"scheme": detected_str},
        )
    else:
        result.add_warning(
            name="image_no_scheme",
            message="Could not detect a supported metadata scheme in the image",
            suggestion="Parameters/EXIF data may be missing or corrupted.",
        )

    if parameters is None:
        result.add_warning(
            name="image_no_metadata",
            message="No metadata payload extracted from the image",
            detail={"has_parameters": False},
        )
        return

    chosen_scheme = scheme
    if chosen_scheme == "auto":
        if isinstance(parameters, dict):
            chosen_scheme = "fooocus"
        elif isinstance(parameters, str):
            chosen_scheme = "a1111"
        else:
            chosen_scheme = None

    if chosen_scheme == "fooocus" and isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
            chosen_scheme = "fooocus"
        except Exception:
            chosen_scheme = "a1111"

    if isinstance(parameters, dict):
        _validate_fooocus_dict(result, parameters, lax)
    elif isinstance(parameters, str):
        _validate_a1111_string(result, parameters, lax)
    else:
        result.add_error(
            name="meta_unknown_type",
            message=f"Unsupported metadata type {type(parameters).__name__}",
        )


def validate_metadata(args: Any) -> ToolResult:
    result = ToolResult(tool_name="validate-metadata")

    lax = getattr(args, "lax", False)
    scheme = getattr(args, "scheme", "auto")

    image = getattr(args, "image", None)
    json_file = getattr(args, "json_file", None)

    any_source = False

    if image:
        any_source = True
        _validate_from_image(result, image, scheme, lax)

    if json_file:
        any_source = True
        if not os.path.exists(json_file):
            result.add_error(
                name="json_file_not_found",
                message=f"JSON file not found: {json_file}",
            )
        else:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                result.add_error(
                    name="json_file_parse",
                    message=f"JSON file is not valid: {e}",
                )
                data = None
            except Exception as e:
                result.add_error(
                    name="json_file_read",
                    message=f"Cannot read JSON file: {type(e).__name__}: {e}",
                )
                data = None

            if data is not None:
                chosen = scheme
                if chosen == "auto":
                    chosen = "fooocus"
                if chosen == "fooocus" and isinstance(data, dict):
                    _validate_fooocus_dict(result, data, lax)
                elif chosen == "a1111" and isinstance(data, str):
                    _validate_a1111_string(result, data, lax)
                elif isinstance(data, dict):
                    _validate_fooocus_dict(result, data, lax)
                elif isinstance(data, str):
                    _validate_a1111_string(result, data, lax)
                else:
                    result.add_error(
                        name="json_file_type",
                        message=f"JSON root has unexpected type {type(data).__name__}",
                    )

    if not any_source:
        result.add_info(
            name="schema_printout",
            message="No source given; printing Fooocus metadata schema reference",
            detail={
                "required_keys": FOOOCUS_METADATA_REQUIRED_KEYS,
                "optional_keys": FOOOCUS_METADATA_OPTIONAL_KEYS,
                "scheme_choices": ["fooocus", "a1111", "auto"],
            },
        )
        result.data["fooocus_schema_reference"] = {
            "required": FOOOCUS_METADATA_REQUIRED_KEYS,
            "optional": FOOOCUS_METADATA_OPTIONAL_KEYS,
        }

    try:
        from modules.flags import MetadataScheme
        schemes = [m.value for m in MetadataScheme]
        result.data["registered_schemes"] = schemes
    except Exception:
        pass

    result.compute_summary()
    return result
