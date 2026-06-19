import os
import sys
from typing import Any, Dict, List, Tuple

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from modules.tools.common import ToolResult


EXPECTED_UOV_ORDER = [
    "Disabled",
    "Vary (Subtle)",
    "Vary (Strong)",
    "Upscale (1.5x)",
    "Upscale (2x)",
    "Upscale (Fast 2x)",
]

EXPECTED_INPUT_TAB_ORDER = [
    "uov_tab",
    "ip_tab",
    "inpaint_tab",
    "describe_tab",
    "enhance_tab",
    "metadata_tab",
]


def add_validate_protocol_args(parser) -> None:
    parser.add_argument(
        "--list-flags",
        action="store_true",
        help="Print all loaded flags lists and enums for review",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat deviations from expected ordering as ERROR instead of WARNING",
    )


def _check_duplicates(result: ToolResult, list_name: str, items: List[str],
                      strict: bool) -> None:
    seen: Dict[str, int] = {}
    for idx, item in enumerate(items):
        if item in seen:
            msg = f"Duplicate entry '{item}' in {list_name} (positions {seen[item]} and {idx})"
            if strict:
                result.add_error(
                    name=f"dupe_{list_name}_{item}",
                    message=msg,
                    suggestion="Remove the duplicate entry from the list definition.",
                )
            else:
                result.add_warning(
                    name=f"dupe_{list_name}_{item}",
                    message=msg,
                    suggestion="Remove the duplicate entry.",
                )
        else:
            seen[item] = idx


def _check_expected_order(
    result: ToolResult, list_name: str, actual: List[str],
    expected: List[str], strict: bool,
) -> None:
    mismatches: List[Tuple[int, str, str]] = []
    for idx, exp in enumerate(expected):
        if idx >= len(actual):
            mismatches.append((idx, exp, "<missing>"))
            continue
        if actual[idx] != exp:
            mismatches.append((idx, exp, actual[idx]))
    if len(actual) > len(expected):
        for idx in range(len(expected), len(actual)):
            mismatches.append((idx, "<unexpected>", actual[idx]))
    if mismatches:
        detail = {
            "expected_order": expected,
            "actual_order": actual,
            "mismatches": [
                {"index": i, "expected": e, "actual": a} for i, e, a in mismatches
            ],
        }
        msg = (f"Order mismatch for {list_name}: "
               f"{len(mismatches)} position(s) differ from expected")
        if strict:
            result.add_error(
                name=f"order_{list_name}",
                message=msg,
                suggestion="Reorder the list to match the documented protocol.",
                detail=detail,
            )
        else:
            result.add_warning(
                name=f"order_{list_name}",
                message=msg,
                suggestion="Reorder to match expected protocol unless the change is intentional.",
                detail=detail,
            )
    else:
        result.add_info(
            name=f"order_{list_name}",
            message=f"{list_name} order matches expected protocol",
        )


def _check_enum_consistency(result: ToolResult) -> None:
    try:
        import modules.flags as flags
        issues: List[str] = []

        perf_values = set(flags.Performance.values())
        for name, perf in [
            ("EXTREME_SPEED", flags.Performance.EXTREME_SPEED),
            ("LIGHTNING", flags.Performance.LIGHTNING),
            ("HYPER_SD", flags.Performance.HYPER_SD),
        ]:
            try:
                steps_int = int(perf.steps() or -1)
                lora = perf.lora_filename()
                result.add_info(
                    name=f"perf_{name}",
                    message=f"Performance {name}: steps={steps_int}, lora={lora}",
                    detail={"performance": perf.value, "steps": steps_int, "lora": lora},
                )
            except Exception as e:
                issues.append(f"{name}: {e}")

        if not flags.Performance.has_restricted_features(flags.Performance.SPEED.value):
            if flags.Performance.SPEED.has_restricted_features is not None:
                pass

        for ms in flags.MetadataScheme:
            if not ms.value:
                issues.append(f"MetadataScheme entry has empty value")

        if issues:
            for issue in issues:
                result.add_warning(
                    name="enum_consistency",
                    message=f"Enum inconsistency: {issue}",
                )
        else:
            result.add_info(
                name="enum_consistency",
                message="Performance and MetadataScheme enums appear consistent",
            )
    except Exception as e:
        result.add_warning(
            name="enum_consistency",
            message=f"Could not validate enum consistency: {e}",
        )


def _validate_sampler_scheduler_consistency(result: ToolResult, strict: bool) -> None:
    try:
        import modules.flags as flags

        sampler_list = flags.sampler_list
        scheduler_list = flags.scheduler_list

        result.add_info(
            name="sampler_count",
            message=f"Registered {len(sampler_list)} samplers, {len(scheduler_list)} schedulers",
            detail={
                "samplers": sampler_list[:10],
                "samplers_count": len(sampler_list),
                "schedulers_count": len(scheduler_list),
            },
        )

        _check_duplicates(result, "sampler_list", sampler_list, strict)
        _check_duplicates(result, "scheduler_list", scheduler_list, strict)

        if len(sampler_list) < 5:
            msg = f"Only {len(sampler_list)} samplers are registered (expected many more)"
            if strict:
                result.add_error(name="sampler_count", message=msg)
            else:
                result.add_warning(name="sampler_count", message=msg)

        for expected_sampler in ["dpmpp_2m_sde_gpu", "dpmpp_2m_sde", "euler", "lcm"]:
            if expected_sampler not in sampler_list:
                result.add_warning(
                    name=f"sampler_missing_{expected_sampler}",
                    message=f"Expected sampler '{expected_sampler}' not in sampler_list",
                )

        for expected_scheduler in ["karras", "normal", "ddim_uniform"]:
            if expected_scheduler not in scheduler_list:
                result.add_warning(
                    name=f"scheduler_missing_{expected_scheduler}",
                    message=f"Expected scheduler '{expected_scheduler}' not in scheduler_list",
                )
    except Exception as e:
        result.add_warning(
            name="sampler_scheduler_consistency",
            message=f"Could not validate sampler/scheduler lists: {e}",
        )


def validate_protocol(args: Any) -> ToolResult:
    result = ToolResult(tool_name="validate-protocol")

    strict = getattr(args, "strict", False)

    try:
        import modules.flags as flags

        result.data["uov_list"] = flags.uov_list
        result.data["input_image_tab_ids"] = flags.input_image_tab_ids
        result.data["sampler_list"] = flags.sampler_list
        result.data["scheduler_list"] = flags.scheduler_list
        result.data["aspect_ratios_count"] = len(flags.sdxl_aspect_ratios)

        _check_expected_order(
            result, "uov_list", flags.uov_list,
            [getattr(flags, k, EXPECTED_UOV_ORDER[i]) for i, k in enumerate([
                "disabled", "subtle_variation", "strong_variation",
                "upscale_15", "upscale_2", "upscale_fast",
            ])],
            strict,
        )
        _check_expected_order(
            result, "input_image_tab_ids",
            flags.input_image_tab_ids, EXPECTED_INPUT_TAB_ORDER, strict,
        )

        _check_duplicates(result, "uov_list", flags.uov_list, strict)
        _check_duplicates(result, "input_image_tab_ids", flags.input_image_tab_ids, strict)
        _check_duplicates(result, "sdxl_aspect_ratios", flags.sdxl_aspect_ratios, strict)

        _validate_sampler_scheduler_consistency(result, strict)
        _check_enum_consistency(result)

        if getattr(args, "list_flags", False):
            result.data["performance_values"] = list(flags.Performance.values())
            result.data["output_formats"] = flags.OutputFormat.list()
            result.data["metadata_schemes"] = [m.value for m in flags.MetadataScheme]
            result.data["aspect_ratios"] = flags.sdxl_aspect_ratios
            result.data["inpaint_engine_versions"] = flags.inpaint_engine_versions
            result.data["ip_list"] = flags.ip_list
            result.add_info(
                name="flag_listing",
                message="Full flag listings included in JSON output (--list-flags)",
            )
    except Exception as e:
        result.add_error(
            name="flags_import",
            message=f"Failed to import modules.flags: {type(e).__name__}: {e}",
            suggestion="Ensure the working directory is the Fooocus project root.",
        )

    result.compute_summary()
    return result
