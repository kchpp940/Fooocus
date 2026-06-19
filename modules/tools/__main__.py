import sys
import os
import argparse

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from modules.tools.common import ExitCode, exit_with_result, ToolResult, add_common_args
from modules.tools.check_env import check_environment, add_check_env_args
from modules.tools.check_config import check_configuration, add_check_config_args
from modules.tools.scan_resources import scan_resources, add_scan_resources_args
from modules.tools.validate_protocol import validate_protocol, add_validate_protocol_args
from modules.tools.validate_metadata import validate_metadata, add_validate_metadata_args
from modules.tools.package_logs import package_logs, add_package_logs_args
from modules.tools.clean_cache import clean_cache, add_clean_cache_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modules.tools",
        description="Fooocus Unified Maintenance CLI - tools for environment checks, "
                    "configuration validation, resource scanning, and cache management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.tools check-env                    # Check Python environment and dependencies
  python -m modules.tools check-config --verbose       # Validate configuration with details
  python -m modules.tools scan-resources --json        # Scan models and output as JSON
  python -m modules.tools validate-protocol            # Validate WebUI protocol ordering
  python -m modules.tools validate-metadata --image output/img.png
  python -m modules.tools package-logs --output logs_bundle.tar.gz
  python -m modules.tools clean-cache --temp --hash    # Clear temp and hash caches
        """.strip()
    )
    subparsers = parser.add_subparsers(dest="command", help="Available maintenance commands")
    subparsers.required = True

    check_env_parser = subparsers.add_parser(
        "check-env",
        help="Check Python environment, version, and required dependencies",
    )
    add_check_env_args(check_env_parser)
    add_common_args(check_env_parser)

    check_config_parser = subparsers.add_parser(
        "check-config",
        help="Validate configuration against schema and check paths",
    )
    add_check_config_args(check_config_parser)
    add_common_args(check_config_parser)

    scan_res_parser = subparsers.add_parser(
        "scan-resources",
        help="Scan model directories and refresh file lists",
    )
    add_scan_resources_args(scan_res_parser)
    add_common_args(scan_res_parser)

    val_proto_parser = subparsers.add_parser(
        "validate-protocol",
        help="Validate WebUI protocol orderings and flag consistency",
    )
    add_validate_protocol_args(val_proto_parser)
    add_common_args(val_proto_parser)

    val_meta_parser = subparsers.add_parser(
        "validate-metadata",
        help="Validate metadata schema and check image metadata",
    )
    add_validate_metadata_args(val_meta_parser)
    add_common_args(val_meta_parser)

    pkg_logs_parser = subparsers.add_parser(
        "package-logs",
        help="Package output logs, configs, and diagnostics into a tarball",
    )
    add_package_logs_args(pkg_logs_parser)
    add_common_args(pkg_logs_parser)

    clean_parser = subparsers.add_parser(
        "clean-cache",
        help="Clean hash cache, temp directories, and output caches",
    )
    add_clean_cache_args(clean_parser)
    add_common_args(clean_parser)

    all_parser = subparsers.add_parser(
        "all",
        help="Run all check-style commands sequentially (check-env, check-config, validate-protocol)",
    )
    add_common_args(all_parser)

    return parser


def _run_all(args) -> ToolResult:
    from modules.tools.common import Severity

    combined = ToolResult(tool_name="all")
    commands = [
        ("check-env", lambda: check_environment(args)),
        ("check-config", lambda: check_configuration(args)),
        ("validate-protocol", lambda: validate_protocol(args)),
    ]
    for name, runner in commands:
        try:
            sub_result = runner()
            combined.data[f"{name}_exit_code"] = sub_result.exit_code.value
            combined.data[f"{name}_errors"] = len(sub_result.errors)
            combined.data[f"{name}_warnings"] = len(sub_result.warnings)
            for c in sub_result.checks:
                c.name = f"{name}/{c.name}"
                combined.add_check(c)
        except Exception as e:
            combined.add_error(
                name=f"{name}/exception",
                message=f"{name} raised an exception: {type(e).__name__}: {e}",
            )
    return combined


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_json = getattr(args, "output_json", False)
    verbose = getattr(args, "verbose", False)

    handlers = {
        "check-env": check_environment,
        "check-config": check_configuration,
        "scan-resources": scan_resources,
        "validate-protocol": validate_protocol,
        "validate-metadata": validate_metadata,
        "package-logs": package_logs,
        "clean-cache": clean_cache,
        "all": _run_all,
    }

    handler = handlers.get(args.command)
    if handler is None:
        result = ToolResult(tool_name=args.command, exit_code=ExitCode.INVALID_ARGS)
        result.add_error("unknown_command", f"Unknown command: {args.command}")
        return exit_with_result(result, output_json, verbose, exit_=False)

    try:
        result = handler(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        result = ToolResult(tool_name=args.command, exit_code=ExitCode.ERROR)
        result.add_error("interrupted", "Operation was interrupted by user")
        return exit_with_result(result, output_json, verbose, exit_=False)
    except Exception as e:
        import traceback
        result = ToolResult(tool_name=args.command, exit_code=ExitCode.ERROR)
        result.add_error(
            name="unhandled_exception",
            message=f"Unhandled {type(e).__name__}: {e}",
            suggestion="Use --verbose for the full traceback (stderr).",
            detail={"exception_type": type(e).__name__},
        )
        if verbose:
            traceback.print_exc()
        return exit_with_result(result, output_json, verbose, exit_=False)

    return exit_with_result(result, output_json, verbose, exit_=False)


if __name__ == "__main__":
    sys.exit(main())
