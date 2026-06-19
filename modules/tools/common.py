import json
import sys
from dataclasses import dataclass, field, asdict
from enum import IntEnum, Enum
from typing import Any, Dict, List, Optional, Union


class ExitCode(IntEnum):
    SUCCESS = 0
    WARNING = 1
    ERROR = 2
    INVALID_ARGS = 3


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CheckResult:
    name: str
    severity: Severity
    message: str
    suggestion: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "detail": self.detail,
        }


@dataclass
class ToolResult:
    tool_name: str
    exit_code: ExitCode = ExitCode.SUCCESS
    checks: List[CheckResult] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def add_check(self, check: CheckResult) -> None:
        self.checks.append(check)
        if check.severity == Severity.ERROR:
            if self.exit_code.value < ExitCode.ERROR.value:
                self.exit_code = ExitCode.ERROR
        elif check.severity == Severity.WARNING:
            if self.exit_code.value < ExitCode.WARNING.value:
                self.exit_code = ExitCode.WARNING

    def add_info(self, name: str, message: str, detail: Optional[Dict[str, Any]] = None) -> None:
        self.add_check(CheckResult(
            name=name,
            severity=Severity.INFO,
            message=message,
            detail=detail,
        ))

    def add_warning(self, name: str, message: str, suggestion: Optional[str] = None,
                    detail: Optional[Dict[str, Any]] = None) -> None:
        self.add_check(CheckResult(
            name=name,
            severity=Severity.WARNING,
            message=message,
            suggestion=suggestion,
            detail=detail,
        ))

    def add_error(self, name: str, message: str, suggestion: Optional[str] = None,
                  detail: Optional[Dict[str, Any]] = None) -> None:
        self.add_check(CheckResult(
            name=name,
            severity=Severity.ERROR,
            message=message,
            suggestion=suggestion,
            detail=detail,
        ))

    @property
    def errors(self) -> List[CheckResult]:
        return [c for c in self.checks if c.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if c.severity == Severity.WARNING]

    @property
    def infos(self) -> List[CheckResult]:
        return [c for c in self.checks if c.severity == Severity.INFO]

    def compute_summary(self) -> str:
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        info_count = len(self.infos)
        parts = [f"[{self.tool_name}]"]
        if error_count > 0:
            parts.append(f"{error_count} error(s)")
        if warning_count > 0:
            parts.append(f"{warning_count} warning(s)")
        if info_count > 0:
            parts.append(f"{info_count} info")
        if error_count == 0 and warning_count == 0:
            parts.append("All checks passed")
        self.summary = " ".join(parts)
        return self.summary

    def to_dict(self) -> Dict[str, Any]:
        self.compute_summary()
        return {
            "tool_name": self.tool_name,
            "exit_code": self.exit_code.value,
            "exit_code_name": self.exit_code.name,
            "summary": self.summary,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "infos_count": len(self.infos),
            "checks": [c.to_dict() for c in self.checks],
            "data": self.data,
        }


def get_json_output(result: ToolResult, indent: int = 2) -> str:
    return json.dumps(result.to_dict(), indent=indent, ensure_ascii=False, default=str)


def get_human_output(result: ToolResult, verbose: bool = False) -> str:
    result.compute_summary()
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {result.tool_name.upper()} - {result.summary}")
    lines.append("=" * 60)
    lines.append("")

    if result.errors:
        lines.append(f"ERRORS ({len(result.errors)}):")
        lines.append("-" * 40)
        for i, check in enumerate(result.errors, 1):
            lines.append(f"  [{i}] {check.message}")
            if check.suggestion:
                lines.append(f"      Suggestion: {check.suggestion}")
            if verbose and check.detail:
                lines.append(f"      Detail: {json.dumps(check.detail, ensure_ascii=False)}")
        lines.append("")

    if result.warnings:
        lines.append(f"WARNINGS ({len(result.warnings)}):")
        lines.append("-" * 40)
        for i, check in enumerate(result.warnings, 1):
            lines.append(f"  [{i}] {check.message}")
            if check.suggestion:
                lines.append(f"      Suggestion: {check.suggestion}")
            if verbose and check.detail:
                lines.append(f"      Detail: {json.dumps(check.detail, ensure_ascii=False)}")
        lines.append("")

    if result.infos and verbose:
        lines.append(f"INFO ({len(result.infos)}):")
        lines.append("-" * 40)
        for i, check in enumerate(result.infos, 1):
            lines.append(f"  [{i}] {check.message}")
            if check.detail:
                lines.append(f"      Detail: {json.dumps(check.detail, ensure_ascii=False)}")
        lines.append("")

    if not verbose and result.infos:
        lines.append(f"Info: {len(result.infos)} informational item(s) (use --verbose to see)")
        lines.append("")

    if result.data:
        lines.append("DATA:")
        lines.append("-" * 40)
        for k, v in result.data.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines.append("=" * 60)
    lines.append(f"Exit Code: {result.exit_code.value} ({result.exit_code.name})")
    lines.append("=" * 60)

    return "\n".join(lines)


def print_result(result: ToolResult, output_json: bool = False, verbose: bool = False) -> None:
    if output_json:
        print(get_json_output(result))
    else:
        print(get_human_output(result, verbose))


def exit_with_result(result: ToolResult, output_json: bool = False, verbose: bool = False, exit_: bool = True) -> int:
    print_result(result, output_json=output_json, verbose=verbose)
    code = result.exit_code.value
    if exit_:
        sys.exit(code)
    return code


def add_common_args(parser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results in JSON format (CI-friendly)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including info-level items",
    )
