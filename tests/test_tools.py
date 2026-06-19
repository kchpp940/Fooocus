import sys
import os
import argparse
import unittest
import tempfile
import json
import pathlib

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from modules.tools.common import (
    ExitCode, Severity, CheckResult, ToolResult,
    get_json_output, get_human_output,
)


def _make_args(**overrides):
    ns = argparse.Namespace()
    defaults = {
        "output_json": False,
        "verbose": False,
        "skip_torch": True,
        "requirements_file": None,
        "config_path": None,
        "fix_paths": False,
        "preset": None,
        "models_only": False,
        "refresh": False,
        "extensions": None,
        "list_flags": False,
        "strict": False,
        "image": None,
        "json_file": None,
        "scheme": "auto",
        "lax": False,
        "output": None,
        "include_outputs": False,
        "include_models": False,
        "max_images": 2,
        "format": "gztar",
        "dry_run": True,
        "clean_hash": False,
        "clean_temp": False,
        "clean_outputs_log": False,
        "clean_gradio_temp": False,
        "clean_all": False,
        "force": True,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


class TestCommon(unittest.TestCase):
    def test_exit_code_values(self):
        self.assertEqual(int(ExitCode.SUCCESS), 0)
        self.assertEqual(int(ExitCode.WARNING), 1)
        self.assertEqual(int(ExitCode.ERROR), 2)
        self.assertEqual(int(ExitCode.INVALID_ARGS), 3)

    def test_tool_result_severity_propagation(self):
        r = ToolResult(tool_name="t")
        self.assertEqual(r.exit_code, ExitCode.SUCCESS)
        r.add_warning("w", "warn")
        self.assertEqual(r.exit_code, ExitCode.WARNING)
        r.add_error("e", "err")
        self.assertEqual(r.exit_code, ExitCode.ERROR)
        self.assertEqual(len(r.errors), 1)
        self.assertEqual(len(r.warnings), 1)

    def test_tool_result_summary_and_json(self):
        r = ToolResult(tool_name="mytest")
        r.add_info("ok", "everything is fine", detail={"k": "v"})
        summary = r.compute_summary()
        self.assertIn("All checks passed", summary)
        j = json.loads(get_json_output(r))
        self.assertEqual(j["tool_name"], "mytest")
        self.assertEqual(j["exit_code"], 0)
        self.assertEqual(j["infos_count"], 1)
        self.assertIn("All checks passed", j["summary"])
        text = get_human_output(r, verbose=True)
        self.assertIn("MYTEST", text)


class TestCheckEnv(unittest.TestCase):
    def test_basic_run(self):
        from modules.tools.check_env import check_environment
        args = _make_args()
        r = check_environment(args)
        self.assertEqual(r.tool_name, "check-env")
        self.assertIn("python_version", r.data)
        self.assertTrue(r.compute_summary())

    def test_json_output_structured(self):
        from modules.tools.check_env import check_environment
        args = _make_args()
        r = check_environment(args)
        j = json.loads(get_json_output(r))
        self.assertIn("dependencies", j["data"])
        self.assertIn("python_version", j["data"])


class TestCheckConfig(unittest.TestCase):
    def test_with_nonexistent_path(self):
        from modules.tools.check_config import check_configuration
        with tempfile.TemporaryDirectory() as td:
            bad_path = os.path.join(td, "no_such_config.txt")
            args = _make_args(config_path=bad_path)
            r = check_configuration(args)
            self.assertEqual(r.data["config_path"], bad_path)
            warnings = [c for c in r.checks if c.severity == Severity.WARNING]
            self.assertTrue(any("does not exist" in w.message for w in warnings))

    def test_invalid_json_config(self):
        from modules.tools.check_config import check_configuration
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "config.txt")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write("{ not valid json ,,, ")
            args = _make_args(config_path=cfg_path)
            r = check_configuration(args)
            errors = [c for c in r.checks if c.severity == Severity.ERROR]
            self.assertTrue(any("not valid JSON" in e.message for e in errors))

    def test_valid_config(self):
        from modules.tools.check_config import check_configuration
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "config.txt")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"default_performance": "Speed", "default_output_format": "png"}, f)
            args = _make_args(config_path=cfg_path, fix_paths=False)
            r = check_configuration(args)
            self.assertIsNotNone(r.exit_code)


class TestScanResources(unittest.TestCase):
    def test_empty_scan(self):
        from modules.tools.scan_resources import scan_resources
        args = _make_args(models_only=True)
        r = scan_resources(args)
        self.assertEqual(r.tool_name, "scan-resources")
        self.assertIn("total_model_files", r.data)

    def test_scan_with_temp_dirs(self):
        from modules.tools.scan_resources import scan_resources
        args = _make_args(models_only=False, refresh=False)
        r = scan_resources(args)
        j = json.loads(get_json_output(r))
        self.assertIn("models_by_category", j["data"])


class TestValidateProtocol(unittest.TestCase):
    def test_protocol_run(self):
        from modules.tools.validate_protocol import validate_protocol
        args = _make_args(strict=False, list_flags=False)
        r = validate_protocol(args)
        self.assertEqual(r.tool_name, "validate-protocol")
        j = json.loads(get_json_output(r))
        self.assertIn("uov_list", j["data"])
        self.assertIn("sampler_list", j["data"])


class TestValidateMetadata(unittest.TestCase):
    def test_schema_reference_no_source(self):
        from modules.tools.validate_metadata import validate_metadata
        args = _make_args()
        r = validate_metadata(args)
        self.assertIn("fooocus_schema_reference", r.data)

    def test_invalid_json_file(self):
        from modules.tools.validate_metadata import validate_metadata
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "meta.json")
            with open(bad, "w", encoding="utf-8") as f:
                f.write("{ bad ")
            args = _make_args(json_file=bad)
            r = validate_metadata(args)
            errors = [c for c in r.checks if c.severity == Severity.ERROR]
            self.assertTrue(any("not valid" in e.message for e in errors))

    def test_valid_json_metadata(self):
        from modules.tools.validate_metadata import validate_metadata
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, "meta.json")
            sample = {
                "prompt": "a cat",
                "negative_prompt": "",
                "base_model": "model.safetensors",
                "seed": 42,
                "steps": 30,
                "performance": "Speed",
                "resolution": "(1024, 1024)",
                "guidance_scale": 7.0,
                "sharpness": 2.0,
                "sampler": "dpmpp_2m_sde_gpu",
                "scheduler": "karras",
                "version": "1.0",
            }
            with open(good, "w", encoding="utf-8") as f:
                json.dump(sample, f)
            args = _make_args(json_file=good, lax=False)
            r = validate_metadata(args)
            self.assertIsNotNone(r.exit_code)


class TestPackageLogs(unittest.TestCase):
    def test_dry_run(self):
        from modules.tools.package_logs import package_logs
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "logs.tar.gz")
            args = _make_args(output=out, dry_run=True, include_outputs=False,
                              include_models=False, max_images=1, format="gztar")
            r = package_logs(args)
            self.assertEqual(r.tool_name, "package-logs")
            self.assertTrue(r.data["dry_run"])
            self.assertFalse(os.path.exists(out))


class TestCleanCache(unittest.TestCase):
    def test_no_action_selected(self):
        from modules.tools.clean_cache import clean_cache
        args = _make_args(clean_all=False, clean_hash=False, clean_temp=False,
                          clean_outputs_log=False, clean_gradio_temp=False,
                          dry_run=True, force=True)
        r = clean_cache(args)
        errors = [c for c in r.checks if c.severity == Severity.ERROR]
        self.assertTrue(any("No cleaning option" in e.message for e in errors))

    def test_hash_only_dry_run(self):
        from modules.tools.clean_cache import clean_cache
        args = _make_args(clean_all=False, clean_hash=True, clean_temp=False,
                          clean_outputs_log=False, clean_gradio_temp=False,
                          dry_run=True, force=True)
        r = clean_cache(args)
        self.assertIsNotNone(r.exit_code)


class TestCLIMain(unittest.TestCase):
    def test_help(self):
        from modules.tools.__main__ import build_parser
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--help"])

    def test_check_env_dispatch(self):
        from modules.tools.__main__ import main
        rc = main(["check-env", "--skip-torch", "--json"])
        self.assertIn(rc, [0, 1, 2])

    def test_all_command(self):
        from modules.tools.__main__ import main
        rc = main(["all", "--json"])
        self.assertIn(rc, [0, 1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
