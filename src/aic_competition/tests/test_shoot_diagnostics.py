import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("shoot_diagnostics_tested", ROOT / "tools" / "shoot_diagnostics.py")
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


class ShootDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = ROOT.parent.parent / ".project_runtime" / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        self.workspace = tempfile.TemporaryDirectory(prefix="shoot_diagnostics_", dir=temporary)
        self.addCleanup(self.workspace.cleanup)
        self.project = Path(self.workspace.name)

    def pulse(self):
        return {"mode": "live", "simulated": False, "pulse_result": "completed",
                "software_execution_complete": True,
                "serial_writes": [dict(command=command, status="written", bytes_written=8)
                                  for command in ("STOP", "START", "STOP", "STOP")]}

    def test_complete_software_pulse_never_claims_physical_confirmation(self):
        summary = DIAG.summarize_record(self.pulse())
        self.assertEqual(summary["finding"], "software_pulse_complete_physical_unknown")
        self.assertNotIn("actual_discharge_confirmed", summary["launcher"])
        self.assertEqual(DIAG.summarize_record({"mode": "live_start_pose_test",
                         "launch_result": self.pulse()})["finding"], summary["finding"])

    def test_failed_partial_or_repeated_start_remains_uncertain(self):
        for status, count in (("failed", None), ("short_write", 7), ("written", 7)):
            data = self.pulse()
            data["serial_writes"][1].update(status=status, bytes_written=count)
            self.assertEqual(DIAG.summarize_record(data)["finding"], "start_write_uncertain")
        data = self.pulse()
        data["serial_writes"].append(data["serial_writes"][1])
        self.assertEqual(DIAG.summarize_record(data)["finding"], "start_write_uncertain")

    def test_stop_must_follow_start_and_all_recorded_stops_must_succeed(self):
        data = self.pulse()
        data["serial_writes"] = data["serial_writes"][:2]
        self.assertEqual(DIAG.summarize_record(data)["finding"], "stop_write_uncertain")
        data = self.pulse()
        data["serial_writes"][-1]["status"] = "failed"
        self.assertEqual(DIAG.summarize_record(data)["finding"], "stop_write_uncertain")

    def test_simulation_missing_record_and_incomplete_states(self):
        for data, finding in (({"mode": "simulation"}, "simulation_only"),
                              ({"mode": "live", "launch_result": None}, "no_launcher_record"),
                              ({"mode": "live"}, "no_start_record")):
            self.assertEqual(DIAG.summarize_record(data)["finding"], finding)
        data = self.pulse()
        data["software_execution_complete"] = False
        self.assertEqual(DIAG.summarize_record(data)["finding"], "software_pulse_incomplete")

    def test_logs_only_read_result_json_and_include_older_live_among_simulations(self):
        for index in range(4):
            path = self.project / "runs" / "发射单项" / str(index) / "result.json"
            path.parent.mkdir(parents=True)
            data = self.pulse() if index == 0 else {"mode": "simulation"}
            data["private_key"] = "do-not-export"
            path.write_text(json.dumps(data), encoding="utf-8")
            os.utime(path, (1000 + index, 1000 + index))
        (self.project / "deepseek.env").write_text("never-read-this", encoding="utf-8")
        records = DIAG.recent_records(self.project)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[-1]["mode"], "live")
        self.assertNotIn("do-not-export", json.dumps(records))
        self.assertNotIn("never-read-this", json.dumps(records))

    def test_malformed_logs_do_not_abort_collection(self):
        for index, content in enumerate(("bad json", "[]", '{"serial_writes": [1]}')):
            path = self.project / "runs" / "固定路线" / str(index) / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(content, encoding="utf-8")
        records = DIAG.recent_records(self.project)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["read_status"] == "unavailable" for record in records))

    def test_missing_and_broken_alias_are_distinct(self):
        path = mock.Mock()
        path.is_symlink.return_value = False
        path.stat.side_effect = FileNotFoundError()
        self.assertEqual(DIAG.inspect_device(path)["status"], "missing")
        path.is_symlink.return_value = True
        self.assertEqual(DIAG.inspect_device(path)["status"], "broken_symlink")

    def test_device_metadata_and_unknown_occupancy_without_opening_device(self):
        path = mock.Mock()
        path.is_symlink.return_value = True
        path.stat.return_value = types.SimpleNamespace(st_mode=stat.S_IFCHR | 0o660,
                                                       st_uid=1000, st_gid=20, st_rdev=0)
        with mock.patch.object(DIAG, "command_output", return_value={"status": "tool_missing"}), \
                mock.patch.object(DIAG, "port_users", side_effect=RuntimeError("no fuser")), \
                mock.patch.object(DIAG.os, "major", return_value=0, create=True), \
                mock.patch.object(DIAG.os, "minor", return_value=0, create=True), \
                mock.patch.object(DIAG.os, "access", return_value=False), \
                mock.patch("builtins.open", side_effect=AssertionError("must not open device")):
            record = DIAG.inspect_device(path)
        self.assertTrue(record["character_device"])
        self.assertFalse(record["writable"])
        self.assertEqual(record["occupancy"]["status"], "unknown")

    def test_external_commands_are_read_only_argv_bounded_and_missing_tools_tolerated(self):
        with mock.patch.object(DIAG.shutil, "which", return_value=None):
            self.assertEqual(DIAG.command_output(["lsusb"])["status"], "tool_missing")
        response = types.SimpleNamespace(returncode=0, stdout="id", stderr="")
        with mock.patch.object(DIAG.shutil, "which", return_value="/usr/bin/udevadm"), \
                mock.patch.object(DIAG.subprocess, "run", return_value=response) as run:
            self.assertEqual(DIAG.command_output(["udevadm", "info", "--name=/dev/shoot"])["status"], "ok")
        self.assertEqual(run.call_args.args[0], ["/usr/bin/udevadm", "info", "--name=/dev/shoot"])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        with mock.patch.object(DIAG.shutil, "which", return_value="/bin/lsusb"), \
                mock.patch.object(DIAG.subprocess, "run", side_effect=subprocess.TimeoutExpired("lsusb", 2)):
            self.assertEqual(DIAG.command_output(["lsusb"])["status"], "unavailable")

    def test_linux_collects_despite_failed_preflight_and_never_opens_serial(self):
        with mock.patch.object(DIAG.sys, "platform", "linux"), \
                mock.patch.object(DIAG.Path, "glob", return_value=[]), \
                mock.patch.object(DIAG, "inspect_device", side_effect=lambda path: {"path": str(path), "status": "missing"}), \
                mock.patch.object(DIAG, "command_output", return_value={"status": "tool_missing"}), \
                contextlib.redirect_stdout(io.StringIO()):
            code = DIAG.run_diagnosis(self.project, "/dev/shoot", mock.Mock(side_effect=RuntimeError("missing")))
        self.assertEqual(code, 0)
        report = json.loads(next(self.project.rglob("diagnosis.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["preflight"]["status"], "failed")
        self.assertFalse(report["serial_opened"])
        self.assertFalse(report["commands_sent"])
        self.assertFalse(report["device_identity_confirmed"])
        self.assertEqual(report["devices"][0]["path"], str(Path("/dev/shoot")))

    def test_non_linux_does_not_fake_car_evidence(self):
        with mock.patch.object(DIAG.sys, "platform", "win32"), \
                mock.patch.object(DIAG, "inspect_device") as inspect, \
                mock.patch.object(DIAG, "command_output") as command, \
                contextlib.redirect_stdout(io.StringIO()):
            code = DIAG.run_diagnosis(self.project, "/dev/shoot", mock.Mock(side_effect=RuntimeError("not Linux")))
        self.assertEqual(code, 2)
        inspect.assert_not_called()
        command.assert_not_called()
        report = json.loads(next(self.project.rglob("diagnosis.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["devices"], [])
        self.assertIn("platform_note", report)


if __name__ == "__main__":
    unittest.main()
