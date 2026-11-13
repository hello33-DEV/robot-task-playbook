import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("manual_shoot_tested", ROOT / "tools" / "manual_shoot.py")
SHOOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHOOT)
START = bytes.fromhex("55 01 12 00 00 00 01 69")
STOP = bytes.fromhex("55 01 11 00 00 00 01 68")


class FakeSerial:
    def __init__(self, fail=None):
        self.frames = []
        self.closed = False
        self.fail = fail

    def write(self, frame):
        self.frames.append(frame)
        if self.fail:
            result = self.fail(self, frame)
            if result is not None:
                return result
        return len(frame)

    def close(self):
        self.closed = True

    def fileno(self):
        return 7


class ManualShootTests(unittest.TestCase):
    def setUp(self):
        temporary = ROOT.parent.parent / ".project_runtime" / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        self.workspace = tempfile.TemporaryDirectory(prefix="manual_shoot_", dir=temporary)
        self.addCleanup(self.workspace.cleanup)

    def main(self, args):
        with mock.patch.object(SHOOT, "PROJECT", Path(self.workspace.name)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = SHOOT.main(args)
        records = [json.loads(path.read_text(encoding="utf-8"))
                   for path in Path(self.workspace.name).rglob("result.json")]
        return code, records

    def execute(self, transport):
        record = SHOOT.new_record(live=True)
        with contextlib.redirect_stdout(io.StringIO()):
            SHOOT.execute_once(transport, record)
        return record

    def test_official_frames_and_single_short_pulse(self):
        transport = FakeSerial()
        record = self.execute(transport)
        self.assertEqual(transport.frames, [STOP, START, STOP, STOP])
        self.assertEqual(SHOOT.PORT, "/dev/shoot")
        self.assertEqual(SHOOT.CONFIG["pulse_seconds"], 0.10)
        self.assertEqual(record["events"], ["PULSE_REQUEST_ATTEMPT", "PULSE_REQUEST_ACCEPTED",
                                            "PULSE_EXECUTION_COMPLETE"])
        self.assertTrue(record["request_accepted"])
        self.assertTrue(record["software_execution_complete"])
        self.assertFalse(record["worker_still_running"])
        self.assertTrue(transport.closed)
        self.assertEqual([entry["command"] for entry in record["serial_writes"]],
                         ["STOP", "START", "STOP", "STOP"])
        self.assertTrue(all(entry["status"] == "written" and entry["bytes_written"] == 8
                            for entry in record["serial_writes"]))
        for name in ("device_acknowledged", "actual_discharge_confirmed", "physical_stop_confirmed", "hit_confirmed"):
            self.assertFalse(record[name])

    def test_official_sequence_is_start_wait_stop_without_pulse_driver(self):
        actions = []
        transport = FakeSerial(lambda channel, frame: actions.append(("write", frame)))
        record = SHOOT.new_record(live=True)
        with mock.patch.object(SHOOT, "PulseDriver") as driver, \
                mock.patch.object(SHOOT.time, "sleep", side_effect=lambda seconds: actions.append(("sleep", seconds))), \
                mock.patch("builtins.print") as printing:
            SHOOT.execute_official_once(transport, record)
        self.assertEqual(actions, [("write", START), ("sleep", 0.10), ("write", STOP)])
        self.assertEqual(record["execution_mode"], "official_serial_sequence")
        self.assertTrue(record["request_accepted"])
        self.assertTrue(record["software_execution_complete"])
        self.assertEqual(record["pulse_result"], "completed")
        self.assertFalse(record["worker_still_running"])
        self.assertTrue(transport.closed)
        self.assertEqual([entry["command"] for entry in record["serial_writes"]], ["START", "STOP"])
        for key in ("device_acknowledged", "actual_discharge_confirmed", "physical_stop_confirmed", "hit_confirmed"):
            self.assertFalse(record[key])
        driver.assert_not_called()
        printing.assert_not_called()

    def test_official_simulation_does_not_open_check_prompt_or_use_driver(self):
        with mock.patch.object(SHOOT, "open_serial") as opening, \
                mock.patch.object(SHOOT, "check_live") as checking, \
                mock.patch.object(SHOOT, "PulseDriver") as driver, \
                mock.patch.object(SHOOT.time, "sleep") as sleep, \
                mock.patch("builtins.input") as prompt:
            code, records = self.main(["--official"])
        self.assertEqual(code, 0)
        self.assertEqual(records[0]["status"], "simulation_complete")
        self.assertEqual(records[0]["execution_mode"], "official_serial_sequence")
        self.assertTrue(records[0]["simulated"])
        self.assertEqual(records[0]["serial_writes"], [])
        self.assertFalse(records[0]["actual_discharge_confirmed"])
        sleep.assert_called_once_with(0.10)
        for operation in (opening, checking, driver, prompt):
            operation.assert_not_called()

    def test_official_live_uses_only_official_sequence_with_fake_transport(self):
        transport = FakeSerial()
        with mock.patch.object(SHOOT, "check_live"), \
                mock.patch.object(SHOOT.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="FIRE"), \
                mock.patch.object(SHOOT.time, "sleep"), \
                mock.patch.object(SHOOT, "execute_once") as old_execution, \
                mock.patch.object(SHOOT, "open_serial", return_value=transport) as opening:
            code, records = self.main(["--live", "--official"])
        self.assertEqual(code, 0)
        self.assertEqual(transport.frames, [START, STOP])
        self.assertTrue(transport.closed)
        self.assertEqual(records[0]["status"], "software_complete")
        self.assertFalse(records[0]["actual_discharge_confirmed"])
        opening.assert_called_once_with()
        old_execution.assert_not_called()

    def test_official_failures_stop_close_and_never_repeat_start(self):
        for point in ("start_short", "start_error", "stop_short", "stop_error", "wait", "interrupt", "close"):
            with self.subTest(point=point):
                def fail(channel, frame):
                    if (frame == START and point.startswith("start")) or (frame == STOP and point.startswith("stop")):
                        if point.endswith("short"):
                            return 7
                        raise OSError(point)
                transport = FakeSerial(fail)
                original_close = transport.close

                def close():
                    original_close()
                    if point == "close":
                        raise OSError(point)

                transport.close = close
                record = SHOOT.new_record(live=True)
                error = KeyboardInterrupt() if point == "interrupt" else OSError("wait") if point == "wait" else None
                with mock.patch.object(SHOOT.time, "sleep", side_effect=error), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaises(KeyboardInterrupt if point == "interrupt" else OSError):
                    SHOOT.execute_official_once(transport, record)
                self.assertEqual(transport.frames, [START, STOP])
                self.assertTrue(transport.closed)
                self.assertFalse(record["software_execution_complete"])
                self.assertEqual(record["pulse_result"], "cancelled" if point == "interrupt" else "failed")
                self.assertTrue(record["fault"])
                self.assertFalse(record["actual_discharge_confirmed"])

    def test_official_log_failure_before_start_closes_without_writes(self):
        transport = FakeSerial()
        record = SHOOT.new_record(live=True)
        with self.assertRaises(OSError), mock.patch.object(SHOOT.time, "sleep") as sleep:
            SHOOT.execute_official_once(transport, record, mock.Mock(side_effect=OSError("log")))
        self.assertEqual(transport.frames, [])
        self.assertTrue(transport.closed)
        self.assertFalse(record["software_execution_complete"])
        sleep.assert_not_called()

    def test_official_keeps_preflight_before_confirmation_and_open(self):
        with mock.patch.object(SHOOT, "check_live", side_effect=OSError("missing port")), \
                mock.patch.object(SHOOT, "open_serial") as opening, \
                mock.patch("builtins.input") as prompt:
            code, records = self.main(["--live", "--official"])
        self.assertEqual(code, 2)
        self.assertEqual(records[0]["status"], "failed")
        opening.assert_not_called()
        prompt.assert_not_called()

    def test_official_cancelled_confirmation_does_not_open(self):
        with mock.patch.object(SHOOT, "check_live"), \
                mock.patch.object(SHOOT.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value=""), \
                mock.patch.object(SHOOT, "open_serial") as opening:
            code, records = self.main(["--live", "--official"])
        self.assertEqual(code, 0)
        self.assertEqual(records[0]["status"], "cancelled_before_open")
        opening.assert_not_called()

    def test_official_rejects_readonly_modes(self):
        for mode in ("--check", "--diagnose"):
            with self.subTest(mode=mode), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as caught:
                SHOOT.main(["--official", mode])
            self.assertEqual(caught.exception.code, 2)

    def test_default_is_disconnected_simulation(self):
        with mock.patch.object(SHOOT, "open_serial") as opening, \
                mock.patch.object(SHOOT, "check_live") as checking, \
                mock.patch("builtins.input") as prompt:
            code, records = self.main([])
        self.assertEqual(code, 0)
        self.assertEqual(records[0]["status"], "simulation_complete")
        self.assertTrue(records[0]["simulated"])
        self.assertIsNone(records[0]["port"])
        self.assertFalse(records[0]["actual_discharge_confirmed"])
        self.assertEqual(records[0]["serial_writes"], [])
        opening.assert_not_called()
        checking.assert_not_called()
        prompt.assert_not_called()

    def test_check_is_read_only(self):
        with mock.patch.object(SHOOT, "check_live") as checking, \
                mock.patch.object(SHOOT, "open_serial") as opening:
            code, records = self.main(["--check"])
        self.assertEqual((code, records), (0, []))
        checking.assert_called_once_with()
        opening.assert_not_called()

    def test_diagnosis_does_not_open_prompt_or_execute(self):
        import shoot_diagnostics
        with mock.patch.object(shoot_diagnostics, "run_diagnosis", return_value=0) as diagnose, \
                mock.patch.object(SHOOT, "open_serial") as opening, \
                mock.patch.object(SHOOT, "execute_once") as execute, \
                mock.patch("builtins.input") as prompt:
            code, records = self.main(["--diagnose"])
        self.assertEqual((code, records), (0, []))
        diagnose.assert_called_once_with(Path(self.workspace.name), SHOOT.PORT, SHOOT.check_live)
        opening.assert_not_called()
        execute.assert_not_called()
        prompt.assert_not_called()

    def test_diagnosis_cannot_be_combined_with_live(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            SHOOT.main(["--live", "--diagnose"])
        self.assertEqual(caught.exception.code, 2)

    def test_failed_preflight_does_not_prompt_or_open(self):
        with mock.patch.object(SHOOT, "check_live", side_effect=RuntimeError("occupied")), \
                mock.patch.object(SHOOT, "open_serial") as opening, \
                mock.patch("builtins.input") as prompt:
            code, records = self.main(["--live"])
        self.assertEqual(code, 2)
        self.assertEqual(records[0]["status"], "failed")
        opening.assert_not_called()
        prompt.assert_not_called()

    def test_declined_confirmation_never_opens(self):
        with mock.patch.object(SHOOT, "check_live"), \
                mock.patch.object(SHOOT.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value=""), \
                mock.patch.object(SHOOT, "open_serial") as opening:
            code, records = self.main(["--live"])
        self.assertEqual(code, 0)
        self.assertEqual(records[0]["status"], "cancelled_before_open")
        self.assertIsNone(records[0]["request_accepted"])
        opening.assert_not_called()

    def test_live_rejects_piped_confirmation(self):
        with mock.patch.object(SHOOT, "check_live"), \
                mock.patch.object(SHOOT.sys.stdin, "isatty", return_value=False), \
                mock.patch("builtins.input") as prompt, \
                mock.patch.object(SHOOT, "open_serial") as opening:
            code, _ = self.main(["--live"])
        self.assertEqual(code, 2)
        prompt.assert_not_called()
        opening.assert_not_called()

    def test_live_main_with_fake_serial_only_once(self):
        transport = FakeSerial()
        with mock.patch.object(SHOOT, "check_live"), \
                mock.patch.object(SHOOT.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="FIRE"), \
                mock.patch.object(SHOOT, "open_serial", return_value=transport) as opening:
            code, records = self.main(["--live"])
        self.assertEqual(code, 0)
        self.assertEqual(records[0]["status"], "software_complete")
        self.assertFalse(records[0]["simulated"])
        self.assertFalse(records[0]["actual_discharge_confirmed"])
        self.assertEqual(transport.frames.count(START), 1)
        opening.assert_called_once_with()

    def test_startup_stop_failure_prevents_start(self):
        def fail(transport, frame):
            if frame == STOP:
                raise OSError("STOP failed")
        transport = FakeSerial(fail)
        record = SHOOT.new_record(live=True)
        with self.assertRaisesRegex(RuntimeError, "启动STOP"):
            SHOOT.execute_once(transport, record)
        self.assertNotIn(START, transport.frames)
        self.assertFalse(record["software_execution_complete"])
        self.assertTrue(transport.closed)

    def test_start_short_write_or_error_never_completes_or_retries(self):
        for short in (True, False):
            with self.subTest(short=short):
                def fail(transport, frame):
                    if frame == START:
                        if short:
                            return len(frame) - 1
                        raise OSError("START failed")
                transport = FakeSerial(fail)
                record = SHOOT.new_record(live=True)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    SHOOT.execute_once(transport, record)
                self.assertEqual(transport.frames.count(START), 1)
                self.assertEqual(transport.frames[-1], STOP)
                self.assertEqual(record["pulse_result"], "failed")
                self.assertFalse(record["software_execution_complete"])
                self.assertTrue(transport.closed)
                entry = next(entry for entry in record["serial_writes"] if entry["command"] == "START")
                self.assertEqual(entry["status"], "short_write" if short else "failed")
                self.assertEqual(entry["bytes_written"], 7 if short else None)
                self.assertFalse(record["actual_discharge_confirmed"])

    def test_execution_stop_failure_is_not_success(self):
        def fail(transport, frame):
            if frame == STOP and START in transport.frames:
                raise OSError("STOP failed")
        transport = FakeSerial(fail)
        record = SHOOT.new_record(live=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            SHOOT.execute_once(transport, record)
        self.assertFalse(record["software_execution_complete"])
        self.assertTrue(transport.closed)

    def test_final_cleanup_stop_failure_is_not_success(self):
        def fail(transport, frame):
            if frame == STOP and transport.frames.count(STOP) == 3:
                raise OSError("cleanup STOP failed")
        transport = FakeSerial(fail)
        record = SHOOT.new_record(live=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "清理STOP"):
            SHOOT.execute_once(transport, record)
        self.assertFalse(record["software_execution_complete"])
        self.assertTrue(transport.closed)

    def test_acceptance_is_not_execution_complete(self):
        transport = FakeSerial()
        record = SHOOT.new_record(live=True)
        saved = []
        with contextlib.redirect_stdout(io.StringIO()):
            SHOOT.execute_once(transport, record, lambda: saved.append(dict(record)))
        accepted = [entry for entry in saved if entry["request_accepted"]]
        self.assertFalse(accepted[0]["software_execution_complete"])
        self.assertTrue(saved[-1]["software_execution_complete"])

    def test_log_failure_after_request_stops_and_does_not_repeat(self):
        transport = FakeSerial()
        record = SHOOT.new_record(live=True)
        def save():
            if record["request_accepted"]:
                raise OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            SHOOT.execute_once(transport, record, save)
        self.assertLessEqual(transport.frames.count(START), 1)
        self.assertEqual(transport.frames[-1], STOP)
        self.assertFalse(record["software_execution_complete"])
        self.assertTrue(transport.closed)

    def test_denied_request_still_closes_without_start(self):
        transport = FakeSerial()
        record = SHOOT.new_record(live=True)
        with mock.patch.object(SHOOT.PulseDriver, "request", return_value=False), \
                self.assertRaisesRegex(RuntimeError, "未被接受"):
            SHOOT.execute_once(transport, record)
        self.assertFalse(record["request_accepted"])
        self.assertNotIn(START, transport.frames)
        self.assertTrue(transport.closed)

    def test_interrupt_and_timeout_always_call_cleanup(self):
        for interrupted in (True, False):
            with self.subTest(interrupted=interrupted):
                driver = mock.Mock()
                driver.snapshot.return_value = dict(pulse_id="test", pulse_result="pending", fault=None, busy=True)
                driver.thread.is_alive.return_value = not interrupted
                if interrupted:
                    driver.thread.join.side_effect = KeyboardInterrupt
                record = SHOOT.new_record(live=True)
                expected = KeyboardInterrupt if interrupted else RuntimeError
                with mock.patch.object(SHOOT, "PulseDriver", return_value=driver), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(expected):
                    SHOOT.execute_once(None, record)
                driver.close.assert_called_once_with()
                driver.request.assert_called_once()
                self.assertFalse(record["software_execution_complete"])

    def test_missing_serial_dependency_fails_closed(self):
        with mock.patch.object(SHOOT, "check_port") as checking, \
                mock.patch.object(SHOOT.importlib.util, "find_spec", return_value=None), \
                self.assertRaisesRegex(RuntimeError, "pyserial"):
            SHOOT.check_live()
        checking.assert_called_once_with("/dev/shoot", label="发射器")

    def test_open_serial_uses_exclusion_and_official_settings(self):
        transport = FakeSerial()
        serial = types.SimpleNamespace(Serial=mock.Mock(return_value=transport))
        fcntl = types.SimpleNamespace(ioctl=mock.Mock())
        termios = types.SimpleNamespace(TIOCEXCL=123)
        with mock.patch.dict("sys.modules", serial=serial, fcntl=fcntl, termios=termios), \
                mock.patch.object(SHOOT, "check_live"), \
                mock.patch.object(SHOOT, "port_users", return_value={os.getpid()}):
            self.assertIs(SHOOT.open_serial(), transport)
        serial.Serial.assert_called_once_with("/dev/shoot", 9600, bytesize=8, parity="N", stopbits=1,
                                             timeout=0.05, write_timeout=0.10, exclusive=True)
        fcntl.ioctl.assert_called_once_with(7, 123)
        self.assertEqual(transport.frames, [])

    def test_open_race_or_exclusion_failure_closes_without_writing(self):
        for raced in (True, False):
            with self.subTest(raced=raced):
                transport = FakeSerial()
                serial = types.SimpleNamespace(Serial=mock.Mock(return_value=transport))
                fcntl = types.SimpleNamespace(ioctl=mock.Mock(side_effect=None if raced else OSError("exclusive")))
                termios = types.SimpleNamespace(TIOCEXCL=123)
                with mock.patch.dict("sys.modules", serial=serial, fcntl=fcntl, termios=termios), \
                        mock.patch.object(SHOOT, "check_live"), \
                        mock.patch.object(SHOOT, "port_users", return_value={os.getpid(), 999999}), \
                        self.assertRaises((RuntimeError, OSError)):
                    SHOOT.open_serial()
                self.assertTrue(transport.closed)
                self.assertEqual(transport.frames, [])

    def test_log_directory_failure_never_opens_device(self):
        with mock.patch.object(SHOOT.Path, "mkdir", side_effect=OSError("read only")), \
                mock.patch.object(SHOOT, "open_serial") as opening:
            code, _ = self.main(["--live"])
        self.assertEqual(code, 2)
        opening.assert_not_called()


if __name__ == "__main__":
    unittest.main()
