import contextlib
import importlib.util
import io
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("manual_drive", ROOT / "tools" / "manual_drive.py")
DRIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVE)


class ManualDriveTests(unittest.TestCase):
    def control(self):
        return DRIVE.PulseControl(0.05, 0.20, 0.25)

    def test_zero_frame_matches_official_protocol(self):
        self.assertEqual(DRIVE.velocity_frame(DRIVE.ZERO).hex(), "abcd0d11000000000000000096")

    def test_signed_axes_and_checksum(self):
        frame = DRIVE.velocity_frame((-0.05, 0.03, -0.20))
        self.assertEqual(len(frame), 13)
        self.assertEqual(struct.unpack(">hhhh", frame[4:12]), (-50, 30, -200, 0))
        self.assertEqual(frame[-1], sum(frame[:-1]) & 255)

    def test_frame_truncates_like_cpp(self):
        frame = DRIVE.velocity_frame((0.0509, -0.0309, 0.2009))
        self.assertEqual(struct.unpack(">hhhh", frame[4:12]), (50, -30, 200, 0))

    def test_frame_rejects_unsafe_values(self):
        for velocity in [(math.nan, 0, 0), (0, math.inf, 0), (0, 0, -math.inf),
                         (0.301, 0, 0), (0, -0.301, 0), (0, 0, 1.201),
                         (-0.301, 0, 0), (0, 0.301, 0), (0, 0, -1.201), (0, 0)]:
            with self.subTest(velocity=velocity), self.assertRaises(ValueError):
                DRIVE.velocity_frame(velocity)

    def test_parameter_limits(self):
        for index, maximum in enumerate((0.30, 1.20, 0.50)):
            for value in (0, -1, math.nan, math.inf, maximum + 0.001):
                values = [0.05, 0.20, 0.25]
                values[index] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    DRIVE.PulseControl(*values)

    def test_new_speed_limits_are_accepted_and_encoded(self):
        control = DRIVE.PulseControl(0.30, 1.20, 0.50)
        self.assertEqual((control.speed, control.turn_speed), (0.30, 1.20))
        for velocity in ((0.30, -0.30, 1.20), (-0.30, 0.30, -1.20)):
            with self.subTest(velocity=velocity):
                frame = DRIVE.velocity_frame(velocity)
                self.assertEqual(struct.unpack(">hhhh", frame[4:12]),
                                 tuple(int(value * 1000) for value in velocity) + (0,))
                self.assertEqual(frame[-1], sum(frame[:-1]) & 255)

    def test_starts_disarmed(self):
        control = self.control()
        for key in "wsad":
            self.assertIsNone(control.command(key, 1.0))
            self.assertEqual(control.current(1.0), DRIVE.ZERO)

    def test_four_directions_without_lateral(self):
        for key, expected in {"w": (0.05, 0, 0), "s": (-0.05, 0, 0),
                              "a": (0, 0, 0.20), "d": (0, 0, -0.20)}.items():
            with self.subTest(key=key):
                control = self.control()
                control.command(" ", 1.0)
                control.command(key, 1.0)
                self.assertEqual(control.current(1.0), expected)
                self.assertEqual(control.current(1.25), DRIVE.ZERO)

    def test_repeat_does_not_extend_deadline_or_rearm(self):
        control = self.control()
        control.command(" ", 1.0)
        control.command("w", 1.0)
        for moment in (1.1, 1.2, 1.3, 5.0):
            self.assertIsNone(control.command("w", moment))
        self.assertEqual(control.deadline, 1.25)
        self.assertEqual(control.current(5.0), DRIVE.ZERO)
        self.assertFalse(control.ready)

    def test_space_stops_and_allows_next_action(self):
        control = self.control()
        control.command(" ", 1)
        control.command("w", 1)
        control.command(" ", 1.1)
        self.assertEqual(control.current(1.1), DRIVE.ZERO)
        control.command("s", 1.15)
        self.assertEqual(control.current(1.15), (-0.05, 0, 0))

    def test_unknown_keys_do_not_move(self):
        control = self.control()
        control.command(" ", 1)
        for key in (None, "", "j", "l", "x"):
            self.assertIsNone(control.command(key, 1))
            self.assertEqual(control.current(1), DRIVE.ZERO)

    def test_default_and_demo_never_open_serial_or_call_api(self):
        with mock.patch.object(DRIVE, "SerialPort") as serial, \
                mock.patch.object(DRIVE.os, "open") as opening, \
                mock.patch.object(DRIVE.subprocess, "run") as external, \
                mock.patch.object(DRIVE, "Keyboard") as keyboard, \
                contextlib.redirect_stdout(io.StringIO()):
            keyboard.return_value.__enter__.return_value.read.return_value = "q"
            self.assertEqual(DRIVE.main([]), 0)
            self.assertEqual(DRIVE.main(["--demo"]), 0)
            serial.assert_not_called()
            opening.assert_not_called()
            external.assert_not_called()

    def test_check_never_opens_device(self):
        with mock.patch.object(DRIVE, "check_port") as check, \
                mock.patch.object(DRIVE.os, "open") as opening, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(DRIVE.main(["--check"]), 0)
            check.assert_called_once_with("/dev/abotbase")
            opening.assert_not_called()

    def test_live_cancel_never_opens_device(self):
        with mock.patch.object(DRIVE, "check_port"), \
                mock.patch("builtins.input", return_value=""), \
                mock.patch.object(DRIVE, "SerialPort") as serial, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(DRIVE.main(["--live"]), 0)
            serial.assert_not_called()

    def test_invalid_cli_values_fail_without_serial(self):
        with mock.patch.object(DRIVE, "SerialPort") as serial, \
                contextlib.redirect_stderr(io.StringIO()):
            for option in ("--speed", "--turn-speed", "--duration"):
                self.assertEqual(DRIVE.main(["--live", option, "nan"]), 2)
            serial.assert_not_called()

    def test_default_device_not_guessed(self):
        with mock.patch.object(DRIVE.sys, "platform", "linux"), \
                mock.patch.object(DRIVE.os, "stat", side_effect=FileNotFoundError) as lookup:
            with self.assertRaises(FileNotFoundError):
                DRIVE.check_port("/dev/abotbase")
            lookup.assert_called_once_with("/dev/abotbase")

    def test_port_requires_linux_absolute_character_device_permissions_and_no_users(self):
        with mock.patch.object(DRIVE.sys, "platform", "win32"):
            with self.assertRaises(RuntimeError):
                DRIVE.check_port("/dev/abotbase")
        with mock.patch.object(DRIVE.sys, "platform", "linux"):
            with self.assertRaises(ValueError):
                DRIVE.check_port("ttyUSB0")
            with mock.patch.object(DRIVE.os, "stat", return_value=types.SimpleNamespace(st_mode=stat.S_IFREG)):
                with self.assertRaises(ValueError):
                    DRIVE.check_port("/dev/abotbase")
            with mock.patch.object(DRIVE.os, "stat", return_value=types.SimpleNamespace(st_mode=stat.S_IFCHR)):
                with mock.patch.object(DRIVE.os, "access", return_value=False):
                    with self.assertRaises(PermissionError):
                        DRIVE.check_port("/dev/abotbase")
                with mock.patch.object(DRIVE.os, "access", return_value=True), \
                        mock.patch.object(DRIVE, "port_users", return_value={123}):
                    with self.assertRaises(RuntimeError):
                        DRIVE.check_port("/dev/abotbase")

    def test_port_users_empty_and_busy(self):
        with mock.patch.object(DRIVE.shutil, "which", return_value="/usr/bin/fuser"), \
                mock.patch.object(DRIVE.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, "", "")
            self.assertEqual(DRIVE.port_users("/dev/abotbase"), set())
            run.return_value = subprocess.CompletedProcess([], 0, " 123 456 ", "")
            self.assertEqual(DRIVE.port_users("/dev/abotbase"), {123, 456})
            self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_port_users_compatible_command_keeps_occupancy_check(self):
        def compatible_fuser(command, **kwargs):
            if "--" in command:
                return subprocess.CompletedProcess(command, 1, "", "未指定进程\nUsage: fuser")
            return subprocess.CompletedProcess(command, 0, " 123 ", "/dev/shoot:")

        with mock.patch.object(DRIVE.shutil, "which", return_value="/usr/bin/fuser"), \
                mock.patch.object(DRIVE.subprocess, "run", side_effect=compatible_fuser) as run:
            self.assertEqual(DRIVE.port_users("/dev/shoot"), {123})
            run.assert_called_once_with(["/usr/bin/fuser", "/dev/shoot"],
                                        capture_output=True, text=True, timeout=2)

    def test_port_users_rejects_nonabsolute_paths_before_running(self):
        with mock.patch.object(DRIVE.subprocess, "run") as run:
            for value in ("-k", "dev/shoot", "", None, "/dev/\x00shoot"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    DRIVE.port_users(value)
            run.assert_not_called()

    def test_port_users_inconsistent_result_or_invalid_pid_is_rejected(self):
        with mock.patch.object(DRIVE.shutil, "which", return_value="/usr/bin/fuser"), \
                mock.patch.object(DRIVE.subprocess, "run") as run:
            for result in (subprocess.CompletedProcess([], 1, "123", ""),
                           subprocess.CompletedProcess([], 0, "0 -1", "")):
                run.return_value = result
                with self.assertRaises(RuntimeError):
                    DRIVE.port_users("/dev/abotbase")
            run.side_effect = subprocess.TimeoutExpired("fuser", 2)
            with self.assertRaises(subprocess.TimeoutExpired):
                DRIVE.port_users("/dev/abotbase")

    def test_port_users_missing_or_failed_is_rejected(self):
        with mock.patch.object(DRIVE.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError):
                DRIVE.port_users("/dev/abotbase")
        with mock.patch.object(DRIVE.shutil, "which", return_value="fuser"), \
                mock.patch.object(DRIVE.subprocess, "run") as run:
            for result in (subprocess.CompletedProcess([], 2, "", "error"),
                           subprocess.CompletedProcess([], 1, "", "permission denied"),
                           subprocess.CompletedProcess([], 0, "", "")):
                run.return_value = result
                with self.assertRaises(RuntimeError):
                    DRIVE.port_users("/dev/abotbase")

    def test_drive_loop_exit_always_requests_zero(self):
        for key in ("q", "\x1b", "\x03"):
            transport = mock.Mock()
            keyboard = mock.Mock()
            keyboard.read.return_value = key
            DRIVE.drive_loop(self.control(), transport, keyboard, clock=lambda: 1)
            self.assertEqual(transport.send.call_args, mock.call(DRIVE.ZERO))

    def test_keyboard_failure_requests_zero(self):
        transport = mock.Mock()
        keyboard = mock.Mock()
        keyboard.read.side_effect = OSError("keyboard failed")
        with self.assertRaises(OSError):
            DRIVE.drive_loop(self.control(), transport, keyboard, clock=lambda: 1)
        self.assertEqual(transport.send.call_args, mock.call(DRIVE.ZERO))

    def test_keyboard_interrupt_requests_zero(self):
        transport = mock.Mock()
        keyboard = mock.Mock()
        keyboard.read.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            DRIVE.drive_loop(self.control(), transport, keyboard, clock=lambda: 1)
        self.assertEqual(transport.send.call_args, mock.call(DRIVE.ZERO))

    def test_loop_moves_only_after_space_and_zeros_on_exit(self):
        transport = mock.Mock()
        keyboard = mock.Mock()
        keyboard.read.side_effect = ["w", " ", "w", "w", "q"]
        with contextlib.redirect_stdout(io.StringIO()):
            DRIVE.drive_loop(self.control(), transport, keyboard, clock=lambda: 1)
        values = [call.args[0] for call in transport.send.call_args_list]
        self.assertEqual(values[:5], [DRIVE.ZERO] * 5)
        self.assertIn((0.05, 0, 0), values)
        self.assertEqual(values[-1], DRIVE.ZERO)

    def test_movement_print_happens_while_stopped(self):
        transport = mock.Mock()
        keyboard = mock.Mock()
        keyboard.read.side_effect = [" ", "w", "q"]
        def check_print(*args, **kwargs):
            self.assertEqual(transport.send.call_args, mock.call(DRIVE.ZERO))
        with mock.patch("builtins.print", side_effect=check_print):
            DRIVE.drive_loop(self.control(), transport, keyboard, clock=lambda: 1)

    def test_send_handles_partial_write(self):
        serial = DRIVE.SerialPort("/dev/test")
        serial.fd = 99
        with mock.patch.object(DRIVE.select, "select", return_value=([], [99], [])), \
                mock.patch.object(DRIVE.os, "write", side_effect=[3, 10]) as write:
            serial.send(DRIVE.ZERO)
        self.assertEqual(write.call_args_list[1].args[1], DRIVE.velocity_frame(DRIVE.ZERO)[3:])

    def test_send_rejects_write_timeout_or_zero_bytes(self):
        serial = DRIVE.SerialPort("/dev/test")
        serial.fd = 99
        with mock.patch.object(DRIVE.select, "select", return_value=([], [], [])):
            with self.assertRaises(TimeoutError):
                serial.send(DRIVE.ZERO)
        with mock.patch.object(DRIVE.select, "select", return_value=([], [99], [])), \
                mock.patch.object(DRIVE.os, "write", return_value=0):
            with self.assertRaises(OSError):
                serial.send(DRIVE.ZERO)

    def test_serial_exit_closes_even_when_stop_fails(self):
        serial = DRIVE.SerialPort("/dev/test")
        serial.fd = 99
        with mock.patch.object(serial, "send", side_effect=OSError("stop failed")), \
                mock.patch.object(DRIVE.os, "close") as closing:
            with self.assertRaises(OSError):
                serial.__exit__(None, None, None)
            closing.assert_called_once_with(99)
        self.assertIsNone(serial.fd)

    def test_serial_initialization_uses_exclusivity_and_115200_8n1(self):
        fcntl = types.SimpleNamespace(flock=mock.Mock(), ioctl=mock.Mock(), LOCK_EX=2, LOCK_NB=4)
        termios = types.SimpleNamespace(TIOCEXCL=1, CS8=0x30, CREAD=0x80, CLOCAL=0x800,
                                        B115200=4098, VMIN=6, VTIME=5, TCSANOW=0,
                                        tcgetattr=mock.Mock(return_value=[1, 2, 3, 4, 5, 6, [0] * 32]),
                                        tcsetattr=mock.Mock())
        with mock.patch.dict("sys.modules", {"fcntl": fcntl, "termios": termios}), \
                mock.patch.object(DRIVE, "check_port"), \
                mock.patch.object(DRIVE, "port_users", return_value={os.getpid()}), \
                mock.patch.object(DRIVE.os, "open", return_value=99), \
                mock.patch.object(DRIVE.os, "close"), \
                mock.patch.object(DRIVE.os, "O_NOCTTY", 256, create=True), \
                mock.patch.object(DRIVE.os, "O_NONBLOCK", 2048, create=True), \
                mock.patch.object(DRIVE.SerialPort, "send") as sending:
            with DRIVE.SerialPort("/dev/test"):
                settings = termios.tcsetattr.call_args.args[2]
                self.assertEqual(settings[:6], [0, 0, 0x8b0, 0, 4098, 4098])
                fcntl.flock.assert_called_once_with(99, 6)
                fcntl.ioctl.assert_called_once_with(99, 1)
            self.assertEqual(sending.call_args_list, [mock.call(DRIVE.ZERO), mock.call(DRIVE.ZERO)])

    def test_sigterm_uses_interrupt_cleanup_path(self):
        with self.assertRaises(KeyboardInterrupt):
            DRIVE.handle_termination(None, None)

    def test_loop_observes_expiry_without_more_movement_keys(self):
        control = self.control()
        control.command(" ", 0)
        control.command("w", 0)
        transport = mock.Mock()
        keyboard = mock.Mock()
        keyboard.read.side_effect = [None, "q"]
        moments = iter([0.1, 0.3, 0.3, 0.4])
        DRIVE.drive_loop(control, transport, keyboard, clock=lambda: next(moments))
        values = [call.args[0] for call in transport.send.call_args_list]
        self.assertEqual(values, [DRIVE.ZERO, (0.05, 0, 0), DRIVE.ZERO, DRIVE.ZERO, DRIVE.ZERO])

    def test_transport_failure_attempts_zero_without_retrying_movement(self):
        transport = mock.Mock()
        transport.send.side_effect = [None, OSError("write failed"), None]
        with self.assertRaises(OSError):
            DRIVE.drive_loop(self.control(), transport, mock.Mock(), clock=lambda: 1)
        self.assertEqual(transport.send.call_count, 3)
        self.assertEqual(transport.send.call_args, mock.call(DRIVE.ZERO))

    def test_exclusivity_failure_closes_before_any_frame(self):
        fcntl = types.SimpleNamespace(flock=mock.Mock(side_effect=OSError("busy")),
                                     LOCK_EX=2, LOCK_NB=4)
        with mock.patch.dict("sys.modules", {"fcntl": fcntl, "termios": types.SimpleNamespace()}), \
                mock.patch.object(DRIVE, "check_port"), \
                mock.patch.object(DRIVE.os, "open", return_value=99), \
                mock.patch.object(DRIVE.os, "close") as closing, \
                mock.patch.object(DRIVE.os, "O_NOCTTY", 256, create=True), \
                mock.patch.object(DRIVE.os, "O_NONBLOCK", 2048, create=True), \
                mock.patch.object(DRIVE.SerialPort, "send") as sending:
            with self.assertRaises(OSError):
                DRIVE.SerialPort("/dev/test").__enter__()
            closing.assert_called_once_with(99)
            sending.assert_not_called()

    def test_confirmed_live_uses_only_explicit_port(self):
        with mock.patch.object(DRIVE, "check_port"), \
                mock.patch("builtins.input", return_value="MOVE"), \
                mock.patch.object(DRIVE, "Keyboard"), \
                mock.patch.object(DRIVE, "SerialPort") as serial, \
                mock.patch.object(DRIVE, "drive_loop"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(DRIVE.main(["--live", "--port", "/dev/confirmed_base"]), 0)
            serial.assert_called_once_with("/dev/confirmed_base")

    def test_live_failed_precheck_does_not_prompt_or_open(self):
        with mock.patch.object(DRIVE, "check_port", side_effect=RuntimeError("busy")), \
                mock.patch("builtins.input") as prompt, \
                mock.patch.object(DRIVE, "SerialPort") as serial, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(DRIVE.main(["--live"]), 2)
            serial.assert_not_called()
            prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
