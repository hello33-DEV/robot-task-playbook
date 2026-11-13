import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PACKAGE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixed_route_tool", PACKAGE / "tools" / "fixed_route.py")
ROUTE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTE)

from audio_fixtures import audio_fixture


def setUpModule():
    global AUDIO_FIXTURE
    AUDIO_FIXTURE = audio_fixture(ROUTE.card)
    AUDIO_FIXTURE.__enter__()


def tearDownModule():
    AUDIO_FIXTURE.__exit__(None, None, None)


class FixedRouteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.project = Path(self.directory.name)
        self.plan_path = self.project / "plan.json"
        self.plan = ROUTE.demo_plan()
        self.write_plan(self.plan)

    def write_plan(self, plan):
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def invoke(self, args):
        with mock.patch.object(ROUTE, "PROJECT", self.project), \
                mock.patch.object(ROUTE.drive, "SerialPort") as serial, \
                mock.patch.object(ROUTE.drive, "check_port") as check, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = ROUTE.main(["--plan", str(self.plan_path)] + args)
            return code, serial, check

    def result(self):
        paths = list(self.project.glob("runs/固定路线/*/result.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def execute(self, transport, chosen=None, sleep=None):
        virtual = ROUTE.VirtualClock()
        events = []
        with contextlib.redirect_stdout(io.StringIO()):
            ROUTE.execute(self.plan, ROUTE.select_steps(self.plan) if chosen is None else chosen,
                          transport, events, virtual.now, sleep or virtual.sleep)
        return events, virtual

    def test_default_only_previews_and_never_opens_hardware(self):
        code, serial, check = self.invoke([])
        self.assertEqual(code, 0)
        serial.assert_not_called()
        check.assert_not_called()
        self.assertFalse((self.project / "runs").exists())

    def test_competition_template_is_unconfigured_but_can_be_previewed(self):
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/route_template.json")
        self.assertEqual(len(plan["steps"]), 25)
        self.assertEqual(sum(step["seconds"] is None for step in plan["steps"]), 20)
        self.assertEqual(tuple(step["checkpoint"] for step in plan["steps"] if step.get("checkpoint")),
                         ROUTE.ROUTE_CHECKPOINTS)
        self.assertTrue(all(value is False for value in plan["route_checks"].values()))
        with self.assertRaises(ValueError):
            ROUTE.select_steps(plan)

    def test_short_test_is_not_a_distance_promise(self):
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/short_route.json")
        self.assertEqual(plan["steps"][0]["seconds"], 0.25)
        self.assertEqual(ROUTE.velocity(plan, plan["steps"][0]), (0.05, 0.0, 0.0))

    def test_invalid_plan_rejected(self):
        variants = []
        for key, value in (("schema_version", True), ("schema_version", 2),
                           ("speed_m_s", True), ("speed_m_s", 0), ("speed_m_s", "0.05"),
                           ("speed_m_s", 0.301), ("speed_m_s", float("nan")),
                           ("turn_speed_rad_s", float("inf")), ("turn_speed_rad_s", 1.201),
                           ("initial_heading", []), ("initial_heading", ""),
                           ("steps", []), ("steps", [None])):
            candidate = copy.deepcopy(self.plan)
            candidate[key] = value
            variants.append(candidate)
        for key, value in (("name", ""), ("action", "not_an_action"), ("action", []),
                           ("seconds", -1), ("seconds", True), ("seconds", 60.01),
                           ("seconds", float("nan"))):
            candidate = copy.deepcopy(self.plan)
            candidate["steps"][0][key] = value
            variants.append(candidate)
        for candidate in variants + [[], None]:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                self.write_plan(candidate)
                ROUTE.load_plan(self.plan_path)

    def test_invalid_step_and_unconfigured_selection_rejected(self):
        for index in (0, -1, 5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                ROUTE.select_steps(self.plan, index)
        for key in ("action", "seconds"):
            candidate = copy.deepcopy(self.plan)
            candidate["steps"][0][key] = None
            with self.assertRaises(ValueError):
                ROUTE.select_steps(candidate, 1)
            self.assertEqual(ROUTE.select_steps(candidate, 2)[0][0], 2)

    def test_total_budget_rejected(self):
        self.plan["steps"] *= 25
        for step in self.plan["steps"]:
            step["seconds"] = 10.0
        with self.assertRaises(ValueError):
            ROUTE.select_steps(self.plan)

    def test_velocity_supports_forward_backward_turns_and_strafe(self):
        expected = [(0.05, 0.0, 0.0), (-0.05, 0.0, 0.0),
                    (0.0, 0.0, 0.20), (0.0, 0.0, -0.20)]
        self.assertEqual([ROUTE.velocity(self.plan, step) for step in self.plan["steps"]], expected)

    def test_start_pose_test_derives_open_loop_times_and_right_strafe(self):
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        prepared = ROUTE.prepare_start_pose_test(plan)
        self.assertEqual([step["action"] for step in prepared["steps"]],
                         ["strafe_right", "forward", "left"])
        self.assertAlmostEqual(prepared["steps"][0]["seconds"], 0.36 / 0.30)
        self.assertAlmostEqual(prepared["steps"][1]["target_distance_m"], 0.39)
        self.assertAlmostEqual(prepared["steps"][1]["seconds"], 0.39 / 0.30)
        self.assertEqual(prepared["steps"][2]["target_angle_deg"], 96)
        self.assertAlmostEqual(prepared["steps"][2]["seconds"], math.radians(96) / 1.2)
        self.assertEqual(ROUTE.velocity(prepared, prepared["steps"][1]), (0.30, 0.0, 0.0))
        self.assertEqual(ROUTE.velocity(prepared, prepared["steps"][2]), (0.0, 0.0, 1.2))
        self.assertEqual(ROUTE.velocity(prepared, prepared["steps"][0]), (0.0, -0.30, 0.0))

    def test_start_pose_test_requires_exact_specialized_plan(self):
        with self.assertRaises(ValueError):
            ROUTE.prepare_start_pose_test(self.plan)
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        plan["steps"].pop()
        with self.assertRaises(ValueError):
            ROUTE.prepare_start_pose_test(plan)

    def test_start_pose_test_simulation_runs_only_three_steps(self):
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        self.write_plan(plan)
        virtual = ROUTE.VirtualClock()
        with mock.patch.object(ROUTE, "PROJECT", self.project), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(ROUTE.drive, "SerialPort") as serial:
            code = ROUTE.main(["--simulate", "--start-pose-test", "--plan", str(self.plan_path)])
        self.assertEqual(code, 0)
        serial.assert_not_called()
        result = self.result()
        self.assertEqual(result["mode"], "start_pose_test_simulation")
        self.assertEqual([event["action"] for event in result["events"]],
                         ["strafe_right", "forward", "left"])
        self.assertTrue(result["software_sequence_complete"])

    def test_start_pose_test_live_requires_run_test_and_does_not_open_on_cancel(self):
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        self.write_plan(plan)
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="MOVE"):
            code, serial, check = self.invoke(["--live", "--start-pose-test", "--plan", str(self.plan_path)])
        self.assertEqual(code, 0)
        serial.assert_not_called()
        check.assert_called_once_with("/dev/abotbase")
        self.assertEqual(self.result()["status"], "cancelled")

    def test_start_pose_test_live_executes_all_three_and_finishes_zero(self):
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        self.write_plan(plan)
        virtual = ROUTE.VirtualClock()
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="RUN TEST"), \
                mock.patch.object(ROUTE.time, "monotonic", side_effect=virtual.now), \
                mock.patch.object(ROUTE.time, "sleep", side_effect=virtual.sleep):
            code, serial, check = self.invoke(["--live", "--start-pose-test", "--plan", str(self.plan_path)])
        self.assertEqual(code, 0)
        check.assert_called_once_with("/dev/abotbase")
        serial.assert_called_once_with("/dev/abotbase")
        transport = serial.return_value.__enter__.return_value
        self.assertEqual(transport.send.call_args, mock.call(ROUTE.drive.ZERO))
        result = self.result()
        self.assertEqual(result["mode"], "live_start_pose_test")
        self.assertEqual(len(result["events"]), 3)
        self.assertTrue(result["software_sequence_complete"])
        self.assertFalse(result["physical_motion_confirmed"])
        self.assertAlmostEqual(virtual.value, 3.25 + math.radians(96) / 1.2)

    def combined_args(self, mode="--live"):
        self.write_plan(ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json"))
        return [mode, "--start-pose-test", "--recognize-after-stop", "--camera", "0"]

    def test_combined_simulation_never_loads_key_captures_requests_or_plays(self):
        with mock.patch.object(ROUTE.card, "prepare_recognition") as prepare, \
                mock.patch.object(ROUTE.card, "run_auto") as recognize:
            code, serial, check = self.invoke(self.combined_args("--simulate"))
        self.assertEqual(code, 0)
        prepare.assert_not_called()
        recognize.assert_not_called()
        serial.assert_not_called()
        check.assert_not_called()
        record = self.result()
        self.assertEqual(record["recognition_status"], "simulation_skipped")
        self.assertEqual(record["return_heading_status"], "simulation_skipped")
        self.assertFalse(record["workflow_software_complete"])
        self.assertEqual(record["return_events"], [])
        self.assertEqual(record["requested_return_step"]["action"], "right")
        self.assertEqual(record["requested_return_step"]["target_angle_deg"], 96)
        self.assertNotIn("recognition_result", record)
        self.assertTrue(record["software_sequence_complete"])

    def test_combined_requires_explicit_valid_mode_camera_and_timeout(self):
        invalid = [["--live", "--recognize-after-stop", "--camera", "0"],
                   ["--simulate", "--start-pose-test", "--recognize-after-stop"],
                   ["--simulate", "--start-pose-test", "--recognize-after-stop", "--camera", " "],
                   ["--simulate", "--camera", "0"]]
        for timeout in ("nan", "inf", "0", "-1", "121"):
            invalid.append(["--simulate", "--start-pose-test", "--recognize-after-stop",
                            "--camera", "0", "--vision-timeout", timeout])
        with mock.patch.object(ROUTE.card, "prepare_recognition") as prepare, \
                mock.patch.object(ROUTE.card, "run_auto") as recognize:
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    code, serial, check = self.invoke(arguments)
                    self.assertEqual(code, 2)
                    serial.assert_not_called()
                    check.assert_not_called()
        prepare.assert_not_called()
        recognize.assert_not_called()

    def test_combined_cancel_does_not_move_or_recognize(self):
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="cancel"), \
                mock.patch.object(ROUTE.card, "prepare_recognition"), \
                mock.patch.object(ROUTE.card, "run_auto") as recognize:
            code, serial, check = self.invoke(self.combined_args())
        self.assertEqual(code, 0)
        serial.assert_not_called()
        recognize.assert_not_called()
        self.assertEqual(self.result()["recognition_status"], "not_started")

    def test_combined_preflight_failure_blocks_before_serial_open(self):
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(ROUTE.card, "prepare_recognition", side_effect=RuntimeError("missing resources")), \
                mock.patch.object(ROUTE.card, "run_auto") as recognize:
            code, serial, check = self.invoke(self.combined_args())
        self.assertEqual(code, 2)
        serial.assert_not_called()
        recognize.assert_not_called()

    def test_combined_motion_failure_never_recognizes(self):
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="RUN TEST"), \
                mock.patch.object(ROUTE.card, "prepare_recognition"), \
                mock.patch.object(ROUTE, "execute", side_effect=RuntimeError("motion failure")), \
                mock.patch.object(ROUTE.card, "run_auto") as recognize:
            code, serial, check = self.invoke(self.combined_args())
        self.assertEqual(code, 2)
        recognize.assert_not_called()
        self.assertFalse(self.result()["software_sequence_complete"])
        serial.return_value.__exit__.assert_called_once()

    def test_combined_close_failure_never_recognizes(self):
        with mock.patch.object(ROUTE, "PROJECT", self.project), \
                mock.patch.object(ROUTE.drive, "SerialPort") as serial, \
                mock.patch.object(ROUTE.drive, "check_port"), \
                mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="RUN TEST"), \
                mock.patch.object(ROUTE.card, "prepare_recognition"), \
                mock.patch.object(ROUTE, "execute"), \
                mock.patch.object(ROUTE.card, "run_auto") as recognize, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            serial.return_value.__exit__.side_effect = OSError("zero on close failed")
            code = ROUTE.main(["--plan", str(self.plan_path)] + self.combined_args())
        self.assertEqual(code, 2)
        recognize.assert_not_called()
        self.assertFalse(self.result()["software_sequence_complete"])

    def test_combined_real_software_pipeline_only_captures_after_motion_and_close(self):
        virtual = ROUTE.VirtualClock()
        answer = {"math_ans": 2, "fruit_code": 6}

        def capture(camera, output, **options):
            serial.return_value.__exit__.assert_called_once()
            transport = serial.return_value.__enter__.return_value
            self.assertEqual(transport.send.call_args, mock.call(ROUTE.drive.ZERO))
            self.assertAlmostEqual(virtual.value, 3.25 + math.radians(96) / 1.2)
            record = self.result()
            self.assertTrue(all(event["software_step_complete"] for event in record["events"]))
            self.assertEqual(record["recognition_status"], "running")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(output.name.encode("utf-8"))
            return output

        with mock.patch.object(ROUTE, "PROJECT", self.project), \
                mock.patch.object(ROUTE.drive, "SerialPort") as serial, \
                mock.patch.object(ROUTE.drive, "check_port"), \
                mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="RUN TEST"), \
                mock.patch.object(ROUTE.time, "monotonic", virtual.now), \
                mock.patch.object(ROUTE.time, "sleep", virtual.sleep), \
                mock.patch.object(ROUTE.card, "prepare_recognition", return_value="offline-only-test-key"), \
                mock.patch.object(ROUTE.card, "capture_card", side_effect=capture) as captures, \
                mock.patch.object(ROUTE.card.VISION, "request_vision",
                                  side_effect=[{"recognized": False}, dict(answer, reliable=True)]) as requests, \
                mock.patch.object(ROUTE.card, "play_audio") as play, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = ROUTE.main(["--plan", str(self.plan_path)] + self.combined_args())
        self.assertEqual(code, 0)
        self.assertEqual(captures.call_count, 2)
        self.assertEqual(requests.call_count, 2)
        self.assertTrue(all(call.kwargs["timeout"] == 30 for call in requests.call_args_list))
        play.assert_called_once()
        self.assertEqual(play.call_args.args[0].name, "answer_2_6.wav")
        record = self.result()
        self.assertEqual(record["recognition_result"], answer)
        self.assertEqual(record["status"], "motion_recognition_and_return_software_complete")
        self.assertEqual(record["return_heading_status"], "software_complete")
        self.assertTrue(record["workflow_software_complete"])
        self.assertEqual(serial.call_count, 2)
        self.assertEqual(serial.return_value.__exit__.call_count, 2)
        self.assertAlmostEqual(virtual.value, 3.5 + 2 * math.radians(96) / 1.2)
        self.assertEqual(record["return_events"][0]["action"], "right")
        self.assertEqual(record["return_events"][0]["step"], 4)
        self.assertTrue(record["return_events"][0]["software_step_complete"])
        commands = [call.args[0] for call in serial.return_value.__enter__.return_value.send.call_args_list]
        self.assertIn((0.0, 0.0, -1.2), commands)
        self.assertEqual(commands[-1], ROUTE.drive.ZERO)
        self.assertFalse(any(value for key, value in record.items() if key.startswith("physical_")))
        detail = json.loads((Path(record["recognition_session"]) / "result.json").read_text(encoding="utf-8"))
        self.assertFalse(detail["confirmation_required"])
        self.assertFalse(detail["confirmation_satisfied"])
        self.assertEqual(detail["successful_attempt_number"], 2)
        self.assertEqual(detail["answer_source"], "model_declared_reliable_single_shot")
        self.assertEqual(detail["text"], "识别到旋转靶2，识别到移动靶6。")
        self.assertNotEqual(detail["attempts"][0]["image_sha256"], detail["attempts"][1]["image_sha256"])
        self.assertNotIn("offline-only-test-key", json.dumps([record, detail]))

    def test_first_shot_broadcast_finishes_before_clockwise_return(self):
        virtual = ROUTE.VirtualClock()
        answer = {"math_ans": 4, "fruit_code": 8}
        with mock.patch.object(ROUTE, "PROJECT", self.project), \
                mock.patch.object(ROUTE.drive, "SerialPort") as serial, \
                mock.patch.object(ROUTE.drive, "check_port") as check, \
                mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="RUN TEST"), \
                mock.patch.object(ROUTE.time, "monotonic", virtual.now), \
                mock.patch.object(ROUTE.time, "sleep", virtual.sleep), \
                mock.patch.object(ROUTE.card, "prepare_recognition", return_value="offline-only-key"), \
                mock.patch.object(ROUTE.card.VISION, "request_vision", return_value=dict(answer, reliable=True)) as request, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):

            def capture(camera, output, **options):
                serial.return_value.__exit__.assert_called_once()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"first-new-frame")
                return output

            def play(audio):
                serial.assert_called_once()
                serial.return_value.__exit__.assert_called_once()
                self.assertEqual(check.call_count, 1)
                self.assertEqual(self.result()["return_heading_status"], "not_started")
                commands = serial.return_value.__enter__.return_value.send.call_args_list
                self.assertNotIn(mock.call((0.0, 0.0, -1.2)), commands)
                self.assertEqual(audio.name, "answer_4_8.wav")

            with mock.patch.object(ROUTE.card, "capture_card", side_effect=capture) as captures, \
                    mock.patch.object(ROUTE.card, "play_audio", side_effect=play) as playback:
                code = ROUTE.main(["--plan", str(self.plan_path)] + self.combined_args())
        self.assertEqual(code, 0)
        captures.assert_called_once()
        request.assert_called_once()
        playback.assert_called_once()
        self.assertEqual(check.call_args_list, [mock.call("/dev/abotbase"), mock.call("/dev/abotbase")])
        self.assertEqual(serial.call_count, 2)
        record = self.result()
        self.assertTrue(record["workflow_software_complete"])
        self.assertEqual(record["recognition_result"], answer)
        self.assertEqual(record["return_heading_status"], "software_complete")
        self.assertEqual(record["return_events"][0]["action"], "right")
        self.assertAlmostEqual(record["return_events"][0]["requested_seconds"], math.radians(96) / 1.2)
        self.assertFalse(record["physical_heading_confirmed"])
        self.assertFalse(record["physical_audio_confirmed"])
        self.assertAlmostEqual(virtual.value, 3.5 + 2 * math.radians(96) / 1.2)

    def test_return_failures_never_mark_workflow_complete_or_repeat_recognition(self):
        for point in ("busy", "open", "execute", "close", "interrupt", "log"):
            with self.subTest(point=point):
                self.project = Path(self.directory.name) / point
                self.project.mkdir()
                self.plan_path = self.project / "plan.json"
                with mock.patch.object(ROUTE, "PROJECT", self.project), \
                        mock.patch.object(ROUTE.drive, "SerialPort") as serial, \
                        mock.patch.object(ROUTE.drive, "check_port") as check, \
                        mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                        mock.patch("builtins.input", return_value="RUN TEST"), \
                        mock.patch.object(ROUTE.card, "prepare_recognition"), \
                        mock.patch.object(ROUTE.card, "run_auto", return_value={"math_ans": 2, "fruit_code": 6}) as recognize, \
                        mock.patch.object(ROUTE, "execute") as execute, \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    failure = KeyboardInterrupt() if point == "interrupt" else OSError("return " + point)
                    if point == "busy":
                        check.side_effect = [None, failure]
                    elif point == "open":
                        serial.return_value.__enter__.side_effect = [mock.Mock(), failure]
                    elif point in ("execute", "interrupt"):
                        execute.side_effect = [None, failure]
                    elif point == "close":
                        serial.return_value.__exit__.side_effect = [None, failure]
                    save = ROUTE.save_result

                    def save_or_fail(path, record):
                        if point == "log" and record["return_heading_status"] == "running":
                            raise failure
                        save(path, record)

                    with mock.patch.object(ROUTE, "save_result", side_effect=save_or_fail):
                        code = ROUTE.main(["--plan", str(self.plan_path)] + self.combined_args())
                self.assertEqual(code, 130 if point == "interrupt" else 2)
                recognize.assert_called_once()
                self.assertEqual(serial.call_count, 1 if point in ("busy", "log") else 2)
                record = self.result()
                self.assertTrue(record["software_sequence_complete"])
                self.assertFalse(record["workflow_software_complete"])
                self.assertEqual(record["recognition_status"], "recognition_and_playback_software_complete")
                self.assertEqual(record["return_heading_status"], "interrupted" if point == "interrupt" else "failed")
                self.assertFalse(record["physical_heading_confirmed"])

    def test_shoot_extension_requires_link_and_distinct_ports(self):
        for extra in (["--simulate", "--shoot-and-retreat"],
                      self.combined_args("--simulate") + ["--shoot-and-retreat", "--port", ROUTE.shoot.PORT]):
            with self.subTest(args=extra), mock.patch.object(ROUTE.shoot, "check_live") as checking, \
                    mock.patch.object(ROUTE.shoot, "open_serial") as opening:
                code, serial, check = self.invoke(extra)
                self.assertEqual(code, 2)
                serial.assert_not_called()
                check.assert_not_called()
                checking.assert_not_called()
                opening.assert_not_called()

    def test_shoot_extension_requires_new_fire_confirmation(self):
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="RUN TEST") as prompt, \
                mock.patch.object(ROUTE.card, "prepare_recognition"), \
                mock.patch.object(ROUTE.shoot, "check_live") as checking, \
                mock.patch.object(ROUTE.shoot, "open_serial") as opening:
            code, serial, check = self.invoke(self.combined_args() + ["--shoot-and-retreat"])
        self.assertEqual(code, 0)
        self.assertIn("RUN TEST FIRE", prompt.call_args[0][0])
        self.assertEqual(self.result()["status"], "cancelled")
        checking.assert_called_once()
        serial.assert_not_called()
        opening.assert_not_called()

    def test_shoot_extension_checks_launcher_before_any_motion(self):
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input") as prompt, \
                mock.patch.object(ROUTE.card, "run_auto") as recognize, \
                mock.patch.object(ROUTE.shoot, "check_live", side_effect=RuntimeError("missing launcher")), \
                mock.patch.object(ROUTE.shoot, "open_serial") as opening:
            code, serial, check = self.invoke(self.combined_args() + ["--shoot-and-retreat"])
        self.assertEqual(code, 2)
        prompt.assert_not_called()
        serial.assert_not_called()
        recognize.assert_not_called()
        opening.assert_not_called()

    def test_yes_runs_complete_workflow_without_prompt(self):
        actions = []
        virtual = ROUTE.VirtualClock()

        def motion(plan, steps, *args):
            actions.extend(step["action"] for ordinal, step in steps)

        def pulse(transport, record, save):
            record.update(request_accepted=True, software_execution_complete=True,
                          pulse_result="completed", fault=None, worker_still_running=False)
            save()

        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", side_effect=AssertionError("must not prompt")) as prompt, \
                mock.patch.object(ROUTE, "execute", side_effect=motion), \
                mock.patch.object(ROUTE.time, "sleep", virtual.sleep), \
                mock.patch.object(ROUTE.card, "prepare_recognition") as prepare, \
                mock.patch.object(ROUTE.card, "run_auto", return_value={"math_ans": 2, "fruit_code": 6}) as recognize, \
                mock.patch.object(ROUTE.shoot, "check_live") as checking, \
                mock.patch.object(ROUTE.shoot, "open_serial") as opening, \
                mock.patch.object(ROUTE.shoot, "execute_once", side_effect=pulse) as launch:
            code, serial, check = self.invoke(self.combined_args() + ["--shoot-and-retreat", "--yes"])
        self.assertEqual(code, 0)
        self.assertEqual(actions, ["strafe_right", "forward", "left", "right", "forward", "backward"])
        prompt.assert_not_called()
        prepare.assert_called_once_with("0")
        checking.assert_called_once()
        recognize.assert_called_once()
        self.assertEqual(opening.call_count, 2)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(serial.call_count, 4)
        self.assertTrue(self.result()["workflow_software_complete"])
        self.assertEqual(self.result()["confirmation_mode"], "explicit_command_flag")
        self.assertFalse(self.result()["launch_result"]["actual_discharge_confirmed"])

    def test_yes_cannot_bypass_launcher_preflight_or_terminal(self):
        for terminal, failure in ((False, None), (True, RuntimeError("missing launcher"))):
            with self.subTest(terminal=terminal), \
                    mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=terminal), \
                    mock.patch("builtins.input") as prompt, \
                    mock.patch.object(ROUTE.shoot, "check_live", side_effect=failure), \
                    mock.patch.object(ROUTE.card, "run_auto") as recognize, \
                    mock.patch.object(ROUTE.shoot, "open_serial") as opening:
                code, serial, check = self.invoke(self.combined_args() + ["--shoot-and-retreat", "--yes"])
            self.assertEqual(code, 2)
            for callback in (prompt, serial, recognize, opening):
                callback.assert_not_called()

    def test_yes_rejected_outside_explicit_live_complete_workflow(self):
        for args in (["--yes"], ["--check", "--yes"], ["--live", "--step", "1", "--yes"],
                     self.combined_args() + ["--yes"],
                     self.combined_args("--simulate") + ["--shoot-and-retreat", "--yes"]):
            with self.subTest(args=args), mock.patch("builtins.input") as prompt:
                code, serial, check = self.invoke(args)
            self.assertEqual(code, 2)
            for callback in (prompt, serial, check):
                callback.assert_not_called()

    def test_shoot_extension_simulation_skips_all_conditional_devices(self):
        with mock.patch.object(ROUTE.card, "prepare_recognition") as prepare, \
                mock.patch.object(ROUTE.card, "run_auto") as recognize, \
                mock.patch.object(ROUTE.shoot, "check_live") as checking, \
                mock.patch.object(ROUTE.shoot, "open_serial") as opening, \
                mock.patch.object(ROUTE.shoot, "execute_once") as launch:
            code, serial, check = self.invoke(self.combined_args("--simulate") + ["--shoot-and-retreat"])
        self.assertEqual(code, 0)
        for callback in (prepare, recognize, checking, opening, launch, serial, check):
            callback.assert_not_called()
        record = self.result()
        steps = record["requested_post_return_steps"]
        self.assertEqual([step["action"] for step in steps], ["forward", "backward"])
        self.assertEqual([step["target_distance_m"] for step in steps], [0.72, 0.86])
        self.assertAlmostEqual(steps[0]["seconds"], 0.72 / 0.3)
        self.assertAlmostEqual(steps[1]["seconds"], 0.86 / 0.3)
        for stage in ("approach_status", "launch_status", "retreat_status"):
            self.assertEqual(record[stage], "simulation_skipped")
        self.assertFalse(record["workflow_software_complete"])
        self.assertFalse(record["launch_result"]["software_execution_complete"])
        self.assertFalse(record["launch_result"]["actual_discharge_confirmed"])

    def test_shoot_distances_and_combined_budget_rejected_before_devices(self):
        base = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        variants = []
        for key in ("approach_distance_m", "retreat_distance_m"):
            for value in (None, True, 0, -0.1, 2.01, "0.60", float("nan")):
                plan = copy.deepcopy(base)
                plan["after_return"][key] = value
                variants.append(plan)
        plan = copy.deepcopy(base)
        plan.pop("after_return")
        variants.append(plan)
        for speed in (0.007, 0.017):
            plan = copy.deepcopy(base)
            plan["speed_m_s"] = speed
            variants.append(plan)
        for plan in variants:
            with self.subTest(plan=plan), mock.patch.object(ROUTE.shoot, "check_live") as checking, \
                    mock.patch.object(ROUTE.shoot, "open_serial") as opening:
                self.write_plan(plan)
                code, serial, check = self.invoke(["--live", "--start-pose-test", "--recognize-after-stop",
                                                  "--camera", "0", "--shoot-and-retreat"])
                self.assertEqual(code, 2)
                for callback in (serial, check, checking, opening):
                    callback.assert_not_called()

    def test_complete_shoot_route_uses_two_pulses_between_stopped_motion(self):
        virtual = ROUTE.VirtualClock()
        actions = []
        active = {"base": False}
        original_execute = ROUTE.execute
        launcher = mock.Mock()
        frames = []

        @contextlib.contextmanager
        def base_port(port):
            self.assertFalse(active["base"])
            active["base"] = True
            try:
                yield mock.Mock()
            finally:
                active["base"] = False

        def write_frame(frame):
            self.assertFalse(active["base"])
            self.assertEqual(self.result()["approach_status"], "software_complete")
            frames.append(frame)
            return len(frame)

        def motion(plan, steps, transport, events, clock, sleep):
            actions.extend((ordinal, step["action"]) for ordinal, step in steps)
            if steps[0][0] == 6:
                self.assertGreaterEqual(launcher.close.call_count, 1)
                self.assertEqual(self.result()["launch_status"], "pulse_software_complete")
                self.assertFalse(self.result()["workflow_software_complete"])
            return original_execute(plan, steps, transport, events, clock, sleep)

        def recognize(*args):
            self.assertFalse(active["base"])
            self.assertEqual(actions, [(1, "strafe_right"), (2, "forward"), (3, "left")])
            return {"math_ans": 2, "fruit_code": 6}

        launcher.write.side_effect = write_frame
        with mock.patch.object(ROUTE, "PROJECT", self.project), \
                mock.patch.object(ROUTE.drive, "SerialPort", side_effect=base_port) as serial, \
                mock.patch.object(ROUTE.drive, "check_port"), \
                mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="RUN TEST FIRE"), \
                mock.patch.object(ROUTE.time, "monotonic", virtual.now), \
                mock.patch.object(ROUTE.time, "sleep", virtual.sleep), \
                mock.patch.object(ROUTE, "execute", side_effect=motion), \
                mock.patch.object(ROUTE.card, "prepare_recognition"), \
                mock.patch.object(ROUTE.card, "run_auto", side_effect=recognize) as recognition, \
                mock.patch.object(ROUTE.shoot, "check_live"), \
                mock.patch.object(ROUTE.shoot, "open_serial", return_value=launcher) as opening, \
                mock.patch.object(ROUTE.shoot, "execute_once", wraps=ROUTE.shoot.execute_once) as launch, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = ROUTE.main(["--plan", str(self.plan_path)] + self.combined_args() + ["--shoot-and-retreat"])
        self.assertEqual(code, 0)
        self.assertEqual(actions, [(1, "strafe_right"), (2, "forward"), (3, "left"),
                                   (4, "right"), (5, "forward"), (6, "backward")])
        self.assertEqual(serial.call_count, 4)
        recognition.assert_called_once()
        self.assertEqual(opening.call_count, 2)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(frames.count(bytes.fromhex(ROUTE.shoot.CONFIG["start_hex"])), 2)
        self.assertEqual(frames[-1], bytes.fromhex(ROUTE.shoot.CONFIG["stop_hex"]))
        record = self.result()
        self.assertTrue(record["workflow_software_complete"])
        self.assertEqual(record["retreat_status"], "software_complete")
        self.assertEqual(record["launch_result"]["pulse_result"], "completed")
        self.assertTrue(record["launch_result"]["request_accepted"])
        self.assertEqual(record["launch_result"]["shots_completed"], 2)
        self.assertEqual([shot["shot_number"] for shot in record["launch_result"]["shot_records"]], [1, 2])
        for key in ("device_acknowledged", "actual_discharge_confirmed", "physical_stop_confirmed", "hit_confirmed"):
            self.assertFalse(record["launch_result"][key])
        self.assertAlmostEqual(virtual.value, (0.36 + 0.39 + 0.72 + 0.86) / 0.3
                                + 2 * math.radians(96) / 1.2 + 6 * 0.25 + 0.90)

    def test_continuation_simulation_records_six_conditional_motion_steps_without_devices(self):
        self.write_plan(ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json"))
        with mock.patch.object(ROUTE.card, "run_auto") as recognize, \
                mock.patch.object(ROUTE.shoot, "open_serial") as launcher:
            code, serial, check = self.invoke(self.combined_args("--simulate") +
                                              ["--shoot-and-retreat", "--continue-to-3"])
        self.assertEqual(code, 0)
        recognize.assert_not_called()
        launcher.assert_not_called()
        serial.assert_not_called()
        check.assert_not_called()
        record = self.result()
        self.assertEqual(record["continuation_status"], "simulation_skipped")
        self.assertEqual({key: shot["status"] for key, shot in record["target_shots"].items()},
                         {"2": "simulation_skipped", "3": "simulation_skipped"})
        self.assertEqual([step["action"] for step in record["requested_continuation_steps"]],
                         ["strafe_right", "forward", "backward",
                          "strafe_right", "forward", "backward", "strafe_right"])
        configured = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        names = ("to_second_strafe_m", "to_second_approach_m", "from_second_retreat_m",
                 "to_third_strafe_m", "to_third_approach_m", "from_third_retreat_m",
                 "return_to_finish_strafe_m")
        self.assertEqual([step["target_distance_m"] for step in record["requested_continuation_steps"]],
                         [configured["after_retreat"][name] for name in names])
        self.assertEqual(record["requested_post_return_steps"][1]["target_distance_m"],
                         configured["after_return"]["retreat_distance_m"])
        self.assertFalse(record["workflow_software_complete"])
        self.assertFalse(record["launch_result"]["hit_confirmed"])

    def test_continuation_requires_valid_sequence_and_distances_but_skips_route_checks(self):
        base = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        simulation_args = self.combined_args("--simulate")
        self.write_plan(base)
        with mock.patch.object(ROUTE.card, "run_auto") as recognize:
            code, serial, check = self.invoke(simulation_args + ["--continue-to-3"])
        self.assertEqual(code, 2)
        recognize.assert_not_called()
        serial.assert_not_called()
        check.assert_not_called()
        for value in (None, True, 0, -0.1, 2.01, "1.24", float("nan")):
            candidate = copy.deepcopy(base)
            candidate["after_retreat"]["to_second_strafe_m"] = value
            self.write_plan(candidate)
            code, serial, check = self.invoke(simulation_args +
                                              ["--shoot-and-retreat", "--continue-to-3"])
            self.assertEqual(code, 2)
            serial.assert_not_called()
            check.assert_not_called()
        base["route_checks"] = {key: False for key in ROUTE.ROUTE_CHECKS}
        live_args = self.combined_args()
        self.write_plan(base)
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=False), \
                mock.patch.object(ROUTE, "require_route_checks") as route_checks, \
                mock.patch.object(ROUTE.card, "prepare_recognition") as prepare, \
                mock.patch.object(ROUTE.shoot, "check_live") as launcher:
            code, serial, check = self.invoke(live_args +
                                              ["--shoot-and-retreat", "--continue-to-3", "--yes"])
        self.assertEqual(code, 2)
        route_checks.assert_not_called()
        prepare.assert_not_called()
        launcher.assert_not_called()
        serial.assert_not_called()
        check.assert_not_called()

    def test_continuation_stops_on_second_position_motion_failure_before_shooting(self):
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        plan["route_checks"] = {key: True for key in ROUTE.ROUTE_CHECKS}
        live_args = self.combined_args()
        self.write_plan(plan)
        actions = []

        def motion(plan, steps, *args):
            for ordinal, step in steps:
                actions.append(ordinal)
                if ordinal == 8:
                    raise OSError("2号前进失败")

        def pulse(transport, record, save):
            record.update(request_accepted=True, software_execution_complete=True,
                          pulse_result="completed", fault=None, worker_still_running=False)
            save()

        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(ROUTE, "execute", side_effect=motion), \
                mock.patch.object(ROUTE.time, "sleep"), \
                mock.patch.object(ROUTE.card, "prepare_recognition"), \
                mock.patch.object(ROUTE.card, "run_auto", return_value={"math_ans": 2, "fruit_code": 6}), \
                mock.patch.object(ROUTE.shoot, "check_live"), \
                mock.patch.object(ROUTE.shoot, "open_serial") as launcher, \
                mock.patch.object(ROUTE.shoot, "execute_once", side_effect=pulse) as launch:
            code, serial, check = self.invoke(live_args +
                                              ["--shoot-and-retreat", "--continue-to-3", "--yes"])
        self.assertEqual(code, 2)
        self.assertEqual(actions, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(serial.call_count, 6)
        self.assertEqual(launcher.call_count, 2)
        self.assertEqual(launch.call_count, 2)
        record = self.result()
        self.assertEqual(record["continuation_status"], "failed")
        self.assertFalse(record["workflow_software_complete"])
        self.assertFalse(record["launch_result"]["hit_confirmed"])

    def test_continuation_fake_transport_completes_with_false_route_check_markers(self):
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        plan["route_checks"] = {key: False for key in ROUTE.ROUTE_CHECKS}
        live_args = self.combined_args()
        self.write_plan(plan)
        virtual = ROUTE.VirtualClock()
        original_execute = ROUTE.execute
        motions = []
        launch = mock.Mock()

        def motion(plan, steps, transport, events, clock, sleep):
            motions.extend(ordinal for ordinal, step in steps)
            if steps[0][0] == 7:
                self.assertTrue(self.result()["retreat_status"] == "software_complete")
                self.assertFalse(self.result()["workflow_software_complete"])
                self.assertEqual(launch.call_count, 2)
            original_execute(plan, steps, transport, events, clock, sleep)

        def pulse(transport, record, save):
            launch()
            record.update(request_accepted=True, software_execution_complete=True,
                          pulse_result="completed", fault=None, worker_still_running=False)
            save()

        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(ROUTE.time, "monotonic", virtual.now), \
                mock.patch.object(ROUTE.time, "sleep", virtual.sleep), \
                mock.patch.object(ROUTE, "execute", side_effect=motion), \
                mock.patch.object(ROUTE.card, "prepare_recognition"), \
                mock.patch.object(ROUTE.card, "run_auto", return_value={"math_ans": 2, "fruit_code": 6}), \
                mock.patch.object(ROUTE.shoot, "check_live"), \
                mock.patch.object(ROUTE.shoot, "open_serial") as opening, \
                mock.patch.object(ROUTE.shoot, "execute_once", side_effect=pulse):
            code, serial, check = self.invoke(live_args +
                                              ["--shoot-and-retreat", "--continue-to-3", "--yes"])
        self.assertEqual(code, 0)
        self.assertEqual(motions, list(range(1, 14)))
        self.assertEqual(serial.call_count, 11)
        self.assertEqual(opening.call_count, 6)
        self.assertEqual(launch.call_count, 6)
        record = self.result()
        self.assertEqual(record["continuation_status"], "software_complete")
        self.assertEqual(record["target_shots"]["2"]["pulse_result"], "completed")
        self.assertEqual(record["target_shots"]["3"]["pulse_result"], "completed")
        self.assertEqual(record["launch_result"]["shots_completed"], 2)
        for target in ("2", "3"):
            self.assertEqual(record["target_shots"][target]["shots_completed"], 2)
            self.assertEqual([shot["shot_number"] for shot in record["target_shots"][target]["shot_records"]],
                             [1, 2])
        self.assertEqual(len(record["continuation_events"]), 7)
        self.assertTrue(all(event["software_step_complete"] for event in record["continuation_events"]))
        self.assertTrue(record["workflow_software_complete"])
        self.assertFalse(record["physical_route_confirmed"])
        self.assertFalse(record["launch_result"]["hit_confirmed"])

    def test_shoot_route_failures_never_advance_or_repeat(self):
        for point in ("recognize", "return", "approach", "open", "accepted_only", "fault", "worker",
                      "close", "interrupt", "retreat", "second_shot"):
            with self.subTest(point=point):
                self.project = Path(self.directory.name) / point
                self.project.mkdir()
                self.plan_path = self.project / "plan.json"
                actions = []

                def motion(plan, steps, *args):
                    actions.extend(ordinal for ordinal, step in steps)
                    if (point, steps[0][0]) in (("return", 4), ("approach", 5), ("retreat", 6)):
                        raise OSError(point)

                def pulse(transport, record, save):
                    record["request_accepted"] = True
                    if point == "interrupt":
                        raise KeyboardInterrupt()
                    if point != "accepted_only":
                        failed_second = point == "second_shot" and record["shot_number"] == 2
                        record.update(software_execution_complete=not failed_second,
                                      pulse_result="failed" if failed_second else "completed",
                                      fault="simulated second pulse failure" if failed_second else (
                                          "stop failed" if point == "fault" else None),
                                      worker_still_running=point == "worker")
                    save()

                with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                        mock.patch("builtins.input", return_value="RUN TEST FIRE"), \
                        mock.patch.object(ROUTE, "execute", side_effect=motion), \
                        mock.patch.object(ROUTE.time, "sleep"), \
                        mock.patch.object(ROUTE.card, "prepare_recognition"), \
                        mock.patch.object(ROUTE.card, "run_auto", return_value={"math_ans": 2, "fruit_code": 6}) as recognize, \
                        mock.patch.object(ROUTE.shoot, "check_live"), \
                        mock.patch.object(ROUTE.shoot, "open_serial") as opening, \
                        mock.patch.object(ROUTE.shoot, "execute_once", side_effect=pulse) as launch:
                    if point == "recognize":
                        recognize.side_effect = OSError(point)
                    if point == "open":
                        opening.side_effect = OSError(point)
                    if point == "close":
                        opening.return_value.close.side_effect = OSError(point)
                    code, serial, check = self.invoke(self.combined_args() + ["--shoot-and-retreat"])
                self.assertEqual(code, 130 if point == "interrupt" else 2)
                recognize.assert_called_once()
                self.assertLessEqual(launch.call_count, 2)
                if point in ("recognize", "return", "approach"):
                    opening.assert_not_called()
                    launch.assert_not_called()
                if point not in ("retreat",):
                    self.assertNotIn(6, actions)
                record = self.result()
                if point == "second_shot":
                    self.assertEqual(launch.call_count, 2)
                    self.assertEqual(len(record["launch_result"]["shot_records"]), 2)
                    self.assertEqual(record["launch_result"]["shots_completed"], 1)
                    self.assertFalse(record["launch_result"]["shot_records"][1]["software_execution_complete"])
                self.assertFalse(record["workflow_software_complete"])
                self.assertEqual(record["status"], "interrupted" if point == "interrupt" else "failed")
                self.assertFalse(record["launch_result"]["actual_discharge_confirmed"])

    def test_combined_return_is_included_in_motion_budget_before_hardware(self):
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/timed_route.json")
        plan["turn_speed_rad_s"] = math.radians(95) / 59
        self.write_plan(plan)
        with mock.patch.object(ROUTE.card, "prepare_recognition") as prepare, \
                mock.patch.object(ROUTE.card, "run_auto") as recognize:
            code, serial, check = self.invoke(["--live", "--start-pose-test", "--recognize-after-stop", "--camera", "0"])
        self.assertEqual(code, 2)
        serial.assert_not_called()
        check.assert_not_called()
        prepare.assert_not_called()
        recognize.assert_not_called()

    def test_combined_recognition_failure_or_interrupt_keeps_motion_complete(self):
        for index, failure in enumerate((RuntimeError("API failed"), ValueError("refused"),
                                         OSError("player failed"), KeyboardInterrupt())):
            with self.subTest(failure=type(failure).__name__):
                self.project = Path(self.directory.name) / str(index)
                self.project.mkdir()
                self.plan_path = self.project / "plan.json"
                with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                        mock.patch("builtins.input", return_value="RUN TEST"), \
                        mock.patch.object(ROUTE.card, "prepare_recognition"), \
                        mock.patch.object(ROUTE, "execute"), \
                        mock.patch.object(ROUTE.card, "run_auto", side_effect=failure) as recognize:
                    code, serial, check = self.invoke(self.combined_args())
                interrupted = isinstance(failure, KeyboardInterrupt)
                self.assertEqual(code, 130 if interrupted else 2)
                self.assertTrue(self.result()["software_sequence_complete"])
                self.assertEqual(self.result()["recognition_status"], "interrupted" if interrupted else "failed")
                self.assertNotIn("recognition_result", self.result())
                self.assertEqual(serial.call_count, 1)
                self.assertEqual(check.call_count, 1)
                self.assertEqual(self.result()["return_heading_status"], "not_started")
                self.assertEqual(self.result()["return_events"], [])
                self.assertFalse(self.result()["workflow_software_complete"])
                recognize.assert_called_once()

    def test_combined_log_failure_before_recognition_never_calls_api(self):
        save = ROUTE.save_result
        writes = []

        def save_or_fail(path, record):
            writes.append(record["recognition_status"])
            if len(writes) == 3:
                raise OSError("cannot persist recognition start")
            save(path, record)

        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="RUN TEST"), \
                mock.patch.object(ROUTE.card, "prepare_recognition"), \
                mock.patch.object(ROUTE, "execute"), \
                mock.patch.object(ROUTE, "save_result", side_effect=save_or_fail), \
                mock.patch.object(ROUTE.card, "run_auto") as recognize:
            code, serial, check = self.invoke(self.combined_args())
        self.assertEqual(code, 2)
        self.assertTrue(self.result()["software_sequence_complete"])
        self.assertEqual(self.result()["recognition_status"], "failed")
        recognize.assert_not_called()

    def test_simulation_logs_software_only_and_never_calls_hardware(self):
        code, serial, check = self.invoke(["--demo"])
        self.assertEqual(code, 0)
        serial.assert_not_called()
        check.assert_not_called()
        result = self.result()
        self.assertEqual(result["status"], "simulation_complete")
        self.assertTrue(result["software_sequence_complete"])
        self.assertEqual(len(result["events"]), 4)
        self.assertTrue(all(event["software_step_complete"] for event in result["events"]))
        self.assertFalse(any(value for key, value in result.items() if key.startswith("physical_")))

    def test_simulation_selected_step_only(self):
        code, serial, check = self.invoke(["--simulate", "--step", "2"])
        self.assertEqual(code, 0)
        self.assertEqual([event["step"] for event in self.result()["events"]], [2])
        serial.assert_not_called()
        check.assert_not_called()

    def test_zero_at_start_between_steps_and_end(self):
        transport = mock.Mock()
        events, virtual = self.execute(transport)
        commands = [call.args[0] for call in transport.send.call_args_list]
        self.assertEqual(commands[0], ROUTE.drive.ZERO)
        self.assertEqual(commands[-1], ROUTE.drive.ZERO)
        last_motion = None
        saw_zero = True
        for command in commands:
            if command == ROUTE.drive.ZERO:
                saw_zero = True
            elif command != last_motion:
                self.assertTrue(saw_zero)
                saw_zero = False
                last_motion = command
        self.assertEqual(len(events), 4)
        self.assertAlmostEqual(virtual.value, 2.0)

    def test_initial_stop_failure_never_moves(self):
        transport = mock.Mock()
        transport.send.side_effect = OSError("failed zero")
        with self.assertRaises(OSError):
            self.execute(transport)
        self.assertTrue(all(call.args[0] == ROUTE.drive.ZERO for call in transport.send.call_args_list))

    def test_motion_or_interruption_failure_requests_zero_and_does_not_continue(self):
        for error in (OSError("write"), KeyboardInterrupt()):
            transport = mock.Mock()
            def send(command):
                if command != ROUTE.drive.ZERO:
                    raise error
            transport.send.side_effect = send
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.execute(transport)
            self.assertEqual(transport.send.call_args, mock.call(ROUTE.drive.ZERO))
            self.assertEqual(sum(call.args[0] != ROUTE.drive.ZERO for call in transport.send.call_args_list), 1)

    def test_stop_failure_prevents_step_completion_and_next_step(self):
        transport = mock.Mock()
        virtual = ROUTE.VirtualClock()
        events = []
        zero_count = 0
        def send(command):
            nonlocal zero_count
            if command == ROUTE.drive.ZERO:
                zero_count += 1
                if zero_count > 1:
                    raise OSError("failed stop")
        transport.send.side_effect = send
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(OSError):
            ROUTE.execute(self.plan, ROUTE.select_steps(self.plan), transport, events,
                          virtual.now, virtual.sleep)
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["software_step_complete"])
        self.assertEqual(transport.send.call_args, mock.call(ROUTE.drive.ZERO))

    def test_backwards_clock_blocks_motion_and_requests_zero(self):
        transport = mock.Mock()
        events = []
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            ROUTE.execute(self.plan, ROUTE.select_steps(self.plan), transport, events,
                          mock.Mock(side_effect=[1.0, 0.9]), mock.Mock())
        self.assertFalse(events[0]["software_step_complete"])
        self.assertTrue(all(call.args[0] == ROUTE.drive.ZERO for call in transport.send.call_args_list))

    def test_late_loop_aborts_instead_of_next_step(self):
        transport = mock.Mock()
        virtual = ROUTE.VirtualClock()
        events = []
        def late_sleep(seconds):
            virtual.value += 1
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            ROUTE.execute(self.plan, ROUTE.select_steps(self.plan), transport, events,
                          virtual.now, late_sleep)
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["software_step_complete"])
        self.assertEqual(transport.send.call_args, mock.call(ROUTE.drive.ZERO))

    def test_send_past_deadline_never_sleeps_negative(self):
        transport = mock.Mock()
        virtual = ROUTE.VirtualClock()
        transport.send.side_effect = lambda command: setattr(virtual, "value", virtual.value + 0.3)
        sleep = mock.Mock()
        with self.assertRaises(RuntimeError):
            ROUTE.hold(transport, (0.05, 0, 0), 0.25, virtual.now, sleep)
        sleep.assert_not_called()

    def test_live_whole_route_and_long_step_rejected_before_preflight(self):
        for args in (["--live"], ["--live", "--step", "0"]):
            code, serial, check = self.invoke(args)
            self.assertEqual(code, 2)
            serial.assert_not_called()
            check.assert_not_called()
        self.plan["steps"][0]["seconds"] = 0.51
        self.write_plan(self.plan)
        code, serial, check = self.invoke(["--live", "--step", "1"])
        self.assertEqual(code, 2)
        serial.assert_not_called()
        check.assert_not_called()

    def test_noninteractive_live_rejected(self):
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=False):
            code, serial, check = self.invoke(["--live", "--step", "1"])
        self.assertEqual(code, 2)
        serial.assert_not_called()
        check.assert_not_called()

    def test_check_does_not_open_port_or_load_plan(self):
        self.plan_path.unlink()
        code, serial, check = self.invoke(["--check"])
        self.assertEqual(code, 0)
        check.assert_called_once_with("/dev/abotbase")
        serial.assert_not_called()

    def test_live_check_failure_no_open(self):
        with mock.patch.object(ROUTE, "PROJECT", self.project), \
                mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(ROUTE.drive, "check_port", side_effect=RuntimeError("busy")), \
                mock.patch.object(ROUTE.drive, "SerialPort") as serial, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ROUTE.main(["--plan", str(self.plan_path), "--live", "--step", "1"]), 2)
            serial.assert_not_called()

    def test_live_cancel_does_not_open(self):
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value=""):
            code, serial, check = self.invoke(["--live", "--step", "1"])
        self.assertEqual(code, 0)
        serial.assert_not_called()
        self.assertEqual(self.result()["status"], "cancelled")

    def test_fake_live_single_step_not_physical_success(self):
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="MOVE"), \
                mock.patch.object(ROUTE, "execute") as execute:
            code, serial, check = self.invoke(["--live", "--step", "2"])
        self.assertEqual(code, 0)
        serial.assert_called_once_with("/dev/abotbase")
        self.assertEqual(len(execute.call_args.args[1]), 1)
        self.assertEqual(execute.call_args.args[1][0][0], 2)
        result = self.result()
        self.assertEqual(result["status"], "software_sequence_complete")
        self.assertFalse(result["physical_stop_confirmed"])
        self.assertFalse(result["physical_route_confirmed"])

    def test_initial_log_failure_blocks_open(self):
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(ROUTE, "save_result", side_effect=OSError("disk")):
            code, serial, check = self.invoke(["--live", "--step", "1"])
        self.assertEqual(code, 2)
        serial.assert_not_called()

    def test_execution_failure_is_recorded_and_transport_cleaned(self):
        for error, expected in ((OSError("write"), 2), (KeyboardInterrupt(), 130)):
            for path in self.project.glob("runs/固定路线/*/result.json"):
                path.unlink()
            with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                    mock.patch("builtins.input", return_value="MOVE"), \
                    mock.patch.object(ROUTE, "execute", side_effect=error):
                code, serial, check = self.invoke(["--live", "--step", "1"])
            self.assertEqual(code, expected)
            serial.return_value.__exit__.assert_called_once()
            self.assertFalse(self.result()["software_sequence_complete"])

    def test_final_log_failure_is_failure_even_after_cancellation(self):
        original = ROUTE.save_result
        calls = []
        def save(path, result):
            calls.append(result["status"])
            if len(calls) > 1:
                raise OSError("disk")
            original(path, result)
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value=""), \
                mock.patch.object(ROUTE, "save_result", side_effect=save):
            code, serial, check = self.invoke(["--live", "--step", "1"])
        self.assertEqual(code, 2)
        serial.assert_not_called()

    def configured_competition_plan(self):
        plan = ROUTE.load_plan(PACKAGE.parents[1] / "config/route_template.json")
        plan["initial_heading"] = "TEST DOUBLE ONLY，不代表现场朝向"
        plan["route_checks"] = dict.fromkeys(ROUTE.ROUTE_CHECKS, True)
        for step in plan["steps"]:
            if step["seconds"] is None:
                step["seconds"] = 0.25
        return plan

    def test_full_route_demo_visits_all_planned_checkpoints_without_hardware(self):
        self.write_plan(ROUTE.load_plan(PACKAGE.parents[1] / "config/route_template.json"))
        code, serial, check = self.invoke(["--demo-route"])
        self.assertEqual(code, 0)
        serial.assert_not_called()
        check.assert_not_called()
        result = self.result()
        self.assertTrue(result["virtual_route_demo"])
        self.assertEqual(result["mode"], "simulation")
        self.assertEqual(len(result["events"]), 25)
        self.assertEqual(result["checkpoints_software_complete"], list(ROUTE.ROUTE_CHECKPOINTS))
        self.assertFalse(result["physical_route_confirmed"])
        self.assertTrue(all(step["seconds"] == 0.25 for step in result["requested_steps"] if step["action"] != "stop"))

    def test_invalid_route_checks_and_checkpoint_metadata_rejected(self):
        for checks in (None, [], {"motion_timings": 1}, {"motion_timings": "true"}):
            candidate = copy.deepcopy(self.plan)
            candidate["route_checks"] = checks
            with self.subTest(checks=checks), self.assertRaises(ValueError):
                self.write_plan(candidate)
                ROUTE.load_plan(self.plan_path)
        for checkpoint, action in (("识别位", "forward"), ("", "stop"), ([], "stop")):
            candidate = copy.deepcopy(self.plan)
            candidate["steps"][0].update(checkpoint=checkpoint, action=action)
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                self.write_plan(candidate)
                ROUTE.load_plan(self.plan_path)

    def test_full_route_flag_cannot_be_used_with_simulation_or_step(self):
        for args in (["--full-route"], ["--simulate", "--full-route"],
                     ["--demo-route", "--full-route"], ["--check", "--full-route"],
                     ["--live", "--full-route", "--step", "1"]):
            with self.subTest(args=args):
                code, serial, check = self.invoke(args)
                self.assertEqual(code, 2)
                serial.assert_not_called()
                check.assert_not_called()

    def test_full_route_missing_each_field_blocks_before_preflight(self):
        for field in ROUTE.ROUTE_CHECKS:
            for value in (False, None):
                plan = self.configured_competition_plan()
                if value is None:
                    del plan["route_checks"][field]
                else:
                    plan["route_checks"][field] = value
                self.write_plan(plan)
                with self.subTest(field=field, value=value):
                    code, serial, check = self.invoke(["--live", "--full-route"])
                    self.assertEqual(code, 2)
                    serial.assert_not_called()
                    check.assert_not_called()

    def test_full_route_unfilled_motion_still_blocks_after_checks(self):
        plan = self.configured_competition_plan()
        plan["steps"][3]["seconds"] = None
        self.write_plan(plan)
        code, serial, check = self.invoke(["--live", "--full-route"])
        self.assertEqual(code, 2)
        serial.assert_not_called()
        check.assert_not_called()

    def test_full_route_missing_reordered_or_duplicate_point_or_motion_after_finish_rejected(self):
        variants = []
        missing = self.configured_competition_plan()
        del missing["steps"][6]["checkpoint"]
        variants.append(missing)
        reordered = self.configured_competition_plan()
        reordered["steps"][6]["checkpoint"], reordered["steps"][12]["checkpoint"] = "2号任务位", "1号任务位"
        variants.append(reordered)
        duplicate = self.configured_competition_plan()
        duplicate["steps"][12]["checkpoint"] = "1号任务位"
        variants.append(duplicate)
        after_finish = self.configured_competition_plan()
        after_finish["steps"].append(copy.deepcopy(after_finish["steps"][0]))
        variants.append(after_finish)
        for plan in variants:
            self.write_plan(plan)
            code, serial, check = self.invoke(["--live", "--full-route"])
            self.assertEqual(code, 2)
            serial.assert_not_called()
            check.assert_not_called()

    def test_full_route_requires_interactive_terminal(self):
        self.write_plan(self.configured_competition_plan())
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=False):
            code, serial, check = self.invoke(["--live", "--full-route"])
        self.assertEqual(code, 2)
        serial.assert_not_called()
        check.assert_not_called()

    def test_full_route_move_confirmation_cancels_without_opening(self):
        self.write_plan(self.configured_competition_plan())
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="MOVE"):
            code, serial, check = self.invoke(["--live", "--full-route"])
        self.assertEqual(code, 0)
        serial.assert_not_called()
        self.assertEqual(self.result()["status"], "cancelled")
        self.assertEqual(self.result()["checkpoints_software_complete"], [])

    def test_fake_live_full_route_executes_all_steps_once_and_finishes_zero(self):
        self.write_plan(self.configured_competition_plan())
        virtual = ROUTE.VirtualClock()
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="RUN ROUTE"), \
                mock.patch.object(ROUTE.time, "monotonic", side_effect=virtual.now), \
                mock.patch.object(ROUTE.time, "sleep", side_effect=virtual.sleep), \
                mock.patch.object(ROUTE.card, "play_audio") as play:
            code, serial, check = self.invoke(["--live", "--full-route"])
        self.assertEqual(code, 0)
        play.assert_called_once()
        self.assertEqual(play.call_args.args[0].name, "finish.wav")
        serial.assert_called_once_with("/dev/abotbase")
        serial.return_value.__exit__.assert_called_once()
        transport = serial.return_value.__enter__.return_value
        self.assertEqual(transport.send.call_args, mock.call(ROUTE.drive.ZERO))
        result = self.result()
        self.assertEqual(result["mode"], "live_full_route")
        self.assertFalse(result["virtual_route_demo"])
        self.assertEqual(len(result["events"]), 25)
        self.assertEqual([event["step"] for event in result["events"]], list(range(1, 26)))
        self.assertEqual(result["checkpoints_software_complete"], list(ROUTE.ROUTE_CHECKPOINTS))
        self.assertTrue(result["software_sequence_complete"])
        self.assertTrue(all(value is False for key, value in result.items() if key.startswith("physical_")))
        self.assertEqual(result["finish_broadcast_status"], "player_returned_success")
        self.assertAlmostEqual(virtual.value, 21.25)

    def test_simulated_finish_does_not_require_audio_assets(self):
        result_path = self.project / "finish.json"
        result = {}
        with mock.patch.object(ROUTE.card, "finish_audio", side_effect=FileNotFoundError) as audio, \
                mock.patch.object(ROUTE.card, "play_audio") as play:
            ROUTE.broadcast_finish(result, result_path, False)
        audio.assert_not_called()
        play.assert_not_called()
        self.assertEqual(result["finish_broadcast_status"], "simulation_skipped")
        self.assertIsNone(result["finish_audio_file"])

    def test_finish_broadcast_failure_is_not_recorded_as_success(self):
        result_path = self.project / "finish.json"
        result = {}
        with mock.patch.object(ROUTE.card, "play_audio", side_effect=OSError("TEST audio failure")):
            with self.assertRaises(OSError):
                ROUTE.broadcast_finish(result, result_path, True)
        self.assertEqual(result["finish_broadcast_status"], "failed")
        self.assertIn("TEST audio failure", result["finish_broadcast_error"])
        self.assertEqual(json.loads(result_path.read_text(encoding="utf-8"))["finish_broadcast_status"], "failed")

    def test_full_route_fault_records_only_completed_checkpoints_and_does_not_resume(self):
        self.write_plan(self.configured_competition_plan())
        original = ROUTE.execute
        virtual = ROUTE.VirtualClock()
        def execute(plan, chosen, transport, events, *ignored):
            def send(command):
                if command[2] != 0:
                    raise OSError("TEST DOUBLE turn fault")
            transport.send.side_effect = send
            original(plan, chosen, transport, events, virtual.now, virtual.sleep)
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="RUN ROUTE"), \
                mock.patch.object(ROUTE, "execute", side_effect=execute):
            code, serial, check = self.invoke(["--live", "--full-route"])
        self.assertEqual(code, 2)
        serial.return_value.__exit__.assert_called_once()
        self.assertEqual(self.result()["status"], "failed")
        self.assertFalse(self.result()["software_sequence_complete"])
        self.assertEqual(len(self.result()["events"]), 3)
        self.assertEqual(self.result()["checkpoints_software_complete"], ["识别位"])
        self.assertFalse(self.result()["events"][-1]["software_step_complete"])

    def test_verified_long_step_can_be_tested_before_whole_route_approval(self):
        plan = self.configured_competition_plan()
        plan["route_checks"]["individual_steps_tested"] = False
        plan["steps"][0]["seconds"] = 12.0
        self.write_plan(plan)
        with mock.patch.object(ROUTE.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="MOVE"), \
                mock.patch.object(ROUTE, "execute") as execute:
            code, serial, check = self.invoke(["--live", "--step", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(len(execute.call_args.args[1]), 1)
        self.assertEqual(execute.call_args.args[1][0][1]["seconds"], 12.0)


if __name__ == "__main__":
    unittest.main()
