import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("card_camera", ROOT / "tools" / "card_camera.py")
CARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CARD)

from audio_fixtures import audio_fixture


def setUpModule():
    global AUDIO_FIXTURE
    AUDIO_FIXTURE = audio_fixture(CARD)
    AUDIO_FIXTURE.__enter__()


def tearDownModule():
    AUDIO_FIXTURE.__exit__(None, None, None)


class CardCameraTests(unittest.TestCase):
    def test_all_15_audio_texts_hashes_and_pcm(self):
        for answer in range(1, 6):
            for fruit in (6, 7, 8):
                with self.subTest(answer=answer, fruit=fruit):
                    self.assertEqual(CARD.VISION.broadcast_text({"math_ans": answer, "fruit_code": fruit}),
                                     "识别到旋转靶%d，识别到移动靶%d。" % (answer, fruit))
                    path = CARD.answer_audio({"math_ans": answer, "fruit_code": fruit})
                    with wave.open(str(path), "rb") as audio:
                        self.assertGreater(audio.getnframes() / audio.getframerate(), 2)
                        self.assertGreater(audio.getsampwidth(), 0)

    def test_speech_test_audio_is_separate(self):
        self.assertEqual(CARD.answer_audio().name, "answer_test.wav")

    def test_finish_audio_is_local_pcm_with_expected_text(self):
        self.assertEqual(CARD.finish_audio().name, "finish.wav")
        with wave.open(str(CARD.finish_audio()), "rb") as audio:
            self.assertGreater(audio.getnframes() / audio.getframerate(), 1)
            self.assertEqual(audio.getcomptype(), "NONE")

    def test_corrupt_audio_or_mismatched_text_rejected(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(CARD, "AUDIO", Path(folder)):
            path = Path(folder)
            entry = {"file": "test.wav", "text": "wrong", "sha256": "wrong"}
            (path / "manifest.json").write_text(json.dumps({"entries": {"1_6": entry}}))
            with self.assertRaisesRegex(ValueError, "文案"):
                CARD.answer_audio({"math_ans": 1, "fruit_code": 6})
            entry["text"] = CARD.VISION.broadcast_text({"math_ans": 1, "fruit_code": 6})
            (path / "test.wav").write_bytes(b"bad")
            (path / "manifest.json").write_text(json.dumps({"entries": {"1_6": entry}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "校验"):
                CARD.answer_audio({"math_ans": 1, "fruit_code": 6})

    def test_audio_path_escape_rejected(self):
        entry = {"file": "../outside.wav", "text": CARD.VISION.broadcast_text({"math_ans": 1, "fruit_code": 6})}
        with mock.patch.object(CARD.json, "loads", return_value={"entries": {"1_6": entry}}):
            with self.assertRaisesRegex(ValueError, "越界"):
                CARD.answer_audio({"math_ans": 1, "fruit_code": 6})

    def test_model_refusal_missing_fields_and_wrong_types(self):
        for payload in ('{"recognized":false}', '[]', '{}', '{"math_ans":true,"fruit_code":6}',
                        '{"math_ans":1,"fruit_code":9}', '{"math_ans":6,"fruit_code":6}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                CARD.VISION.parse_model_result(payload)

    def test_no_mode_does_not_default_to_live(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit), \
                mock.patch.object(CARD.VISION, "request_vision") as request:
            CARD.main([])
        request.assert_not_called()

    def test_check_no_capture_network_audio_or_env_loading(self):
        with mock.patch.object(CARD, "capture_card") as capture, \
                mock.patch.object(CARD.VISION, "request_vision") as request, \
                mock.patch.object(CARD.VISION, "load_env_file") as load, \
                mock.patch.object(CARD, "play_audio") as play, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(CARD.main(["--check"]), 0)
        self.assertTrue(json.loads(output.getvalue())["answer_audio_valid"])
        for method in (capture, request, load, play):
            method.assert_not_called()

    def test_speech_test_only_plays_test_audio(self):
        with mock.patch.object(CARD, "play_audio") as play, \
                mock.patch.object(CARD.VISION, "request_vision") as request, \
                mock.patch.object(CARD, "capture_card") as capture, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(CARD.main(["--speech-test"]), 0)
        self.assertEqual(play.call_args.args[0].name, "answer_test.wav")
        request.assert_not_called()
        capture.assert_not_called()

    def test_live_requires_explicit_source(self):
        with mock.patch.object(CARD, "run_recognition") as run, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(CARD.main(["--live"]), 2)
        run.assert_not_called()

    def test_preview_requires_camera(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(CARD.main(["--preview"]), 2)

    def test_invalid_timeouts_never_request(self):
        with mock.patch.object(CARD, "run_recognition") as run, contextlib.redirect_stderr(io.StringIO()):
            for value in ("nan", "inf", "0", "-1", "121"):
                self.assertEqual(CARD.main(["--live", "--image", "unused.jpg", "--timeout", value]), 2)
        run.assert_not_called()

    def fake_cv(self, key=32, valid=True, opened=True):
        camera = mock.Mock()
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        camera.isOpened.return_value = opened
        camera.read.return_value = (valid, frame if valid else None)
        camera.grab.return_value = valid
        camera.retrieve.return_value = (valid, frame if valid else None)
        fake = types.SimpleNamespace(VideoCapture=mock.Mock(return_value=camera), CAP_OPENNI2=900,
                                     CAP_OPENNI_BGR_IMAGE=5, CAP_V4L2=200, IMWRITE_JPEG_QUALITY=1,
                                     WND_PROP_VISIBLE=4, imshow=mock.Mock(), waitKey=mock.Mock(return_value=key),
                                     getWindowProperty=mock.Mock(return_value=1), destroyAllWindows=mock.Mock(),
                                     imencode=mock.Mock(return_value=(True, np.array([1, 2, 3], dtype=np.uint8))))
        return fake, camera

    def test_capture_uses_current_frame_and_releases_device(self):
        fake, camera = self.fake_cv()
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict("sys.modules", {"cv2": fake}):
            output = Path(folder) / "中文" / "input.jpg"
            self.assertEqual(CARD.capture_card("0", output), output)
            self.assertEqual(output.read_bytes(), b"\x01\x02\x03")
        fake.VideoCapture.assert_called_once_with(0)
        camera.release.assert_called_once()
        fake.destroyAllWindows.assert_called_once()

    def test_openni_explicit_color_stream(self):
        fake, camera = self.fake_cv()
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict("sys.modules", {"cv2": fake}):
            CARD.capture_card("openni2", Path(folder) / "input.jpg")
        fake.VideoCapture.assert_called_once_with(fake.CAP_OPENNI2)
        camera.retrieve.assert_called_once_with(0, fake.CAP_OPENNI_BGR_IMAGE)
        camera.read.assert_not_called()

    def test_video_file_rejected_as_live_source(self):
        fake, camera = self.fake_cv()
        with mock.patch.dict("sys.modules", {"cv2": fake}), self.assertRaises(ValueError):
            CARD.capture_card("old-video.mp4", Path("unused.jpg"))
        fake.VideoCapture.assert_not_called()

    def test_cancel_unavailable_and_empty_camera_never_save(self):
        for options in ({"key": ord("q")}, {"valid": False}, {"opened": False}):
            with self.subTest(options=options):
                fake, camera = self.fake_cv(**options)
                with tempfile.TemporaryDirectory() as folder, mock.patch.dict("sys.modules", {"cv2": fake}):
                    output = Path(folder) / "input.jpg"
                    with self.assertRaises(RuntimeError):
                        CARD.capture_card("0", output)
                    self.assertFalse(output.exists())
                camera.release.assert_called_once()

    def test_preview_space_never_captures(self):
        fake, camera = self.fake_cv()
        fake.waitKey.side_effect = [32, ord("q")]
        with mock.patch.dict("sys.modules", {"cv2": fake}), self.assertRaises(RuntimeError):
            CARD.capture_card("0", Path("unused.jpg"), preview_only=True)
        fake.imencode.assert_not_called()

    def test_linux_player_failure_propagates_without_shell(self):
        with mock.patch.dict("os.environ", {"AIC_AUDIO_DEVICE": ""}), \
                mock.patch.object(CARD, "audio_player", return_value="aplay"), \
                mock.patch.object(CARD.subprocess, "run", side_effect=subprocess.CalledProcessError(
                    1, "aplay", stderr=b"audio device busy")) as run:
            with self.assertRaisesRegex(RuntimeError, "audio device busy"):
                CARD.play_audio(Path("test.wav"))
        self.assertEqual(run.call_args.args[0], ["aplay", "-q", "test.wav"])
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["stderr"], subprocess.PIPE)

    def test_explicit_alsa_output_is_passed_as_one_argument(self):
        device = "plughw:CARD=Speaker,DEV=0"
        with mock.patch.dict("os.environ", {"AIC_AUDIO_DEVICE": device}), \
                mock.patch.object(CARD.sys, "platform", "linux"), \
                mock.patch.object(CARD.shutil, "which", return_value="/usr/bin/aplay"), \
                mock.patch.object(CARD.subprocess, "run") as run:
            CARD.play_audio(Path("test.wav"))
        self.assertEqual(run.call_args.args[0], ["aplay", "-q", "-D", device, "test.wav"])
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_explicit_alsa_output_never_falls_back_to_other_player(self):
        with mock.patch.dict("os.environ", {"AIC_AUDIO_DEVICE": "plughw:CARD=Speaker,DEV=0"}), \
                mock.patch.object(CARD.sys, "platform", "linux"), \
                mock.patch.object(CARD.shutil, "which", return_value=None), \
                self.assertRaisesRegex(RuntimeError, "aplay"):
            CARD.audio_player()

    def test_audio_timeout_does_not_retry(self):
        with mock.patch.object(CARD, "audio_player", return_value="aplay"), \
                mock.patch.object(CARD.subprocess, "run", side_effect=subprocess.TimeoutExpired("aplay", 30)) as run, \
                self.assertRaisesRegex(RuntimeError, "30秒"):
            CARD.play_audio(Path("test.wav"))
        run.assert_called_once()

    def test_missing_player_rejected(self):
        with mock.patch.object(CARD.sys, "platform", "linux"), \
                mock.patch.object(CARD.shutil, "which", return_value=None), self.assertRaises(RuntimeError):
            CARD.audio_player()

    def recognition_context(self, folder):
        image = Path(folder) / "source.jpg"
        image.write_bytes(b"image bytes for mocked request")
        return image, Path(folder) / "session"

    def test_real_pipeline_mocked_request_plays_exact_answer_and_records_stages(self):
        result = {"math_ans": 2, "fruit_code": 6}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(CARD.VISION, "load_env_file"), \
                mock.patch.dict("os.environ", {"DOUBAO_API_KEY": "private-test-not-a-real-key"}), \
                mock.patch.object(CARD.VISION, "request_vision", return_value=result) as request, \
                mock.patch.object(CARD, "audio_player", return_value="mock"), \
                mock.patch.object(CARD, "play_audio") as play, contextlib.redirect_stdout(io.StringIO()):
            image, session = self.recognition_context(folder)
            self.assertEqual(CARD.run_recognition(image, session, 60), result)
            record_text = (session / "result.json").read_text(encoding="utf-8")
            record = json.loads(record_text)
            self.assertEqual(record["vision_status"], "model_result_validated")
            self.assertEqual(record["playback_status"], "player_returned_success")
            self.assertFalse(record["physical_audio_confirmed"])
            self.assertNotIn("private-test-not-a-real-key", record_text)
            self.assertEqual(play.call_args.args[0].name, "answer_2_6.wav")
            request.assert_called_once()
            self.assertEqual(record["image_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertEqual(record["attempts"][0]["request_provider"], "doubao")
            self.assertEqual(record["attempts"][0]["request_model"], "doubao-seed-2-1-pro-260915")
            self.assertIsNone(record["attempts"][0]["image_detail"])
            self.assertEqual(record["attempts"][0]["thinking"], "disabled")
            self.assertIsNone(record["attempts"][0]["reasoning_effort"])

    def test_wrong_model_in_environment_blocks_recognition(self):
        with mock.patch.object(CARD.VISION, "load_env_file"), \
                mock.patch.dict("os.environ", {"AIC_VISION_PROVIDER": "deepseek",
                                               "AIC_VISION_API_KEY": "test",
                                               "AIC_VISION_MODEL": "deepseek-v4-pro"}), \
                mock.patch.object(CARD, "audio_player", return_value="mock"), \
                mock.patch.object(CARD.VISION, "request_vision") as request, \
                mock.patch.object(CARD, "play_audio") as play:
            with self.assertRaisesRegex(ValueError, "固定使用 deepseek-flash"):
                CARD.prepare_recognition()
            request.assert_not_called()
            play.assert_not_called()

    def test_network_failure_no_answer_no_audio_no_retry(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(CARD.VISION, "load_env_file"), \
                mock.patch.dict("os.environ", {"DOUBAO_API_KEY": "test"}), \
                mock.patch.object(CARD.VISION, "request_vision", side_effect=RuntimeError("offline")) as request, \
                mock.patch.object(CARD, "audio_player", return_value="mock"), \
                mock.patch.object(CARD, "play_audio") as play, contextlib.redirect_stdout(io.StringIO()):
            image, session = self.recognition_context(folder)
            with self.assertRaises(RuntimeError):
                CARD.run_recognition(image, session, 60)
            record = json.loads((session / "result.json").read_text())
            self.assertEqual(record["vision_status"], "request_attempted")
            self.assertEqual(record["playback_status"], "not_started")
            self.assertNotIn("result", record)
            request.assert_called_once()
            play.assert_not_called()

    def test_audio_failure_keeps_vision_result_but_not_playback_success(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(CARD.VISION, "load_env_file"), \
                mock.patch.dict("os.environ", {"DOUBAO_API_KEY": "test"}), \
                mock.patch.object(CARD.VISION, "request_vision", return_value={"math_ans": 3, "fruit_code": 7}), \
                mock.patch.object(CARD, "audio_player", return_value="mock"), \
                mock.patch.object(CARD, "play_audio", side_effect=RuntimeError("speaker failed")), \
                contextlib.redirect_stdout(io.StringIO()):
            image, session = self.recognition_context(folder)
            with self.assertRaises(RuntimeError):
                CARD.run_recognition(image, session, 60)
            record = json.loads((session / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(record["vision_status"], "model_result_validated")
            self.assertEqual(record["playback_status"], "failed")
            self.assertEqual(record["playback_error"], "speaker failed")
            self.assertEqual(record["error_type"], "RuntimeError")

    def test_missing_key_never_requests_or_plays(self):
        with mock.patch.object(CARD.VISION, "load_env_file"), \
                mock.patch.dict("os.environ", {"DOUBAO_API_KEY": ""}), \
                mock.patch.object(CARD, "audio_player", return_value="mock"), \
                mock.patch.object(CARD.VISION, "request_vision") as request, \
                mock.patch.object(CARD, "play_audio") as play, self.assertRaises(RuntimeError):
            CARD.run_recognition(Path("unused.jpg"), Path("unused-session"), 60)
        request.assert_not_called()
        play.assert_not_called()

    def test_motion_preflight_checks_resources_without_opening_camera_or_network(self):
        fake, camera = self.fake_cv()
        with mock.patch.dict("sys.modules", {"cv2": fake}), \
                mock.patch.object(CARD.VISION, "load_env_file"), \
                mock.patch.dict("os.environ", {"DOUBAO_API_KEY": "offline-preflight-test"}), \
                mock.patch.object(CARD, "audio_player", return_value="mock"), \
                mock.patch.object(CARD.VISION, "request_vision") as request, \
                mock.patch.object(CARD, "play_audio") as play:
            self.assertEqual(CARD.prepare_recognition("0"), "offline-preflight-test")
        fake.VideoCapture.assert_not_called()
        request.assert_not_called()
        play.assert_not_called()

    def test_motion_preflight_rejects_missing_opencv_before_loading_key(self):
        with mock.patch.dict("sys.modules", {"cv2": None}), \
                mock.patch.object(CARD.VISION, "load_env_file") as load, \
                self.assertRaisesRegex(RuntimeError, "cv2"):
            CARD.prepare_recognition("0")
        load.assert_not_called()

    def test_motion_preflight_rejects_old_video_without_loading_key(self):
        with mock.patch.object(CARD.VISION, "load_env_file") as load, \
                self.assertRaises(ValueError):
            CARD.prepare_recognition("old-video.mp4")
        load.assert_not_called()

    def test_live_camera_receives_fresh_capture_path(self):
        with mock.patch.object(CARD, "capture_card", return_value=Path("new_capture.jpg")) as capture, \
                mock.patch.object(CARD, "run_recognition") as recognize, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(CARD.main(["--live", "--camera", "0"]), 0)
        capture.assert_called_once()
        self.assertEqual(recognize.call_args.args[0], Path("new_capture.jpg"))

    def test_auto_requires_camera_and_explicit_mode(self):
        with mock.patch.object(CARD, "capture_card") as capture, \
                mock.patch.object(CARD, "run_recognition") as recognize, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(CARD.main(["--auto"]), 2)
            self.assertEqual(CARD.main(["--auto", "--image", "old.jpg"]), 2)
        capture.assert_not_called()
        recognize.assert_not_called()

    def test_auto_main_enables_stability_and_confirmation(self):
        with mock.patch.object(CARD, "capture_card", return_value=Path("fresh.jpg")) as capture, \
                mock.patch.object(CARD, "run_recognition") as recognize, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(CARD.main(["--auto", "--camera", "0", "--timeout", "10"]), 0)
            self.assertTrue(capture.call_args.kwargs["automatic"])
            callback = recognize.call_args.kwargs["confirm_capture"]
            self.assertEqual(callback(Path("confirmation.jpg")), Path("fresh.jpg"))
            self.assertEqual(capture.call_count, 2)

    def test_auto_capture_waits_for_stability_ignores_space_and_handles_motion(self):
        import cv2
        for moving in (False, True):
            with self.subTest(moving=moving):
                fake, camera = self.fake_cv()
                fake.COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY
                for name in ("cvtColor", "resize", "mean", "absdiff"):
                    setattr(fake, name, getattr(cv2, name))
                if moving:
                    camera.read.side_effect = [(True, np.full((20, 20, 3), value, dtype=np.uint8))
                                               for value in range(0, 40, 2)]
                with tempfile.TemporaryDirectory() as folder, mock.patch.dict("sys.modules", {"cv2": fake}), \
                        mock.patch.object(CARD.time, "monotonic", side_effect=itertools.count(0, 0.25)):
                    output = Path(folder) / "input.jpg"
                    if moving:
                        with self.assertRaisesRegex(RuntimeError, "超时"):
                            CARD.capture_card("0", output, timeout=3, automatic=True)
                        self.assertFalse(output.exists())
                        fake.imencode.assert_not_called()
                    else:
                        self.assertEqual(CARD.capture_card("0", output, automatic=True), output)
                        self.assertGreaterEqual(camera.read.call_count, 3)
                camera.release.assert_called_once()
                fake.destroyAllWindows.assert_called_once()

    def test_auto_capture_cancel_never_encodes(self):
        fake, camera = self.fake_cv(key=ord("q"))
        with mock.patch.dict("sys.modules", {"cv2": fake}), self.assertRaisesRegex(RuntimeError, "取消"):
            CARD.capture_card("0", Path("unused.jpg"), automatic=True)
        fake.imencode.assert_not_called()
        camera.release.assert_called_once()

    def test_final_capture_saves_first_valid_new_frame_without_stability_or_space(self):
        fake, camera = self.fake_cv(key=0)
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict("sys.modules", {"cv2": fake}):
            output = Path(folder) / "new.jpg"
            self.assertEqual(CARD.capture_card("0", output, immediate=True), output)
            self.assertTrue(output.exists())
        camera.read.assert_called_once()
        camera.release.assert_called_once()
        fake.destroyAllWindows.assert_called_once()

    def test_final_capture_cancellation_does_not_encode(self):
        fake, camera = self.fake_cv(key=ord("q"))
        with mock.patch.dict("sys.modules", {"cv2": fake}), self.assertRaisesRegex(RuntimeError, "取消"):
            CARD.capture_card("0", Path("unused.jpg"), immediate=True)
        fake.imencode.assert_not_called()
        camera.release.assert_called_once()

    def test_legacy_response_timeout_triggers_fresh_final_photo_and_local_audio(self):
        class LegacySocketTimeout(OSError):
            pass

        answer = {"math_ans": 2, "fruit_code": 6}
        first_response = mock.MagicMock()
        first_response.__enter__.return_value.read.side_effect = LegacySocketTimeout("The read operation timed out")
        final_response = mock.MagicMock()
        final_response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": json.dumps(answer)}}]}).encode("utf-8")

        def capture(camera, output, **kwargs):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(str(output).encode("utf-8"))
            return output

        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(CARD, "prepare_recognition", return_value="offline-only-test-key"), \
                mock.patch.object(CARD.VISION.socket, "timeout", LegacySocketTimeout), \
                mock.patch.object(CARD.VISION.urllib.request, "urlopen",
                                  side_effect=[first_response, final_response]) as opened, \
                mock.patch.object(CARD, "capture_card", side_effect=capture) as captured, \
                mock.patch.object(CARD, "play_audio") as play, \
                contextlib.redirect_stdout(io.StringIO()):
            session = Path(folder) / "session"
            self.assertEqual(CARD.run_auto("0", session, 30), answer)
            self.assertEqual(opened.call_count, 2)
            self.assertEqual(captured.call_count, 2)
            self.assertTrue(captured.call_args.kwargs["immediate"])
            play.assert_called_once_with(CARD.answer_audio(answer))
            first = json.loads((session / "result.json").read_text(encoding="utf-8"))
            final = json.loads((session / "timeout_final" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(first["error_type"], "VisionTimeoutError")
            self.assertEqual(final["answer_source"], "timeout_single_shot")
            self.assertFalse(final["confirmation_satisfied"])
            self.assertNotEqual(first["image_sha256"], final["image_sha256"])

    def test_auto_timeout_final_attempt_is_fresh_bounded_and_explicitly_unconfirmed(self):
        first = {"math_ans": 1, "fruit_code": 8}
        second = {"math_ans": 2, "fruit_code": 6}
        final = {"math_ans": 3, "fruit_code": 7}
        network_timeout = CARD.VISION.VisionTimeoutError("response timeout")
        camera_timeout = CARD.CaptureTimeoutError("stability timeout")
        cases = [
            ("first_request", [network_timeout, final], [], True, final, None),
            ("third_request", [first, second, network_timeout, final], [], True, final, None),
            ("first_capture", [final], [camera_timeout, None], True, final, None),
            ("second_capture", [first, final], [None, camera_timeout, None], True, final, None),
            ("final_timeout", [network_timeout, network_timeout], [], True, None, RuntimeError),
            ("final_refusal", [network_timeout, {"recognized": False}], [], True, None, ValueError),
            ("final_invalid", [network_timeout, {"math_ans": 1, "fruit_code": 9}], [], True, None, ValueError),
            ("final_unreliable", [network_timeout, dict(final, reliable=False)], [], True, None, ValueError),
            ("final_cancel", [network_timeout], [None, RuntimeError("cancelled")], True, None, RuntimeError),
            ("normal_cancel", [], [RuntimeError("cancelled")], False, None, RuntimeError),
            ("normal_auth", [RuntimeError("HTTP 401")], [], False, None, RuntimeError),
            ("normal_refusal", [{"recognized": False}] * 3 + [final], [], True, final, None),
            ("all_refused", [{"recognized": False}] * 4, [], True, None, ValueError),
            ("second_success", [ValueError("refused"), dict(second, reliable=True)], [], False, second, None),
            ("third_success", [{"recognized": False}, dict(first, reliable=False),
                               dict(final, reliable=True)], [], False, final, None),
            ("normal_unconfirmed", [first, second, first, final], [], True, final, None),
            ("normal_success", [dict(first, reliable=True)], [], False, first, None),
            ("interrupt", [KeyboardInterrupt()], [], False, None, KeyboardInterrupt),
        ]
        for name, responses, capture_errors, fallback, expected, error_type in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as folder, \
                    mock.patch.object(CARD.VISION, "load_env_file"), \
                    mock.patch.dict("os.environ", {"DOUBAO_API_KEY": "test-not-real"}), \
                    mock.patch.object(CARD.VISION, "request_vision", side_effect=responses) as request, \
                    mock.patch.object(CARD, "audio_player", return_value="mock"), \
                    mock.patch.object(CARD, "play_audio") as play, contextlib.redirect_stdout(io.StringIO()):
                session = Path(folder) / "session"
                errors = iter(capture_errors)
                def capture(camera, output, **kwargs):
                    error = next(errors, None)
                    if error is not None:
                        raise error
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(str(output).encode())
                    return output
                with mock.patch.object(CARD, "capture_card", side_effect=capture) as captured:
                    if error_type is not None:
                        with self.assertRaises(error_type):
                            CARD.run_auto("0", session, 10)
                    else:
                        self.assertEqual(CARD.run_auto("0", session, 10), expected)
                self.assertEqual(request.call_count, len(responses))
                self.assertLessEqual(request.call_count, 4)
                self.assertEqual(play.call_count, int(expected is not None))
                final_captures = [call for call in captured.call_args_list if call.kwargs.get("immediate")]
                self.assertEqual(len(final_captures), int(fallback))
                self.assertEqual((session / "fallback.json").exists(), fallback)
                if fallback:
                    metadata = json.loads((session / "fallback.json").read_text(encoding="utf-8"))
                    self.assertFalse(metadata["confirmation_satisfied"])
                    final_path = session / "timeout_final" / "result.json"
                    if final_path.exists():
                        record = json.loads(final_path.read_text(encoding="utf-8"))
                        self.assertTrue(record["confirmation_required"])
                        self.assertFalse(record["confirmation_satisfied"])
                        self.assertFalse(record["physical_audio_confirmed"])
                        if expected is not None:
                            expected_source = ("attempt_limit_single_shot" if
                                               metadata["trigger"] == "RecognitionAttemptsExhausted"
                                               else "timeout_single_shot")
                            self.assertEqual(record["answer_source"], expected_source)
                            self.assertEqual(record["result"], expected)
                            self.assertEqual(play.call_args.args[0], CARD.answer_audio(expected))
                            self.assertEqual(request.call_args.args[0], final_path.parent / "input.jpg")
                        else:
                            self.assertNotIn("result", record)
                    paths = [call.args[0] for call in request.call_args_list]
                    self.assertEqual(len(set(paths)), len(paths))

    def test_auto_retries_fresh_images_until_first_reliable_result_or_limit(self):
        first = {"math_ans": 1, "fruit_code": 8}
        second = {"math_ans": 2, "fruit_code": 6}
        third = {"math_ans": 3, "fruit_code": 7}
        cases = [([dict(first, reliable=True)], 1, first),
                 ([{"recognized": False}, dict(second, reliable=True)], 2, second),
                 ([first, dict(second, reliable=False), dict(third, reliable=True)], 3, third),
                 ([dict(first, reliable=False)] * 3, 3, None),
                 ([first, second, first], 3, None), ([first, RuntimeError("offline")], 2, None),
                 ([first, {"recognized": False}, first], 3, None),
                 ([first, second, third], 3, None)]
        for responses, count, expected in cases:
            with self.subTest(responses=responses), tempfile.TemporaryDirectory() as folder, \
                    mock.patch.object(CARD.VISION, "load_env_file"), \
                    mock.patch.dict("os.environ", {"DOUBAO_API_KEY": "test-not-real"}), \
                    mock.patch.object(CARD.VISION, "request_vision", side_effect=responses) as request, \
                    mock.patch.object(CARD, "audio_player", return_value="mock"), \
                    mock.patch.object(CARD, "play_audio") as play, contextlib.redirect_stdout(io.StringIO()):
                image, session = self.recognition_context(folder)
                def capture(output):
                    output.write_bytes(output.name.encode())
                    return output
                callback = mock.Mock(side_effect=capture)
                if expected is None:
                    with self.assertRaises((RuntimeError, ValueError)):
                        CARD.run_recognition(image, session, 10, confirm_capture=callback)
                    play.assert_not_called()
                else:
                    self.assertEqual(CARD.run_recognition(image, session, 10, confirm_capture=callback), expected)
                    play.assert_called_once_with(CARD.answer_audio(expected))
                self.assertEqual(request.call_count, count)
                self.assertEqual(callback.call_count, count - 1)
                uploaded = [call.args[0] for call in request.call_args_list]
                self.assertEqual(len(set(uploaded)), count)
                record = json.loads((session / "result.json").read_text(encoding="utf-8"))
                self.assertFalse(record["confirmation_required"])
                self.assertFalse(record["confirmation_satisfied"])
                self.assertEqual(record["recognition_policy"], "first_reliable_result")
                self.assertEqual(len(record["attempts"]), count)
                self.assertFalse(record["physical_audio_confirmed"])
                if expected is None:
                    self.assertNotIn("result", record)
                    self.assertEqual(record["playback_status"], "not_started")
                else:
                    self.assertEqual(record["successful_attempt_number"], count)
                    self.assertEqual(record["selected_image"], uploaded[-1].name)
                    self.assertEqual(record["answer_source"], "model_declared_reliable_single_shot")

    def test_auto_first_reliable_pair_speaks_once_without_second_capture(self):
        answer = {"math_ans": 2, "fruit_code": 6}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(CARD, "prepare_recognition", return_value="offline-only-key"), \
                mock.patch.object(CARD.VISION, "request_vision", return_value=dict(answer, reliable=True)) as request, \
                mock.patch.object(CARD, "play_audio") as play, contextlib.redirect_stdout(io.StringIO()):
            session = Path(folder) / "session"

            def capture(camera, output, **options):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"new-first-frame")
                return output

            with mock.patch.object(CARD, "capture_card", side_effect=capture) as captures:
                self.assertEqual(CARD.run_auto("0", session, 30), answer)
            captures.assert_called_once()
            request.assert_called_once()
            play.assert_called_once_with(CARD.answer_audio(answer))
            record = json.loads((session / "result.json").read_text(encoding="utf-8"))
            self.assertTrue(record["first_shot_fast_path"])
            self.assertFalse(record["confirmation_required"])
            self.assertFalse(record["confirmation_satisfied"])
            self.assertFalse(record["physical_audio_confirmed"])
            self.assertEqual(record["answer_source"], "model_declared_reliable_single_shot")
            self.assertEqual(record["playback_status"], "player_returned_success")
            self.assertEqual(len(record["attempts"]), 1)
            self.assertTrue(record["attempts"][0]["model_declared_reliable"])
            self.assertFalse((session / "fallback.json").exists())

    def test_second_reliable_result_speaks_immediately_without_pair_confirmation(self):
        answer = {"math_ans": 2, "fruit_code": 6}
        responses = [dict(answer, reliable=False), dict(answer, reliable=True)]
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(CARD, "prepare_recognition", return_value="offline-only-key"), \
                mock.patch.object(CARD.VISION, "request_vision", side_effect=responses) as request, \
                mock.patch.object(CARD, "play_audio") as play, contextlib.redirect_stdout(io.StringIO()):
            image, session = self.recognition_context(folder)

            def capture(output):
                output.write_bytes(b"new-confirmation-frame")
                return output

            with mock.patch.object(CARD, "capture_card", side_effect=lambda camera, output, **options: capture(output)):
                result = CARD.run_recognition(image, session, 30, confirm_capture=capture)
            self.assertEqual(result, answer)
            self.assertEqual(request.call_count, 2)
            play.assert_called_once()
            record = json.loads((session / "result.json").read_text(encoding="utf-8"))
            self.assertFalse(record["first_shot_fast_path"])
            self.assertFalse(record["confirmation_required"])
            self.assertFalse(record["confirmation_satisfied"])
            self.assertEqual(record["successful_attempt_number"], 2)
            self.assertEqual(record["answer_source"], "model_declared_reliable_single_shot")

    def test_invalid_reliable_result_never_fast_paths_or_speaks(self):
        invalid_results = [{"math_ans": 6, "fruit_code": 6, "reliable": True},
                           {"math_ans": 2, "reliable": True},
                           {"math_ans": 2, "fruit_code": 6, "reliable": "true"},
                           {"recognized": False, "math_ans": 2, "fruit_code": 6, "reliable": True}]
        for invalid in invalid_results:
            with self.subTest(result=invalid), tempfile.TemporaryDirectory() as folder, \
                    mock.patch.object(CARD, "prepare_recognition", return_value="offline-only-key"), \
                    mock.patch.object(CARD.VISION, "request_vision", return_value=invalid) as request, \
                    mock.patch.object(CARD, "play_audio") as play, contextlib.redirect_stdout(io.StringIO()):
                image, session = self.recognition_context(folder)
                def capture(output):
                    output.write_bytes(output.name.encode())
                    return output
                confirm = mock.Mock(side_effect=capture)
                with self.assertRaises(CARD.RecognitionAttemptsExhausted):
                    CARD.run_recognition(image, session, 30, confirm_capture=confirm)
                self.assertEqual(request.call_count, 3)
                self.assertEqual(confirm.call_count, 2)
                play.assert_not_called()
                record = json.loads((session / "result.json").read_text(encoding="utf-8"))
                self.assertFalse(record["first_shot_fast_path"])
                self.assertNotIn("result", record)

    def test_fast_path_player_failure_propagates_without_retry(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(CARD, "prepare_recognition", return_value="offline-only-key"), \
                mock.patch.object(CARD.VISION, "request_vision", return_value={
                    "math_ans": 2, "fruit_code": 6, "reliable": True}) as request, \
                mock.patch.object(CARD, "play_audio", side_effect=OSError("player failed")) as play, \
                contextlib.redirect_stdout(io.StringIO()):
            image, session = self.recognition_context(folder)
            confirm = mock.Mock()
            with self.assertRaises(OSError):
                CARD.run_recognition(image, session, 30, confirm_capture=confirm)
            request.assert_called_once()
            confirm.assert_not_called()
            play.assert_called_once()
            record = json.loads((session / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(record["playback_status"], "failed")
            self.assertEqual(record["playback_error"], "player failed")
            self.assertEqual(record["error_type"], "OSError")

    def test_auto_first_request_failure_does_not_capture_confirmation(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(CARD.VISION, "load_env_file"), \
                mock.patch.dict("os.environ", {"DOUBAO_API_KEY": "test-not-real"}), \
                mock.patch.object(CARD.VISION, "request_vision", side_effect=RuntimeError("offline")) as request, \
                mock.patch.object(CARD, "audio_player", return_value="mock"), \
                mock.patch.object(CARD, "play_audio") as play, contextlib.redirect_stdout(io.StringIO()):
            image, session = self.recognition_context(folder)
            callback = mock.Mock()
            with self.assertRaises(RuntimeError):
                CARD.run_recognition(image, session, 10, confirm_capture=callback)
            callback.assert_not_called()
            request.assert_called_once()
            play.assert_not_called()

    def test_auto_confirmation_camera_failure_never_uses_first_answer(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(CARD.VISION, "load_env_file"), \
                mock.patch.dict("os.environ", {"DOUBAO_API_KEY": "test-not-real"}), \
                mock.patch.object(CARD.VISION, "request_vision", return_value={"math_ans": 1, "fruit_code": 8}) as request, \
                mock.patch.object(CARD, "audio_player", return_value="mock"), \
                mock.patch.object(CARD, "play_audio") as play, contextlib.redirect_stdout(io.StringIO()):
            image, session = self.recognition_context(folder)
            callback = mock.Mock(side_effect=RuntimeError("camera cancelled"))
            with self.assertRaises(RuntimeError):
                CARD.run_recognition(image, session, 10, confirm_capture=callback)
            request.assert_called_once()
            play.assert_not_called()
            record = json.loads((session / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(record["vision_status"], "awaiting_next_image")
            self.assertNotIn("result", record)


if __name__ == "__main__":
    unittest.main()
