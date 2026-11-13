import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_deepseek_vision", ROOT / "tools" / "verify_deepseek_vision.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyDeepSeekVisionTests(unittest.TestCase):
    def test_doubao_uses_openai_compatible_image_request_without_thinking(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"math_ans":3,"fruit_code":7}'}}]
        }).encode("utf-8")
        with patch.object(MODULE, "image_data_url", return_value="data:image/jpeg;base64,eA=="), \
                patch.object(MODULE.urllib.request, "urlopen", return_value=response) as opened:
            self.assertEqual(MODULE.request_vision("unused.jpg", "test-key", timeout=30),
                             {"math_ans": 3, "fruit_code": 7})
        request = opened.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        self.assertEqual(payload["model"], "doubao-seed-2-1-pro-260915")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["max_tokens"], 64)
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["messages"][0]["content"][1]["image_url"],
                         {"url": "data:image/jpeg;base64,eA=="})
        self.assertFalse(payload["stream"])
        self.assertEqual(opened.call_args.kwargs["timeout"], 30)

    def test_deepseek_keeps_its_existing_request_options(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"math_ans":3,"fruit_code":7}'}}]
        }).encode("utf-8")
        with patch.object(MODULE, "image_data_url", return_value="data:image/jpeg;base64,eA=="), \
                patch.object(MODULE.urllib.request, "urlopen", return_value=response) as opened:
            MODULE.request_vision("unused.jpg", "test-key", provider="deepseek", timeout=30)
        payload = json.loads(opened.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(opened.call_args.args[0].full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertEqual(payload["messages"][0]["content"][1]["image_url"]["detail"], "original")

    def test_other_models_are_rejected_before_network_or_image_read(self):
        for model in ("deepseek-v4-pro", "deepseek-reasoner", "deepseek-v4-flash-vision-exp"):
            with self.subTest(model=model), \
                    patch.object(MODULE, "image_data_url") as image, \
                    patch.object(MODULE.urllib.request, "urlopen") as opened:
                with self.assertRaisesRegex(ValueError, "固定使用 deepseek-flash"):
                    MODULE.request_vision("unused.jpg", "test-key", model=model, provider="deepseek")
                image.assert_not_called()
                opened.assert_not_called()

    def test_reliability_is_optional_but_strictly_boolean(self):
        for reliable in (True, False):
            result = MODULE.parse_model_result(json.dumps({"math_ans": 3, "fruit_code": 7, "reliable": reliable}))
            self.assertIs(result["reliable"], reliable)
        for invalid in ("true", "false", 1, 0, None, [], {}):
            with self.subTest(reliable=invalid), self.assertRaises(ValueError):
                MODULE.parse_model_result(json.dumps({"math_ans": 3, "fruit_code": 7, "reliable": invalid}))
        for invalid_pair in ({"math_ans": 6, "fruit_code": 7, "reliable": True},
                             {"math_ans": 3, "reliable": True},
                             {"recognized": False, "math_ans": 3, "fruit_code": 7, "reliable": True}):
            with self.subTest(result=invalid_pair), self.assertRaises(ValueError):
                MODULE.parse_model_result(json.dumps(invalid_pair))

    def test_valid_result_and_broadcast(self):
        result = MODULE.parse_model_result('{"math_ans":3,"fruit_code":7}')
        self.assertEqual(result, {"math_ans": 3, "fruit_code": 7})
        self.assertEqual(MODULE.broadcast_text(result), "识别到旋转靶3，识别到移动靶7。")

    def test_user_supplied_demo_broadcast(self):
        result = MODULE.parse_model_result('{"math_ans":1,"fruit_code":8}')
        self.assertEqual(MODULE.broadcast_text(result), "识别到旋转靶1，识别到移动靶8。")

    def test_fenced_or_invalid_result(self):
        self.assertEqual(
            MODULE.parse_model_result("```json\n{\"math_ans\":5,\"fruit_code\":8}\n```")
            ["fruit_code"],
            8,
        )
        with self.assertRaises(ValueError):
            MODULE.parse_model_result(json.dumps({"math_ans": 2.5, "fruit_code": 7}))

    def test_env_file_does_not_overwrite_process_key(self):
        with tempfile.TemporaryDirectory() as folder:
            env_file = Path(folder) / "doubao.env"
            env_file.write_text("DOUBAO_API_KEY=file-key\n", encoding="utf-8")
            with patch.dict("os.environ", {"DOUBAO_API_KEY": "process-key"}, clear=False):
                MODULE.load_env_file(env_file)
                self.assertEqual(MODULE.os.environ["DOUBAO_API_KEY"], "process-key")

    def test_image_data_url(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "sample.jpg"
            image.write_bytes(b"not-real-jpeg-but-encoding-is-tested")
            data_url = MODULE.image_data_url(image)
            self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))

    def test_network_failures_are_reported_without_retry(self):
        failures = (
            (MODULE.urllib.error.URLError(TimeoutError("TLS handshake timed out")),
             "TLS 握手"),
            (TimeoutError("read timed out"), "响应超时"),
            (ConnectionResetError(10054, "connection reset"), "连接中断"),
        )
        for failure, expected_message in failures:
            with self.subTest(error_type=type(failure).__name__):
                with patch.object(MODULE, "image_data_url", return_value="data:image/jpeg;base64,eA=="):
                    with patch.object(MODULE.urllib.request, "urlopen", side_effect=failure) as mocked_open:
                        with self.assertRaisesRegex(RuntimeError, expected_message):
                            MODULE.request_vision("unused.jpg", "test-key", timeout=1)
                        mocked_open.assert_called_once()

    def test_legacy_socket_timeouts_are_normalized_without_retry(self):
        class LegacySocketTimeout(OSError):
            pass

        for wrapped in (False, True):
            failure = LegacySocketTimeout("The read operation timed out")
            if wrapped:
                failure = MODULE.urllib.error.URLError(failure)
            with self.subTest(wrapped=wrapped), \
                    patch.object(MODULE.socket, "timeout", LegacySocketTimeout), \
                    patch.object(MODULE, "image_data_url", return_value="data:image/jpeg;base64,eA=="), \
                    patch.object(MODULE.urllib.request, "urlopen", side_effect=failure) as opened:
                with self.assertRaises(MODULE.VisionTimeoutError):
                    MODULE.request_vision("unused.jpg", "test-key", timeout=30)
                opened.assert_called_once()

    def test_legacy_socket_timeout_during_response_read_is_normalized(self):
        class LegacySocketTimeout(OSError):
            pass

        response = MagicMock()
        response.__enter__.return_value.read.side_effect = LegacySocketTimeout("The read operation timed out")
        with patch.object(MODULE.socket, "timeout", LegacySocketTimeout), \
                patch.object(MODULE, "image_data_url", return_value="data:image/jpeg;base64,eA=="), \
                patch.object(MODULE.urllib.request, "urlopen", return_value=response) as opened:
            with self.assertRaises(MODULE.VisionTimeoutError):
                MODULE.request_vision("unused.jpg", "test-key", timeout=30)
            opened.assert_called_once()
            response.__exit__.assert_called_once()

    def test_http_error_is_not_reported_as_network_timeout(self):
        failure = MODULE.urllib.error.HTTPError(
            "https://api.deepseek.com/chat/completions", 401, "Unauthorized", None, None
        )
        with patch.object(MODULE, "image_data_url", return_value="data:image/jpeg;base64,eA=="):
            with patch.object(MODULE.urllib.request, "urlopen", side_effect=failure) as mocked_open:
                        with self.assertRaisesRegex(RuntimeError, "HTTP错误：401"):
                            MODULE.request_vision("unused.jpg", "test-key", timeout=1)
                        mocked_open.assert_called_once()

    def test_timeouts_have_a_distinct_type_without_classifying_other_errors_as_timeouts(self):
        for failure in (TimeoutError("read"), MODULE.urllib.error.URLError(TimeoutError("TLS"))):
            with self.subTest(failure=failure), \
                    patch.object(MODULE, "image_data_url", return_value="data:image/jpeg;base64,eA=="), \
                    patch.object(MODULE.urllib.request, "urlopen", side_effect=failure):
                with self.assertRaises(MODULE.VisionTimeoutError):
                    MODULE.request_vision("unused.jpg", "test-key", timeout=1)
        with patch.object(MODULE, "image_data_url", return_value="data:image/jpeg;base64,eA=="), \
                patch.object(MODULE.urllib.request, "urlopen", side_effect=ConnectionResetError("reset")):
            with self.assertRaises(RuntimeError) as raised:
                MODULE.request_vision("unused.jpg", "test-key", timeout=1)
        self.assertNotIsInstance(raised.exception, MODULE.VisionTimeoutError)


if __name__ == "__main__":
    unittest.main()
