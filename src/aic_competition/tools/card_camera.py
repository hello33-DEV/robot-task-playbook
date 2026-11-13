#!/usr/bin/env python3
"""One-shot card capture, configured vision client, and pre-generated Chinese audio."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import wave


PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent.parent
AUDIO = PACKAGE / "assets" / "answer_audio"
FINISH_TEXT = "到达终点，比赛结束。"
SPEC = importlib.util.spec_from_file_location("card_vision_client", PACKAGE / "tools" / "verify_deepseek_vision.py")
VISION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VISION)


class RecognitionAttemptsExhausted(RuntimeError):
    pass


def _validated_audio(key, expected):
    manifest = json.loads((AUDIO / "manifest.json").read_text(encoding="utf-8-sig"))
    entry = manifest["entries"][key]
    if entry["text"] != expected:
        raise ValueError("本地语音文案与识别编号不一致，拒绝播报")
    path = (AUDIO / entry["file"]).resolve()
    if path.parent != AUDIO.resolve():
        raise ValueError("语音资源路径越界")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("语音文件校验失败，请重新复制完整工程")
    with wave.open(str(path), "rb") as audio:
        if audio.getnframes() == 0 or audio.getcomptype() != "NONE":
            raise ValueError("本地语音不是有效PCM音频")
    return path


def answer_audio(result=None):
    key = "test" if result is None else "{math_ans}_{fruit_code}".format(**result)
    expected = "语音测试，小车扬声器播放测试。这不是图片识别结果。" if result is None else VISION.broadcast_text(result)
    return _validated_audio(key, expected)


def finish_audio():
    return _validated_audio("finish", FINISH_TEXT)


def audio_player():
    if os.environ.get("AIC_AUDIO_DEVICE", "").strip():
        if not sys.platform.startswith("linux") or not shutil.which("aplay"):
            raise RuntimeError("指定AIC_AUDIO_DEVICE需要Linux及aplay；不会静默改用其他输出设备")
        return "aplay"
    if sys.platform == "win32":
        return "winsound"
    for name in ("aplay", "mplayer", "ffplay"):
        if shutil.which(name):
            return name
    raise RuntimeError("没有可用WAV播放器（aplay/mplayer/ffplay）；先补齐系统音频播放条件")


def play_audio(path):
    player = audio_player()
    device = os.environ.get("AIC_AUDIO_DEVICE", "").strip()
    print("开始本地播音：播放器={}，输出={}，文件={}；尚未确认实际出声。".format(
        player, device or "系统默认音频输出", path.name), flush=True)
    if player == "winsound":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
    else:
        arguments = {"aplay": ["aplay", "-q"] + (["-D", device] if device else []) + [str(path)],
                     "mplayer": ["mplayer", "-really-quiet", str(path)],
                     "ffplay": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)]}
        try:
            subprocess.run(arguments[player], check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            detail = error.stderr or b""
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise RuntimeError("本地播音失败（{}，退出码{}）：{}".format(
                player, error.returncode, detail.strip()[:2000] or "播放器未返回错误详情")) from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("本地播音超过30秒，已终止播放器；不重复识别或自动重播") from error


def inspect_environment():
    report = {"mode": "read_only_check", "camera_opened": False, "api_called": False,
              "audio_device": os.environ.get("AIC_AUDIO_DEVICE", "").strip() or "system_default",
              "private_env_file_exists": VISION.DEFAULT_ENV_FILE.is_file(),
              "camera_type_env": os.environ.get("CAMERA", "unset"), "video_devices": []}
    for device in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            report["video_devices"].append({"device": "/dev/" + device.name,
                                             "name": (device / "name").read_text().strip()})
        except OSError:
            pass
    try:
        import cv2
        report["opencv"] = cv2.__version__
        report["openni_build_lines"] = [line.strip() for line in cv2.getBuildInformation().splitlines()
                                         if "OpenNI" in line]
    except ImportError:
        report["opencv"] = "missing_for_this_python"
    try:
        report["audio_player"] = audio_player()
    except RuntimeError as error:
        report["audio_player"] = str(error)
    report["answer_audio_valid"] = all(answer_audio({"math_ans": answer, "fruit_code": fruit}).is_file()
                                        for answer in range(1, 6) for fruit in (6, 7, 8))
    print(json.dumps(report, ensure_ascii=False, indent=2))


class CaptureTimeoutError(RuntimeError):
    pass


def validate_camera_source(source):
    if not isinstance(source, str) or not (source == "openni2" or source.isdecimal() or
            (sys.platform.startswith("linux") and source.startswith("/dev/video"))):
        raise ValueError("相机只支持明确的编号、/dev/video设备或openni2，不接受视频文件冒充实时相机")


def capture_card(source, output, preview_only=False, timeout=90, automatic=False, immediate=False):
    if immediate and (preview_only or automatic):
        raise ValueError("最后一次立即拍摄不能同时使用预览或稳定等待")
    if automatic and preview_only:
        raise ValueError("自动拍摄不能同时使用只读预览")
    validate_camera_source(source)
    try:
        import cv2
    except ImportError:
        raise RuntimeError("当前Python不能导入cv2；先运行--check，不安装或覆盖系统Python") from None
    openni = source == "openni2"
    if openni:
        camera = cv2.VideoCapture(cv2.CAP_OPENNI2)
    elif source.isdecimal():
        camera = cv2.VideoCapture(int(source))
    elif sys.platform.startswith("linux") and source.startswith("/dev/video"):
        camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        raise ValueError("相机只支持明确的编号、/dev/video设备或openni2，不接受视频文件冒充实时相机")
    title = "Card: AUTO, Q=cancel (no driving)" if automatic else "Card: SPACE=capture, Q=cancel (no driving)"
    try:
        if not camera.isOpened():
            raise RuntimeError("相机无法打开；设备、权限、占用或驱动未确认，不自动改用其他相机")
        started = time.monotonic()
        previous = reference = None
        stable_since = started
        while time.monotonic() - started < timeout:
            if openni:
                ok = camera.grab()
                ok, frame = camera.retrieve(0, cv2.CAP_OPENNI_BGR_IMAGE) if ok else (False, None)
            else:
                ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError("未获取有效彩色画面；不上传空图或回退旧照片")
            cv2.imshow(title, frame)
            key = cv2.waitKey(20) & 0xff
            if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("已取消拍摄，没有发送识别请求")
            ready = False
            if automatic:
                current = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
                now = time.monotonic()
                if previous is None or max(cv2.mean(cv2.absdiff(current, previous))[0],
                                           cv2.mean(cv2.absdiff(current, reference))[0]) > 3.0:
                    stable_since, reference = now, current
                previous = current
                ready = now - stable_since >= 1.0
            if immediate or ready or (key == 32 and not preview_only and not automatic):
                output.parent.mkdir(parents=True, exist_ok=True)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError("题卡图像编码失败")
                output.write_bytes(encoded.tobytes())
                return output
        raise CaptureTimeoutError("相机预览等待超时，没有发送识别请求")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def prepare_recognition(camera=None):
    if camera is not None:
        validate_camera_source(camera)
        try:
            __import__("cv2")
        except ImportError:
            raise RuntimeError("当前Python不能导入cv2；先运行src/aic_competition/tools/card_camera.py --check") from None
    audio_player()
    for answer in range(1, 6):
        for fruit in (6, 7, 8):
            answer_audio({"math_ans": answer, "fruit_code": fruit})
    VISION.load_env_file(VISION.DEFAULT_ENV_FILE)
    settings = VISION.get_vision_settings()
    api_key = VISION.VISION_CONFIG.get_api_key(settings["provider"])
    if not api_key:
        raise RuntimeError("未配置视觉密钥，请填写src/aic_competition/config/doubao.env")
    return api_key


def run_recognition(image, session, timeout, confirm_capture=None, fallback_reason=None):
    api_key = prepare_recognition()
    record = {"mode": "live_api", "started_at": datetime.now(timezone.utc).isoformat(),
              "vision_status": "not_sent", "playback_status": "not_started",
              "physical_audio_confirmed": False,
              "confirmation_required": fallback_reason is not None,
              "confirmation_satisfied": False, "answer_source": None, "fallback_reason": fallback_reason,
              "recognition_policy": "first_reliable_result" if confirm_capture is not None else "single_shot",
              "first_shot_fast_path": False,
              "attempts": []}
    session.mkdir(parents=True, exist_ok=True)
    try:
        for attempt_index in range(3 if confirm_capture is not None else 1):
            if attempt_index:
                image = confirm_capture(session / ("input_%d.jpg" % (attempt_index + 1)))
            image = Path(image)
            snapshot = session / (("input" if attempt_index == 0 else "input_%d" % (attempt_index + 1))
                                  + image.suffix.lower())
            if image.resolve() != snapshot.resolve():
                shutil.copyfile(str(image), str(snapshot))
            attempt = {"image": snapshot.name, "image_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                       "vision_status": "request_attempted",
                       "request_provider": VISION.get_vision_settings()["provider"],
                       "request_model": VISION.get_vision_settings()["model"],
                       "image_detail": VISION.get_vision_settings()["image_detail"],
                       "thinking": VISION.get_vision_settings()["thinking"],
                       "reasoning_effort": VISION.get_vision_settings()["reasoning_effort"]}
            record["attempts"].append(attempt)
            record["image_sha256"] = attempt["image_sha256"]
            record["vision_status"] = "request_attempted"
            print("正在发送第%d张本次实拍图。" % (attempt_index + 1), flush=True)
            try:
                settings = VISION.get_vision_settings()
                validated = VISION.parse_model_result(json.dumps(VISION.request_vision(
                    snapshot, api_key, base_url=settings["base_url"], model=settings["model"],
                    provider=settings["provider"], timeout=timeout)))
            except ValueError as error:
                attempt.update(vision_status="result_rejected", error_type=type(error).__name__)
                if confirm_capture is None:
                    raise
                record["vision_status"] = "awaiting_next_image"
                print("本次未得到合法识别结果，不播报；继续拍摄新图。", flush=True)
                continue
            result = {"math_ans": validated["math_ans"], "fruit_code": validated["fruit_code"]}
            attempt.update(vision_status="model_result_validated", result=result,
                           model_declared_reliable=validated.get("reliable"))
            print("模型返回：" + json.dumps(validated, ensure_ascii=False), flush=True)
            if confirm_capture is None:
                if validated.get("reliable") is False:
                    raise ValueError("模型明确声明本次答案不可靠，不播报")
                break
            if validated.get("reliable") is True:
                record["first_shot_fast_path"] = attempt_index == 0
                print("第%d张两个编号合法且模型自评可靠，立即播报，不再拍下一张复核。"
                      % (attempt_index + 1), flush=True)
                break
            attempt["vision_status"] = "result_not_reliable"
            record["vision_status"] = "awaiting_next_image"
            print("本次未声明可靠，不播报；继续拍摄新图，正常识别最多三次。", flush=True)
        else:
            record["vision_status"] = "attempt_limit_reached"
            raise RecognitionAttemptsExhausted("三次正常识别均未得到可靠的两个编号")
        record["vision_status"] = "model_result_validated"
        record["answer_source"] = ("timeout_single_shot" if fallback_reason is not None else
                                   "model_declared_reliable_single_shot" if confirm_capture is not None else "single_shot")
        if fallback_reason == "RecognitionAttemptsExhausted":
            record["answer_source"] = "attempt_limit_single_shot"
        record["successful_attempt_number"] = attempt_index + 1
        record["selected_image"] = snapshot.name
        record["result"] = result
        record["text"] = VISION.broadcast_text(result)
        print("播报内容：" + record["text"], flush=True)
        record["audio_device"] = os.environ.get("AIC_AUDIO_DEVICE", "").strip() or "system_default"
        record["playback_status"] = "requested"
        try:
            play_audio(answer_audio(result))
        except BaseException as error:
            record["playback_status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            record["playback_error"] = str(error)[:2000]
            raise
        record["playback_status"] = "player_returned_success"
        print("播放器运行完成；请人工确认确实听到，且答案与手中题卡一致。")
        return result
    except BaseException as error:
        record["error_type"] = type(error).__name__
        raise
    finally:
        (session / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("本次图片与结果记录：" + str(session))


def run_auto(camera, session, timeout):
    try:
        image = capture_card(camera, session / "input.jpg", timeout=15, automatic=True)
        return run_recognition(
            image, session, timeout,
            confirm_capture=lambda output: capture_card(camera, output, timeout=15, automatic=True))
    except (CaptureTimeoutError, VISION.VisionTimeoutError, RecognitionAttemptsExhausted) as error:
        reason = type(error).__name__
        print("正常识别超时或达到次数上限：最后重新拍摄一次，不等稳定、不再复核；仅接受本次合法且未明确声明不可靠的模型答案。", flush=True)
        print("这是未经过两次一致确认的兜底答案；最后请求仍可能超时，不保证一定得到答案。", flush=True)
        final_session = session / "timeout_final"
        session.mkdir(parents=True, exist_ok=True)
        record = {"trigger": reason, "final_session": "timeout_final", "outcome": "failed",
                  "confirmation_satisfied": False, "physical_audio_confirmed": False}
        try:
            image = capture_card(camera, final_session / "input.jpg", timeout=15, immediate=True)
            result = run_recognition(image, final_session, timeout, fallback_reason=reason)
            record["outcome"] = "player_returned_success"
            return result
        except BaseException as final_error:
            record["error_type"] = type(final_error).__name__
            raise
        finally:
            (session / "fallback.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="只读检查，不开相机、不联网、不播放")
    mode.add_argument("--preview", action="store_true", help="仅显示相机画面，不联网、不播放")
    mode.add_argument("--speech-test", action="store_true", help="播放独立测试语音，不识别、不联网")
    mode.add_argument("--live", action="store_true", help="明确授权上传一次图片识别并本地播报")
    mode.add_argument("--auto", action="store_true", help="任意一次可靠识别即播报；拒识继续拍新图，正常最多三次，超时或耗尽再加一次末拍，总计最多四次请求")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", help="明确的相机编号、/dev/video设备或openni2")
    source.add_argument("--image", type=Path, help="单张文件（不是实时拍摄）")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 120:
            raise ValueError("请求超时必须大于0且不超过120秒")
        if args.check:
            inspect_environment()
            return 0
        if args.speech_test:
            print("独立语音自检：不是图片识别，不调用视觉API。", flush=True)
            play_audio(answer_audio())
            print("播放器运行完成，请人工确认是否听到中文测试音频。")
            return 0
        if args.preview and not args.camera:
            raise ValueError("预览须明确指定--camera；先运行--check查看设备信息")
        if args.live and args.camera is None and args.image is None:
            raise ValueError("真实识别须明确选择--camera或--image，不自动使用旧照片")
        if args.auto and args.camera is None:
            raise ValueError("自动识别须明确指定--camera，不接受旧图片文件")
        session = PROJECT / "runs" / "识别播报" / (
            datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
        if args.camera is not None:
            if args.auto:
                print("自动模式：任意一次两个编号合法且模型自评可靠即播报；未成功继续拍新图，正常最多三次，超时或耗尽再加一次末拍。")
                run_auto(args.camera, session, args.timeout)
                return 0
            else:
                print("将完整题卡对准彩色摄像头、保持静止，文字清楚；空格拍摄一次，Q取消。")
                image = capture_card(args.camera, session / "input.jpg", preview_only=args.preview)
        else:
            image = args.image
        if args.live:
            run_recognition(image, session, args.timeout)
        return 0
    except KeyboardInterrupt:
        print("已中断；若请求已经发出，无法保证服务端撤销或不计费。", file=sys.stderr)
        return 130
    except Exception as error:
        if isinstance(error, (OSError, ValueError, RuntimeError, subprocess.SubprocessError)):
            message = str(error)
        else:
            message = type(error).__name__
        print("识别播报未完成：" + message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
