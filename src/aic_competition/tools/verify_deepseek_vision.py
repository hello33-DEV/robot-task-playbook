#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upload one local image to the configured vision provider and print the answer."""

import argparse
import base64
import json
import mimetypes
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
from aic_competition import vision_config as VISION_CONFIG

DEFAULT_ENV_FILE = PACKAGE_ROOT / "config" / "doubao.env"
DEFAULT_PROVIDER = VISION_CONFIG.DEFAULT_PROVIDER
PROVIDER_DEFAULTS = VISION_CONFIG.PROVIDER_DEFAULTS
DEFAULT_BASE_URL = PROVIDER_DEFAULTS[DEFAULT_PROVIDER]["base_url"]
DEFAULT_MODEL = PROVIDER_DEFAULTS[DEFAULT_PROVIDER]["model"]
IMAGE_DETAIL = "original"
REASONING_EFFORT = "max"
MAX_IMAGE_BYTES = 32 * 1024 * 1024

PROMPT = "识别题卡上方算术题并计算答案(1-5)，识别下方水果并编码(榴莲6、杨桃7、山竹8)。两项都清晰且能确定才返回reliable=true；否则只返回{\"recognized\":false}，不要猜。仅输出JSON：{\"math_ans\":整数,\"fruit_code\":整数,\"reliable\":布尔值}。"


class VisionTimeoutError(RuntimeError):
    pass


load_env_file = VISION_CONFIG.load_env_file


def parse_model_result(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    result = json.loads(text)
    if not isinstance(result, dict) or result.get("recognized") is False:
        raise ValueError("图片未能可靠识别，请调整题卡位置、清晰度和光线后手动重试")
    if "math_ans" not in result or "fruit_code" not in result:
        raise ValueError("模型返回缺少数学答案或水果编号")
    math_ans, fruit_code = result["math_ans"], result["fruit_code"]
    if type(math_ans) is not int or type(fruit_code) is not int:
        raise ValueError("模型返回的编号必须是整数")
    if math_ans not in range(1, 6) or fruit_code not in (6, 7, 8):
        raise ValueError("模型返回的编号超出比赛允许范围")
    validated = {"math_ans": math_ans, "fruit_code": fruit_code}
    if "reliable" in result:
        if type(result["reliable"]) is not bool:
            raise ValueError("模型返回的reliable必须是布尔值")
        validated["reliable"] = result["reliable"]
    return validated


def broadcast_text(result):
    return "识别到旋转靶%d，识别到移动靶%d。" % (
        result["math_ans"],
        result["fruit_code"],
    )


def image_data_url(image_path):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError("图片不存在：%s" % image_path)
    image_size = image_path.stat().st_size
    if image_size <= 0:
        raise ValueError("图片文件为空")
    if image_size > MAX_IMAGE_BYTES:
        raise ValueError("图片超过32MiB限制，请先压缩")
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        raise ValueError("仅支持 JPG、PNG、WEBP 或 GIF 图片")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return "data:%s;base64,%s" % (mime_type, encoded)


def get_vision_settings(provider=None, base_url=None, model=None):
    settings = VISION_CONFIG.get_settings(provider, base_url, model)
    settings["thinking"] = "enabled" if settings["provider"] == "deepseek" else (
        "disabled" if settings["provider"] == "doubao" else "not_set")
    settings["reasoning_effort"] = REASONING_EFFORT if settings["provider"] == "deepseek" else None
    settings["image_detail"] = IMAGE_DETAIL if settings["provider"] == "deepseek" else None
    return settings


def request_vision(image_path, api_key, base_url=None, model=None, timeout=60,
                   provider=None):
    settings = get_vision_settings(provider, base_url, model)
    provider, base_url, model = settings["provider"], settings["base_url"], settings["model"]
    image = {"url": image_data_url(image_path)}
    if settings["image_detail"]:
        image["detail"] = settings["image_detail"]
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": image},
            ],
        }],
        "stream": False,
        "max_tokens": 64,
    }
    if provider == "doubao":
        payload["thinking"] = {"type": "disabled"}
    elif provider == "deepseek":
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = REASONING_EFFORT
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    print("视觉请求配置：provider=%s，model=%s，thinking=%s；等待上限%s秒。"
          % (provider, model, settings["thinking"], timeout), flush=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError("视觉API HTTP错误：%s" % exc.code) from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise VisionTimeoutError(
                "连接视觉API超时（可能发生在 TLS 握手阶段）；"
                "请检查网络或代理后手动重试。尚未取得识别结果。"
            ) from None
        raise RuntimeError("无法连接视觉API：%s" % exc.reason) from None
    except (TimeoutError, socket.timeout):
        raise VisionTimeoutError(
            "等待视觉API响应超时；尚未取得识别结果，服务端是否已处理无法确认。"
            "请检查网络或代理，勿连续重复提交。"
        ) from None
    except ConnectionError:
        raise RuntimeError(
            "与视觉API的连接中断；尚未取得识别结果，服务端是否已处理无法确认。"
            "请检查网络或代理后手动重试。"
        ) from None

    usage = response_body.get("usage", {})
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if prompt_tokens is not None or completion_tokens is not None:
            print("API用量：输入%s tokens，输出%s tokens，总计%s tokens。"
                  % (prompt_tokens if prompt_tokens is not None else "未知",
                     completion_tokens if completion_tokens is not None else "未知",
                     usage.get("total_tokens", "未知")), flush=True)

    try:
        content = response_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("视觉API返回格式不符合预期") from exc
    if not isinstance(content, str):
        raise RuntimeError("视觉API返回内容不是文本")
    return parse_model_result(content)


def validate_model(model, provider=None):
    return VISION_CONFIG.get_settings(provider, model=model)["model"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="用一张本地图片验证配置的视觉模型")
    parser.add_argument("image", help="题卡图片路径，例如 images/card.jpg")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE),
                        help="密钥配置文件，默认是 aic_competition/config/doubao.env")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    load_env_file(Path(args.env_file))
    settings = get_vision_settings()
    api_key = VISION_CONFIG.get_api_key(settings["provider"])
    if not api_key:
        raise SystemExit(
            "未找到视觉API密钥，请在 %s 的等号后填写密钥。" % args.env_file
        )

    result = request_vision(args.image, api_key, timeout=args.timeout,
                            provider=settings["provider"], base_url=settings["base_url"],
                            model=settings["model"])
    print(json.dumps(result, ensure_ascii=False))
    print("模拟播报文本：" + broadcast_text(result))
    print("说明：以上是视觉识别和播报文本验证，不会调用 ROS/TTS、底盘或发射器。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print("验证失败：%s" % exc, file=sys.stderr)
        raise SystemExit(1)
