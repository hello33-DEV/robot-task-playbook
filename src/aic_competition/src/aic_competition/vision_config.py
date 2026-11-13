"""Shared provider defaults and local credential-file loading."""
import os
from pathlib import Path


DEFAULT_PROVIDER = "doubao"
PROVIDER_DEFAULTS = {
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-2-1-pro-260915",
    },
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash"},
    "ark": {"base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": ""},
}


def load_env_file(path):
    path = Path(path)
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "DOUBAO_API_KEY":
            continue
        value = value.strip()
        if value[:1] in ("'", '"') and value[-1:] == value[:1]:
            value = value[1:-1]
        os.environ.setdefault("DOUBAO_API_KEY", value)


def get_api_key(provider):
    name = "DOUBAO_API_KEY" if provider == "doubao" else "AIC_VISION_API_KEY"
    return os.environ.get(name, "").strip()


def get_settings(provider=None, base_url=None, model=None):
    provider = (provider or os.environ.get("AIC_VISION_PROVIDER", DEFAULT_PROVIDER)).lower()
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError("不支持的视觉服务提供方：%s" % provider)
    defaults = PROVIDER_DEFAULTS[provider]
    settings = {
        "provider": provider,
        "base_url": base_url or os.environ.get("AIC_VISION_BASE_URL", defaults["base_url"]),
        "model": model or os.environ.get("AIC_VISION_MODEL", defaults["model"]),
    }
    if provider == "deepseek" and settings["model"] != PROVIDER_DEFAULTS["deepseek"]["model"]:
        raise ValueError("本工程 DeepSeek 识图固定使用 deepseek-flash；不自动切换模型")
    if not settings["model"]:
        raise ValueError("未配置 AIC_VISION_MODEL")
    return settings


def doubao_env_file():
    try:
        import rospkg
        package_root = Path(rospkg.RosPack().get_path("aic_competition"))
    except Exception:
        package_root = Path(__file__).resolve().parents[2]
    return package_root / "config" / "doubao.env"
