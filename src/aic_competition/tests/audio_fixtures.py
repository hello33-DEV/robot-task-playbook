"""Temporary silence for file-format tests only; never a playable voice resource."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock
import wave


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from prepare_audio import build_manifest


@contextmanager
def audio_fixture(card):
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        filenames = ["answer_%d_%d.wav" % (answer, fruit)
                     for answer in range(1, 6) for fruit in (6, 7, 8)] + ["answer_test.wav", "finish.wav"]
        for filename in filenames:
            with wave.open(str(directory / filename), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(8000)
                audio.writeframes(b"\0\0" * 24000)
        manifest = build_manifest(directory)
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        with mock.patch.object(card, "AUDIO", directory):
            yield
