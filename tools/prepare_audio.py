import argparse
import hashlib
import json
from pathlib import Path
import sys
import wave


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "aic_competition"
sys.path.insert(0, str(PACKAGE / "tools"))
from card_camera import FINISH_TEXT, VISION


def build_manifest(directory):
    directory = directory.resolve()
    labels = {"%d_%d" % (answer, fruit): ("answer_%d_%d.wav" % (answer, fruit),
              VISION.broadcast_text({"math_ans": answer, "fruit_code": fruit}))
              for answer in range(1, 6) for fruit in (6, 7, 8)}
    labels["test"] = ("answer_test.wav", "语音测试，小车扬声器播放测试。这不是图片识别结果。")
    labels["finish"] = ("finish.wav", FINISH_TEXT)
    entries = {}
    for key, (filename, text) in labels.items():
        path = (directory / filename).resolve()
        if path.parent != directory:
            raise ValueError("Audio file must be inside the requested directory")
        with wave.open(str(path), "rb") as audio:
            if audio.getnframes() == 0 or audio.getcomptype() != "NONE":
                raise ValueError("Expected non-empty PCM WAV: " + filename)
        entries[key] = {"file": filename, "text": text,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"entries": entries}


def main():
    parser = argparse.ArgumentParser(description="Index user-supplied WAVs; does not synthesize or play audio.")
    parser.add_argument("--directory", type=Path, default=PACKAGE / "assets" / "answer_audio")
    args = parser.parse_args()
    manifest = build_manifest(args.directory)
    (args.directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Indexed %d WAV files; text correctness and audibility still require listening." % len(manifest["entries"]))


if __name__ == "__main__":
    main()
