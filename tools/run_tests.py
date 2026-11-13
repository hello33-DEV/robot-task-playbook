import argparse
import json
from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "aic_competition"


def main():
    parser = argparse.ArgumentParser(description="Offline tests; no ROS, hardware or real API requests.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    for relative in ("src", "tools", "tests"):
        sys.path.insert(0, str(PACKAGE / relative))
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(PACKAGE / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if args.report:
        report = {"test_count": result.testsRun, "failures": len(result.failures),
                  "errors": len(result.errors), "skipped": len(result.skipped),
                  "successful": result.wasSuccessful(), "seconds": round(time.monotonic() - started, 3),
                  "scope": "Offline doubles and generated fixtures; not a hardware or model accuracy benchmark."}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
