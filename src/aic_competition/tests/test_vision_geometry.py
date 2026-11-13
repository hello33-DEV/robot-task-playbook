"""Generated pixel patterns exercise geometry, not real card recognition accuracy."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from aic_competition.vision import CardMatcher, read_image


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cards = Path(self.directory.name)
        self.image = np.random.default_rng(23).integers(0, 256, (600, 400, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", self.image)
        self.assertTrue(success)
        (self.cards / "synthetic.png").write_bytes(encoded.tobytes())
        (self.cards / "index.json").write_text(json.dumps([
            {"file": "synthetic.png", "math_answer": 2, "fruit_id": 6}]), encoding="utf-8")
        self.matcher = CardMatcher(self.cards)

    def test_same_pattern_matches_both_regions(self):
        result, diagnostics = self.matcher.match(self.image)
        self.assertIsNotNone(result, diagnostics)
        self.assertEqual((result["math_answer"], result["fruit_id"]), (2, 6))

    def test_blank_rejected(self):
        result, diagnostics = self.matcher.match(np.full_like(self.image, 255))
        self.assertIsNone(result, diagnostics)

    def test_unrelated_pattern_rejected(self):
        image = np.random.default_rng(71).integers(0, 256, self.image.shape, dtype=np.uint8)
        result, diagnostics = self.matcher.match(image)
        self.assertIsNone(result, diagnostics)

    def test_unreadable_image_returns_none(self):
        path = self.cards / "broken.png"
        path.write_bytes(b"not an image")
        self.assertIsNone(read_image(path))

    def test_missing_template_rejected(self):
        (self.cards / "index.json").write_text('[{"file":"missing.png"}]', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing template"):
            CardMatcher(self.cards)
