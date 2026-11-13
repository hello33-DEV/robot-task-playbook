"""Known-card recognition; requires independent text and fruit geometric evidence."""
import json
from pathlib import Path
import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags) if data else None


class CardMatcher:
    def __init__(self, cards_dir, min_text=14, min_fruit=12, margin=5, ratio=1.18):
        self.cards_dir = Path(cards_dir)
        self.min_text, self.min_fruit = min_text, min_fruit
        self.margin, self.ratio = margin, ratio
        self.orb = cv2.ORB_create(nfeatures=4500, fastThreshold=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.templates = []
        for entry in json.loads((self.cards_dir / 'index.json').read_text(encoding='utf-8-sig')):
            image = read_image(self.cards_dir / entry['file'], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError('missing template ' + entry['file'])
            h, w = image.shape
            text_mask = np.zeros_like(image)
            text_mask[int(.08*h):int(.30*h), int(.08*w):int(.92*w)] = 255
            fruit_mask = np.zeros_like(image)
            fruit_mask[int(.32*h):int(.85*h), int(.10*w):int(.90*w)] = 255
            tk, td = self.orb.detectAndCompute(image, text_mask)
            fk, fd = self.orb.detectAndCompute(image, fruit_mask)
            self.templates.append((entry, (w, h), tk, td, fk, fd))

    def pairs(self, desc, scene_desc):
        if desc is None or scene_desc is None or len(scene_desc) < 2:
            return []
        pairs = self.matcher.knnMatch(desc, scene_desc, k=2)
        good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .72*b.distance]
        # Multiple template points mapping to one scene feature must count once.
        unique = {}
        for match in sorted(good, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        return list(unique.values())

    def match(self, image):
        if image is None or image.size == 0:
            return None, {'reason': 'empty image'}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        sk, sd = self.orb.detectAndCompute(gray, None)
        candidates = []
        for entry, size, tk, td, fk, fd in self.templates:
            matches = self.pairs(td, sd)
            if len(matches) < self.min_text:
                continue
            src = np.float32([tk[m.queryIdx].pt for m in matches])
            dst = np.float32([sk[m.trainIdx].pt for m in matches])
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            if homography is None or mask is None or not np.isfinite(homography).all():
                continue
            good = mask.ravel().astype(bool)
            text_count = int(good.sum())
            if text_count < self.min_text or text_count / len(matches) < .45:
                continue
            w, h = size
            spread = np.ptp(src[good], axis=0)
            if spread[0] < .25*w or spread[1] < .025*h:
                continue
            corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
            quad = cv2.perspectiveTransform(corners, homography).reshape(-1,2)
            area = abs(cv2.contourArea(quad))
            if not np.isfinite(quad).all() or not cv2.isContourConvex(quad) or area < 2500 \
                    or area > gray.size*2 or (quad[:,0] < -.2*gray.shape[1]).any() \
                    or (quad[:,0] > 1.2*gray.shape[1]).any() \
                    or (quad[:,1] < -.2*gray.shape[0]).any() \
                    or (quad[:,1] > 1.2*gray.shape[0]).any():
                continue
            fm = self.pairs(fd, sd)
            fruit_count = 0
            if fm:
                fs = np.float32([fk[m.queryIdx].pt for m in fm]).reshape(-1,1,2)
                actual = np.float32([sk[m.trainIdx].pt for m in fm])
                projected = cv2.perspectiveTransform(fs, homography).reshape(-1,2)
                fruit_count = int((np.linalg.norm(projected-actual, axis=1) < 7.0).sum())
            if fruit_count < self.min_fruit:
                continue
            candidates.append(dict(entry, text_inliers=text_count, fruit_inliers=fruit_count,
                                   score=float(text_count), corners=quad.tolist()))
        candidates.sort(key=lambda c: c['score'], reverse=True)
        if not candidates:
            return None, {'reason': 'insufficient text/fruit geometry'}
        best = candidates[0]
        second = candidates[1]['score'] if len(candidates) > 1 else 0
        diagnostics = {'best': best['file'], 'score': best['score'], 'runner_up': second}
        best_quad = np.float32(best['corners'])
        best_center = best_quad.mean(axis=0)
        separation = .2 * np.sqrt(abs(cv2.contourArea(best_quad)))
        for other in candidates[1:]:
            if np.linalg.norm(np.float32(other['corners']).mean(axis=0)-best_center) > separation:
                return None, dict(diagnostics, reason='multiple different cards visible')
        if best['score'] - second < self.margin or best['score'] < self.ratio * second:
            return None, dict(diagnostics, reason='ambiguous card')
        return best, diagnostics
