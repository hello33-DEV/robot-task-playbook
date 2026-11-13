#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
import cv2
from aic_competition.vision import CardMatcher, read_image
p=argparse.ArgumentParser(description='Offline known-card recognition; no ROS/hardware')
p.add_argument('image')
p.add_argument('--annotated')
a=p.parse_args()
im=read_image(a.image)
if im is None:
    raise SystemExit('Cannot read '+a.image)
result,diagnostics=CardMatcher(root/'assets/cards').match(im)
print(json.dumps(dict(result=result,diagnostics=diagnostics),ensure_ascii=False,indent=2))
if result and a.annotated:
    import numpy as np
    cv2.polylines(im,[np.int32(result['corners'])],True,(0,180,0),2)
    cv2.putText(im,'math=%d fruit=%d'%(result['math_answer'],result['fruit_id']),(15,25),
                cv2.FONT_HERSHEY_SIMPLEX,.7,(0,180,0),2)
    success, encoded = cv2.imencode(Path(a.annotated).suffix, im)
    if not success:
        raise SystemExit('Cannot encode '+a.annotated)
    Path(a.annotated).write_bytes(encoded.tobytes())
sys.exit(0 if result else 1)
