#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
import yaml
from aic_competition.core import CompetitionMission, validate_config
from aic_competition.simulation import SimIO

p = argparse.ArgumentParser(description='Offline competition simulation; no ROS/hardware required')
p.add_argument('--config', default=str(root/'config/competition.yaml'))
p.add_argument('--scenario', default='success', choices=['success','nav_failure','vision_failure','speech_failure',
                                                        'target_lost','wrong_alignment','stale_sensor','estop'])
p.add_argument('--json', dest='output', help='write full simulation trace JSON')
a = p.parse_args()
c = validate_config(yaml.safe_load(Path(a.config).read_text(encoding='utf-8-sig')))
io = SimIO(c, a.scenario, print)
ok = CompetitionMission(c, io).run()
report = dict(success=ok, scenario=a.scenario, elapsed=io.now(), shots_commanded=io.shots,
              visited=io.visited, cleanup=io.cleanup_done, states=io.states,
              disclaimer='Deterministic software simulation; NOT hardware or accuracy validation')
if a.output:
    Path(a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
sys.exit(0 if ok else 1)
