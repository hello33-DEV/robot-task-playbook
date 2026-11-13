#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record map-frame waypoints while robot is manually positioned; never moves it."""
import argparse
import json
import math
import os
import rospy
import tf

p = argparse.ArgumentParser()
p.add_argument('name', help='e.g. recognition, ring, corridor_1, rotating, moving, finish')
p.add_argument('--output', default=os.path.expanduser('~/.ros/aic_waypoints.json'))
a = p.parse_args(rospy.myargv()[1:])
rospy.init_node('aic_record_waypoint', anonymous=True)
listener = tf.TransformListener()
listener.waitForTransform('map', 'base_link', rospy.Time(0), rospy.Duration(10))
stamp = listener.getLatestCommonTime('map', 'base_link')
if (rospy.Time.now()-stamp).to_sec() > 1.5:
    raise RuntimeError('stale TF; fix localization before recording')
xyz, q = listener.lookupTransform('map', 'base_link', rospy.Time(0))
pose = [round(xyz[0],4), round(xyz[1],4), round(math.degrees(tf.transformations.euler_from_quaternion(q)[2]),3)]
data = {}
if os.path.isfile(a.output):
    with open(a.output, encoding='utf-8') as f:
        data = json.load(f)
data[a.name] = pose
parent = os.path.dirname(os.path.abspath(a.output))
os.makedirs(parent, exist_ok=True)
with open(a.output, 'w', encoding='utf-8') as f:
    json.dump(data,f,ensure_ascii=False,indent=2)
print('  %s: [%s]  # x_m, y_m, yaw_deg' % (a.name, ', '.join(str(x) for x in pose)))
print('Saved to '+os.path.abspath(a.output)+'; copy reviewed values to competition.yaml/waypoints.')
