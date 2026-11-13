#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print environment metadata without network requests, robot control or serial access."""
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys


RUNTIME_PROBE = '''
import importlib, json, platform, sys
report = {'python': platform.python_version(), 'executable': sys.executable, 'imports': {}}
for name in ('rospy', 'cv_bridge', 'geometry_msgs.msg', 'TTS_audio.srv', 'cv2', 'openai'):
    try:
        importlib.import_module(name)
        report['imports'][name] = {'ok': True}
    except Exception as exc:
        report['imports'][name] = {'ok': False, 'error_type': type(exc).__name__}
print(json.dumps(report))
'''


def inspect_python(executable):
    try:
        result = subprocess.run([executable, '-B', '-c', RUNTIME_PROBE],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                universal_newlines=True, timeout=8)
        if result.returncode != 0:
            return {'ok': False, 'error_type': 'ProcessFailed'}
        report = json.loads(result.stdout.strip().splitlines()[-1])
        if not isinstance(report, dict) or not all(key in report for key in ('python', 'executable', 'imports')):
            raise ValueError('invalid runtime report')
        return dict(report, ok=True)
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return {'ok': False, 'error_type': type(exc).__name__}


def compare_python():
    reports, cache = {}, {}
    for name in ('python', 'python2', 'python3'):
        executable = shutil.which(name)
        if executable is None:
            reports[name] = {'available': False}
            continue
        resolved = os.path.normcase(os.path.realpath(executable))
        if resolved not in cache:
            cache[resolved] = inspect_python(executable)
        reports[name] = dict(cache[resolved], available=True, command_path=executable)
    return reports


def os_release(path='/etc/os-release'):
    try:
        with open(path, encoding='utf-8') as stream:
            pairs = [line.rstrip().split('=', 1) for line in stream if '=' in line]
        return {key: value.strip('"') for key, value in pairs
                if key in ('ID', 'VERSION_ID', 'PRETTY_NAME')}
    except OSError:
        return {}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-python', action='store_true',
                        help='Compare available Python interpreters; each probe is limited to 8 seconds.')
    args = parser.parse_args(argv)
    report = {'python': sys.version.split()[0], 'python_executable': sys.executable,
              'platform': platform.platform(), 'ros_distro_env': os.environ.get('ROS_DISTRO','unset'),
              'os_release': os_release(), 'architecture': platform.machine(),
              'imports': {}, 'credentials_present': {}, 'serial_paths': {},
              'mplayer_available': bool(shutil.which('mplayer'))}
    for name in ('rospy','actionlib','rosgraph','tf','geometry_msgs.msg','nav_msgs.msg',
                 'ar_track_alvar_msgs.msg','find_object_2d.msg','TTS_audio.srv',
                 'cv_bridge','cv2','numpy','yaml','serial','openai','websockets'):
        try:
            module = importlib.import_module(name)
            report['imports'][name] = {'ok':True,'version':getattr(module,'__version__','unknown')}
        except Exception as exc:
            report['imports'][name] = {'ok':False,'error_type':type(exc).__name__}
    for name in ('DOUBAO_API_KEY','AIC_VISION_API_KEY','AIC_VISION_MODEL','AIC_TTS_APP_ID','AIC_TTS_TOKEN','AIC_TTS_UID'):
        report['credentials_present'][name] = bool(os.environ.get(name))
    for path in ('/dev/shoot','/dev/abot','/dev/abotbase','/dev/video0'):
        report['serial_paths'][path] = {'exists':os.path.exists(path),
                                      'read_write':os.access(path,os.R_OK|os.W_OK)}
    if shutil.which('rosversion'):
        try:
            result=subprocess.run(['rosversion','-d'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                  universal_newlines=True,timeout=5)
            report['rosversion_d']=result.stdout.strip()
        except (OSError,subprocess.TimeoutExpired):
            report['rosversion_d']='unavailable'
    if args.compare_python:
        report['python_runtimes'] = compare_python()
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if all(item['ok'] for item in report['imports'].values()) else 1


if __name__=='__main__':sys.exit(main())
