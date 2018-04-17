#!/usr/bin/env python

import os
import yaml
import argparse
import scripts.extract_features as ef
import scripts.build_models as bm
import scripts.classify_activity as ca

"""
main.py

To add a new command:
- define a function with the signature `<command>(config=None)`
- add <command> to VALID_COMMANDS
- if desired, add a <command> section in the config file

To run, specify parameters in the config file and run `python main.py <command> <config path>`.
"""

# ====================================
# - MODIFY IF A NEW COMMAND IS ADDED -
# ====================================

VALID_COMMANDS = {
    'extract_features',
    'build_models',
    'classify_activity',
}

# ====================
# - COMMAND HANDLERS -
# ====================

def extract_features(config=None):
    save_path = config.get('save_path', None)
    st = config.get('st', None)
    lk = config.get('lk', None)
    features = config.get('features', None)
    if 'base_dir' in config:
        ef.process_all_video_dirs(config['base_dir'], features=features,
                                  save_path=save_path, st=st, lk=lk)
    elif 'video_dir' in config:
        ef.process_video_dir(config['video_dir'], features=features,
                             save_path=save_path, st=st, lk=lk)
    elif 'video_path' in config:
        ef.process_video(config['video_path'], features=features,
                         save_path=save_path, st=st, lk=lk)

def build_models(config=None):
    n_components = config['n_components']
    h5_dir = config['h5_dir']
    model_dir = config['model_dir']
    bm.populate_model_dir(h5_dir, model_dir, n_components)

def classify_activity(config=None):
    video_path = config['video_path']
    model_dir = config['model_dir']
    activity_probs = ca.get_activity_probs(video_path, model_dir)
    print(activity_probs)

# ===============
# - ENTRY POINT -
# ===============

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, help='command')
    parser.add_argument('config', type=str, help='config path')
    args = parser.parse_args()

    # Parse command
    # Note: prefixes work as well (e.g. 'extract' instead of 'extract_features')
    # For correct behavior, make sure the passed-in command string obeys the prefix property
    command = None
    for vc in VALID_COMMANDS:
        if vc.startswith(args.command.lower()):
            command = eval(vc)
            break
    if command is None:
        print('[-] ERROR: unrecognized command %s' % args.command)

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    command(config=config.get(command, None))
