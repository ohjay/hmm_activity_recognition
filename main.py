#!/usr/bin/env python

import os
import yaml
import argparse
from pprint import pprint
import scripts.extract_features as ef
import scripts.build_models as bm
import scripts.classify_activity as ca

"""
main.py

To add a new command:
- define a function with the signature `<command>(config)`
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


def extract_features(config):
    ef_params = config['extract_features']
    save_path = ef_params.get('save_path', None)
    if 'base_dir' in ef_params:
        ef.process_all_video_dirs(ef_params['base_dir'], save_path=save_path, config=ef_params)
    elif 'video_dir' in ef_params:
        ef.process_video_dir(ef_params['video_dir'], save_path=save_path, config=ef_params)
    elif 'video_path' in ef_params:
        ef.process_video(ef_params['video_path'], save_path=save_path, config=ef_params)


def build_models(config):
    bm_params = config['build_models']
    h5_dir = bm_params['h5_dir']
    model_dir = bm_params['model_dir']
    all_model_args = bm_params['mconf']
    compute_stats = bm_params('compute_stats', False)
    n_features = bm_params.get('n_features', None)
    bm.populate_model_dir(h5_dir, model_dir, all_model_args, n_features, compute_stats)


def classify_activity(config):
    ef_params = config['extract_features']
    ca_params = config['classify_activity']
    path = ca_params['path']
    model_dir = ca_params['model_dir']
    target = 'all' if bool(ca_params.get('all', False)) else 'single'
    eval_fraction = ca_params.get('eval_fraction', 1.0)
    n_features = ca_params.get('n_features', None)
    result = ca.get_activity_probs(path, model_dir, target, ef_params,
                                   eval_fraction, n_features=n_features)
    pprint(result)
    print('total: %.2f' % sum(result.values()))


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
            command = vc
            break
    if command is None:
        print('[-] ERROR: unrecognized command %s' % args.command)

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    eval(command)(config=config)
