#!/usr/bin/env python

import os
import yaml
import argparse
import scripts.extract_features as ef
import scripts.build_model as bm
import scripts.classify_activity as ca

"""
main.py

To add a new mode, define a function with the signature `<mode_name>(config, options=None)`.
To run, specify MODE at the top of the config file (along with any mode-specific parameters)
and execute `main.py` with the config path as an argument.

Mode options:
- extract_features
- build_models
- classify_activity
"""

def extract_features(config, options=None):
    save_path = options.get('save_path', None)
    st = options.get('st', None)
    lk = options.get('lk', None)
    if 'base_dir' in options:
        ef.process_all_video_dirs(options['base_dir'], save_path=save_path, st=st, lk=lk)
    elif 'video_dir' in options:
        ef.process_video_dir(options['video_dir'], save_path=save_path, st=st, lk=lk)
    elif 'video_path' in options:
        ef.process_video(options['video_path'], save_path=save_path, st=st, lk=lk)

def build_models(config, options=None):
    n_components = options['n_components']
    h5_dir = options['h5_dir']
    model_dir = options['model_dir']
    bm.populate_model_dir(h5_dir, model_dir, n_components)

def classify_activity(config, options=None):
    video_path = options['video_path']
    model_dir = options['model_dir']
    activity_probs = ca.get_activity_probs(video_path, model_dir)
    print(activity_probs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    eval(config['mode'])(config, options=config[config['mode']])
