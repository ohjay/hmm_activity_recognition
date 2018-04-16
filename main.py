#!/usr/bin/env python

import os
import yaml
import argparse
import scripts.extract_features as ef
import scripts.build_model as bm

def extract_features(config, options=None):
    save_path = options.get('save_path', None)
    if 'video_dir' in options:
        ef.process_video_dir(options['video_dir'], save_path=save_path)
    else:
        ef.process_video(options['video_path'], save_path=save_path)

def build_model(config, options=None):
    bm.learn_params(options['n_components'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    eval(config['mode'])(config, options=config[config['mode']])
