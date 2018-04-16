#!/usr/bin/env python

import os
import yaml
import argparse
import scripts.extract_features as ef

def extract_features(config, options=None):
    save_path = options.get('save_path', None)
    ef.process_video_dir(options['video_dir'], save_path=save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    eval(config['mode'])(config, options=config[config['mode']])
