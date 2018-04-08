#!/usr/bin/env python

import os
import yaml
import argparse
import scripts.extract_features as ef

def test_feature_extraction(config, options=None):
    ef.process_video(options['video_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    eval(config['mode'])(config, options=config[config['mode']])
