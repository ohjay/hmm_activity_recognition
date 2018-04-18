#!/usr/bin/env python

import os
import re
import random
import warnings
import operator
import numpy as np
from collections import defaultdict
from sklearn.externals import joblib
from .extract_features import process_video


def classify_single(video_path, models, stats, ef_params, n_features=None):
    """Classify a single video.

    Parameters
    ----------
    video_path: str
        path to a single video

    models: dict
        a dictionary containing mappings from activity names to lists of trained HMMs

    stats: dict
        a dictionary containing mappings from activity names to lists of (mean, stdev) tuples

    ef_params: dict
        a dictionary specifying the feature params used for training

    n_features: int
        desired size of feature dimension (set to None if no adjustment should be made)

    Returns
    -------
    result: list
        a sorted list of log probabilities for each activity
        (or None if the video could not be processed)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        feature_seq_matrix = process_video(video_path, config=ef_params)
    if feature_seq_matrix is not None:
        activity_probs = defaultdict(list)
        for feature_matrix in feature_seq_matrix:
            if n_features is not None:
                feature_matrix = feature_matrix[:, :n_features]
                _zfm = np.zeros((feature_matrix.shape[0], n_features))
                _zfm[:, :feature_matrix.shape[1]] = feature_matrix
                feature_matrix = _zfm

            for activity in models:
                model_list = models[activity]
                stats_list = stats[activity]
                log_probs = []
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    for model, (mean, stdev) in zip(model_list, stats_list):
                        try:
                            normalized_score = (model.score(feature_matrix) - mean) / stdev
                            log_probs.append(normalized_score)
                        except ValueError:
                            pass  # ignore bad models
                activity_probs[activity].append(np.mean(log_probs))
        for activity in activity_probs:
            activity_probs[activity] = np.mean(activity_probs[activity])
        sorted_activities = sorted([(k, v) for k, v in activity_probs.items()],
                                   key=operator.itemgetter(1), reverse=True)
        return sorted_activities
    return None


def get_activity_probs(path, model_dir, target,
                       ef_params, eval_fraction=1.0, n_features=None):
    """Estimate the most likely activity for the observed sequence.

    Parameters
    ----------
    path: str
        either (a) a filepath for a single activity video
        or (b) a filepath to the directory containing all video subdirectories;
        interpretation depends on the value of TARGET

    model_dir: str
        directory where models are located

    target: str
        either 'all' or 'single', representing whether
        we are classifying a single video or ALL videos

    ef_params: dict
        a dictionary specifying the feature params used for training

    eval_fraction: float
        fraction of each population's training set to classify
        (only meaningful if TARGET == 'all')

    n_features: int
        desired size of feature dimension (set to None if no adjustment should be made)

    Returns
    -------
    result: dict or list
        if TARGET == 'all':    a dictionary of classification accuracies for each activity
        if TARGET == 'single': a sorted list of log probabilities for each activity
    """
    # Load models
    models = defaultdict(list)  # {activity: model_list}
    stats = joblib.load(os.path.join(model_dir, 'stats.pkl'))
    stats_tmp = defaultdict(list)  # {activity: list of (mean, stdev) tuples}
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for filename in filenames:
            if filename.endswith('.pkl'):
                m = re.match(r'([a-zA-Z]+)\d+.pkl', filename)
                try:
                    activity = m.group(1)
                    model = joblib.load(os.path.join(model_dir, filename))
                    models[activity].append(model)
                    stats_tmp[activity].append(stats[filename])
                except AttributeError:
                    pass
    stats = stats_tmp

    # Determine activity probabilities for videos
    if target == 'all':
        # Classify all videos in the training set
        acc = {}
        for video_dir in os.listdir(path):
            video_dir = os.path.join(path, video_dir)
            if not os.path.isdir(video_dir):
                continue
            label_activity = os.path.basename(os.path.normpath(video_dir)).lower()

            # Evaluate on a random subset of the training data
            eval_set = [os.path.join(video_dir, f)
                        for f in os.listdir(video_dir) if f.endswith('.avi')]
            sample_size = int(eval_fraction * len(eval_set))
            eval_set = random.sample(eval_set, sample_size)
            num_correct, total = 0, 0
            for i, video_path in enumerate(eval_set):
                sorted_activities = classify_single(video_path, models, stats,
                                                    ef_params, n_features)
                if sorted_activities is None:
                    continue
                elif sorted_activities[0][0] == label_activity:
                    num_correct += 1
                total += 1
                print('[o] Classified %d / %d in the %s evaluation set.'
                      % (i + 1, len(eval_set), label_activity))
                print('[o] Current accuracy: %.2f' % (float(num_correct) / total))
            acc[label_activity] = float(num_correct) / total
        return acc
    else:
        # Classify a single video
        sorted_activities = classify_single(path, models, stats, ef_params)
        return sorted_activities
