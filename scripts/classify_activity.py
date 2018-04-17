#!/usr/bin/env python

import os
import operator
from hmmlearn import hmm
from sklearn.externals import joblib
from .extract_features import process_video

def get_activity_probs(path, model_dir, target='single'):
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
    """
    feature_matrix = process_video(path)
    feature_matrix = feature_matrix[:, :20]  # TODO adaptive feature sizes
    activity_probs = {}
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for filename in filenames:
            if filename.endswith('.pkl'):
                activity = filename[:-4]
                model = joblib.load(os.path.join(model_dir, filename))
                log_prob = model.score(feature_matrix)
                activity_probs[activity] = log_prob
    sorted_activities = sorted([(k, v) for k, v in activity_probs.items()],
                               key=operator.itemgetter(1))
    sorted_activities = list(reversed(sorted_activities))
    return sorted_activities
