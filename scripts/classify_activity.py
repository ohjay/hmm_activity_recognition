#!/usr/bin/env python

import os
import operator
from hmmlearn import hmm
from sklearn.externals import joblib
from .extract_features import process_video

def get_activity_probs(video_path, model_dir):
    """ Estimate the most likely activity for the observed sequence

        Parameters
        ----------
            video_path: string
                filepath for the activity video

            model_dir: string
                directory where models are located
    """
    activity_probs = {}
    for (dirpath, dirnames, filenames) in os.walk(model_dir):
        for filename in filenames:
            if filename.endswith('.pkl'):
                activity = filename[:-4]
                model = joblib.load(filename)
                feature_matrix = process_video(video_path)
                log_prob = model.score(feature_matrix)
                activity_probs[activity] = log_prob
    sorted_activities = sorted([(k, v) for k, v in activity_probs.iteritems()],
                               key=operator.itemgetter(1))
    return sorted_activities
