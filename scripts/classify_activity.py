#!/usr/bin/env python

import os
from hmmlearn import hmm
from sklearn.externals import joblib

def get_activity_probs(activity_h5, model_dir):
    """ Estimate the most likely activity for the observed sequence

        Parameters
        ----------
            activity_h5: string
                filepath for the h5 file containing the feature matrix
                and sequence length vector associated with the activity

            model_dir: string
                directory where models are located
    """
    activity_probs = {}
    for (dirpath, dirnames, filenames) in os.walk(model_dir):
        for filename in filenames:
            if filename.endswith('.pkl'):
                activity = filename[:-4]
                model = joblib.load(filename)
                # TODO extract feature matrix for one video instead
                #      of a training suite
                feature_matrix, seq_lengths = load_features(activity_h5)
                log_prob = model.score(feature_matrix)
                activity_probs[activity] = log_prob
    sorted_activities = sorted([(k, v) for k, v in activity_probs.iteritems()],
                               key = operator.itemgetter(1))
    return sorted_activities
