#!/usr/bin/env python

import os
import operator
from hmmlearn import hmm
from sklearn.externals import joblib
from .extract_features import process_video

def classify_single(video_path, models):
    """Classify a single video.

    Parameters
    ----------
    video_path: str
        path to a single video

    models: dict
        a dictionary containing mappings from activity names to trained HMMs
    """
    feature_matrix = process_video(video_path)
    feature_matrix = feature_matrix[:, :20]  # TODO adaptive feature sizes
    activity_probs = {}

    for activity, model in models.items():
        log_prob = model.score(feature_matrix)
        activity_probs[activity] = log_prob
    sorted_activities = sorted([(k, v) for k, v in activity_probs.items()],
                               key=operator.itemgetter(1))
    sorted_activities = list(reversed(sorted_activities))
    return sorted_activities

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

    Returns
    -------
    result: dict or list
        if TARGET == 'all':    a dictionary of classification accuracies for each activity
        if TARGET == 'single': a sorted list of log probabilities for each activity
    """
    # Load models
    models = {}  # {activity: model}
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for filename in filenames:
            if filename.endswith('.pkl'):
                activity = filename[:-4]
                model = joblib.load(os.path.join(model_dir, filename))
                models[activity] = model

    # Determine activity probabilities for videos
    if target == 'all':
        # Classify all videos in the training set
        acc = {}
        num_correct, total = 0, 0
        for video_dir in os.listdir(path):
            video_dir = os.path.join(path, video_dir)
            if not os.path.isdir(video_dir):
                continue
            label_activity = os.path.basename(os.path.normpath(video_dir)).lower()
            for _file in os.listdir(video_dir):
                if not _file.endswith('.avi'):
                    continue
                video_path = os.path.join(video_dir, _file)
                sorted_activities = classify_single(video_path, models)
                if sorted_activities[0][0] == label_activity:
                    num_correct += 1
                total += 1
            acc[label_activity] = float(num_correct) / total
        return acc
    else:
        # Classify a single video
        sorted_activities = classify_single(path, models)
        return sorted_activities
