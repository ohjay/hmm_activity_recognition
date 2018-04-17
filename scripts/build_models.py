#!/usr/bin/env python -W ignore::DeprecationWarning

import os
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
from .extract_features import load_features

# Initialize initial transition matrix as per the paper
TRANSMAT_PRIOR = np.array([[1/3, 1/3, 1/3, 0],
                           [0,   1/3, 1/3, 1/3],
                           [0,   0,   1/2, 1/2],
                           [0,   0,   0,   1]])

# Estimate model parameters from observed features
def learn_params(activity_h5, model_file, n_components,
                 transmat_prior=TRANSMAT_PRIOR, n_features=None):
    """Save an HMM model (Gaussian emissions) with learned parameters
    (transition and emission probabilities) to model_file.

    Parameters
    ----------
    activity_h5: str
        filepath for the h5 file containing the feature matrix
        and sequence length vector associated with the activity

    mode_file: str
        filepath to save file (should have .pkl file extension)

    n_components: int
        number of states in the model

    transmat_prior: array-like, shape (n_components, n_components)
        prior transition matrix

    n_features: int
        desired size of feature dimension (set to None if no adjustment should be made)
    """
    model = hmm.GMMHMM(n_components=n_components,
                       transmat_prior=transmat_prior,
                       init_params='t', verbose=True, n_iter=20)
    feature_matrix, seq_lengths = load_features(activity_h5)
    if n_features is not None:
        feature_matrix = feature_matrix[:, :n_features]
    print('[o] Feature matrix: %r' % (feature_matrix.shape,))
    print('[o] n_sequences: %d' % len(seq_lengths))
    model.fit(feature_matrix, seq_lengths)
    joblib.dump(model, model_file)
    return model

def populate_model_dir(h5_dir, model_dir, n_components,
                       transmat_prior=TRANSMAT_PRIOR, n_features=None):
    """Populate the model directory with trained models corresponding
    to each h5 file in h5_dir.

    Parameters
    ----------
    h5_dir: str
        directory for h5 files

    model_dir: str
        directory to store models

    n_components: int
        number of states for each model

    transmat_prior: array-like, shape (n_components, n_components)
        prior transition matrix for each model

    n_features: int
        desired size of feature dimension (set to None if no adjustment should be made)
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    for dirpath, dirnames, filenames in os.walk(h5_dir):
        for filename in filenames:
            if filename.endswith('.h5'):
                activity_h5 = os.path.join(h5_dir, filename)
                activity_pkl = filename[:-3] + '.pkl'
                model_file = os.path.join(model_dir, activity_pkl)
                learn_params(activity_h5, model_file, n_components,
                             transmat_prior, n_features=n_features)
