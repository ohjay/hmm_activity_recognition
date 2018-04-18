#!/usr/bin/env python -W ignore::DeprecationWarning

import os
import warnings
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
from .extract_features import load_features

# Initialize transition matrix as per the paper
# In general, the transition matrix should have shape (n_components, n_components)
TRANSMAT_PRIOR_4x4 = np.array([[1.0/3, 1.0/3, 1.0/3,     0],
                               [0,     1.0/3, 1.0/3, 1.0/3],
                               [0,         0, 1.0/2, 1.0/2],
                               [0,         0,     0,   1.0]])


def learn_params(activity_h5, model_file, model_args, n_features=None):
    """Estimate model parameters from observed features.

    Save an HMM model (Gaussian emissions) with learned parameters
    (transition and emission probabilities) to model_file.

    Parameters
    ----------
    activity_h5: str
        filepath for the h5 file containing the feature matrix
        and sequence length vector associated with the activity

    model_file: str
        filepath to save file (should have .pkl file extension)

    model_args: dict
        model settings (e.g. number of states in the model)

    n_features: int
        desired size of feature dimension (set to None if no adjustment should be made)
    """
    # Model settings
    # n_components, transmat_prior, init_params, verbose, n_iter
    _args = {'init_params': 't', 'verbose': True}
    if 'n_components' in model_args:
        _args['n_components'] = model_args['n_components']
        if _args['n_components'] == 4:
            _args['transmat_prior'] = TRANSMAT_PRIOR_4x4
    _args['n_iter'] = model_args.get('n_iter', 20)
    m_type = model_args.get('m_type', 'gmm').lower()
    print('Initializing %s model with args:\n%r' % (m_type, _args))

    # Initialize model
    if 'gmmhmm'.startswith(m_type):
        Model = hmm.GMMHMM
    else:
        Model = hmm.GaussianHMM
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = Model(**_args)

    # Fit model to feature matrix
    feature_matrix, seq_lengths = load_features(activity_h5)
    if n_features is not None:
        feature_matrix = feature_matrix[:, :n_features]
    print('[o] Feature matrix: %r' % (feature_matrix.shape,))
    print('[o] n_sequences: %d' % len(seq_lengths))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(feature_matrix, seq_lengths)
    joblib.dump(model, model_file)
    return model


def populate_model_dir(h5_dir, model_dir, model_args, n_features=None):
    """Populate the model directory with trained models corresponding
    to each h5 file in h5_dir.

    Parameters
    ----------
    h5_dir: str
        directory for h5 files

    model_dir: str
        directory to store models

    model_args: dict
        model settings (e.g. number of states in the model)

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
                print('%s' % filename)
                print('--------------')
                learn_params(activity_h5, model_file, model_args, n_features)
                print('')
