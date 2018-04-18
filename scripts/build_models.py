#!/usr/bin/env python -W ignore::DeprecationWarning

import os
import shutil
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


def rm_rf(dir, confirmation_prompt=None):
    """Remove a directory and all of its contents.
    Warning: this is potentially a very dangerous operation.
    """
    if type(confirmation_prompt) == str:
        try:
            confirmation = raw_input('%s ' % confirmation_prompt)
        except NameError:
            confirmation = input('%s ' % confirmation_prompt)
    else:
        confirmation = True
    if (isinstance(confirmation, bool) and confirmation) \
            or (isinstance(confirmation, str) and confirmation.lower() == 't'):
        shutil.rmtree(dir)
        print('Successfully removed `%s` and all of its contents.' % dir)
        return True
    else:
        print('Operation `rm -rf %s` aborted.' % dir)
        return False


def subsample_feature_matrix(feature_matrix, seq_lengths, p):
    """Sample sequences (each with probability P) from the feature matrix.
    Return a list of feature matrix components and an array of lengths.
    """
    mask = np.random.random(len(seq_lengths)) <= p
    the_chosen = []
    the_lengths = []
    curr_idx = 0
    for i in range(len(seq_lengths)):
        curr_len = seq_lengths[i]
        if mask[i]:
            the_chosen.append(feature_matrix[curr_idx:curr_idx+curr_len])
            the_lengths.append(curr_len)
        curr_idx += curr_len
    return the_chosen, np.array(the_lengths)


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
    subsample = model_args.get('subsample', 1.0)  # fraction of data to use
    m_type = model_args.get('m_type', 'gmm').lower()
    print('Initializing `%s` model with args:\n%r' % (m_type, _args))
    print('subsample p: %.2f' % subsample)

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
    the_chosen, seq_lengths = \
        subsample_feature_matrix(feature_matrix, seq_lengths, subsample)
    feature_matrix = np.concatenate(the_chosen, axis=0)
    log_probs = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(feature_matrix, seq_lengths)
        for feature_singleseq in the_chosen:
            try:
                log_probs.append(model.score(feature_singleseq))
            except ValueError:
                print('[-] nan alert! Dropping this model.')
                return None, None
    model_stats = (np.mean(log_probs), np.std(log_probs))
    joblib.dump(model, model_file)
    return model, model_stats


def populate_model_dir(h5_dir, model_dir, all_model_args, n_features=None):
    """Populate the model directory with trained models corresponding
    to each h5 file in h5_dir.

    Parameters
    ----------
    h5_dir: str
        directory for h5 files

    model_dir: str
        directory to store models

    all_model_args: list
        settings for each model in the ensemble
        (if an ensemble is not desired, should be length 1)

    n_features: int
        desired size of feature dimension (set to None if no adjustment should be made)
    """
    try:
        os.makedirs(model_dir)
    except OSError:
        if rm_rf(model_dir, '------------------\n'
                            'ATTENTION REQUIRED\n'
                            '------------------\n'
                            '%s already exists! '
                            'Okay to clear it? (T/F)' % model_dir):
            os.makedirs(model_dir)
    stats = {}
    for dirpath, dirnames, filenames in os.walk(h5_dir):
        for filename in filenames:
            if filename.endswith('.h5'):
                activity_h5 = os.path.join(h5_dir, filename)

                print('%s' % filename)
                print('--------------')
                for i, model_args in enumerate(all_model_args):
                    print('model %d:' % i)
                    activity_pkl = filename[:-3] + str(i) + '.pkl'
                    model_file = os.path.join(model_dir, activity_pkl)
                    _, model_stats = learn_params(activity_h5, model_file,
                                                  model_args, n_features)
                    if model_stats is not None:
                        stats[activity_pkl] = model_stats
                print('')
    joblib.dump(stats, os.path.join(model_dir, 'stats.pkl'))
