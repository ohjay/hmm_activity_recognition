#!/usr/bin/env python

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

# estimate model parameters from observed features
# potential: http://larsmans.github.io/seqlearn/
def learn_params(activity_h5, model_file, n_components,
                 transmat_prior=TRANSMAT_PRIOR):
    """ Save an HMM model (Gaussian emissions) with learned parameters
        (transition and emission probabilities) to model_file

        Parameters
        ----------
            activity_h5: string
                filepath for the h5 file containing the feature matrix
                and sequence length vector associated with the activity

            mode_file: string
                filepath to save file (should have .pkl file extension)

            n_components: int
                number of states in the model

            transmat_prior: array-like, shape (n_components, n_components)
                prior transition matrix
    """
    model = hmm.GaussianHMM(n_components=n_components,
                            transmat_prior=transmat_prior,
                            init_params = 't', verbose=True)
    feature_matrix, seq_lengths = load_features(activity_h5)
    model.fit(feature_matrix, seq_lengths)
    joblib.dump(model, model_file)
    return model

def populate_model_dir(h5_dir, model_dir, n_components,
                       transmat_prior=TRANSMAT_PRIOR):
    """ Populate the model directory with trained models corresponding
        to each h5 file in h5_dir

        Parameters
        ----------
            h5_dir: string
                directory for h5 files

            model_dir: string
                directory to store models

            n_components: int
                number of states for each model

            transmat_prior: array-like, shape (n_components, n_components)
                prior transition matrix for each model
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    for (dirpath, dirnames, filenames) in os.walk(h5_dir):
        for filename in filenames:
            if filename.endswith('.h5'):
                activity_h5 = os.path.join(h5_dir, filename)
                activity_pkl = filename[:-3] + '.pkl'
                model_file = os.path.join(model_dir, activity_pkl)
                learn_params(activity_h5, model_file, n_components,
                             transmat_prior)

if __name__ == '__main__':
    learn_params(sys.argv)
