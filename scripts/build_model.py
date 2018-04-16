#!/usr/bin/env python

import numpy as np
from hmmlearn import hmm
import extract_features

# Initialize initial transition matrix as per the paper
TRANSMAT_PRIOR = np.array([[1/3, 1/3, 1/3, 0],
                           [0,   1/3, 1/3, 1/3],
                           [0,   0,   1/2, 1/2],
                           [0,   0,   0,   1]])

# estimate model parameters from observed features
# potential: http://larsmans.github.io/seqlearn/
def learn_params(activity_h5, n_components, transmat_prior=TRANSMAT_PRIOR):
    """Return an HMM model with learned parameters
       (transition and emission probabilities) 

       Parameters
       ----------
           activity_h4: string
               filepath for the h5 file containing the feature matrix
               and sequence length vector associated with the activity

           n_components: int
               the number of states in the model

           transmat_prior: array-like, shape (n_components, n_components)
               prior transition matrix
    """
    model = hmm._BaseHMM(n_components=n_components,
                         transmat_prior=transmat_prior,
                         verbose=True)
    feature_matrix, seq_lengths = extract_features.load_features(activity_h5)
    model.fit(feature_matrix, seq_lengths)
    return model

if __name__ == '__main__':
    learn_params(sys.argv)
