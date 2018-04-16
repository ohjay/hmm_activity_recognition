#!/usr/bin/env python

import numpy as np
from hmmlearn import hmm

# Initialize initial transition matrix as per the paper
TRANSMAT_PRIOR = np.array([[1/3, 1/3, 1/3, 0],
                           [0,   1/3, 1/3, 1/3],
                           [0,   0,   1/2, 1/2],
                           [0,   0,   0,   1]])

# estimate model parameters from observed features
# potential: http://larsmans.github.io/seqlearn/
def learn_params(n_components, transmat_prior=TRANSMAT_PRIOR):
    """Return an HMM model with learned parameters
       (transition and emission probabilities) 

       Parameters
       ----------
           n_components: int
               the number of states in the model

           transmat_prior: array-like, shape (n_components, n_components)
               prior transition matrix
    """
    model = hmm._BaseHMM(n_components=n_components,
                         transmat_prior=transmat_prior,
                         verbose=True)
    # TODO get feature matrix and sequence lengths vector
    model.fit(feature_matrix, sequence_lengths)
    return model

def classify_sequence(sequence):
    pass
