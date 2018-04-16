#!/usr/bin/env python

import numpy as np
from hmmlearn import hmm

# 
N_COMPONENTS = 4

# Ininitialize initial transition matrix as per the paper
TRANSMAT_PRIOR = np.array([[1/3, 1/3, 1/3, 0],
                           [0,   1/3, 1/3, 1/3],
                           [0,   0,   1/2, 1/2],
                           [0,   0,   0,   1]])

# estimate model parameters from observed features
# potential: http://larsmans.github.io/seqlearn/
model = hmm._BaseHMM(n_components=N_COMPONENTS,
                     transmat_prior=TRANSMAT_PRIOR,
                     verbose=True)
