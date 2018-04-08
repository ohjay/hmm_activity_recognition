#!/usr/bin/env python

from hmmlearn import hmm

# TODO read from config file
N_COMPONENTS = 3

# estimate model parameters from observed features
# TODO get features first
model = hmm.GaussianHMM(n_components=N_COMPONENTS, covariance_type='full')
