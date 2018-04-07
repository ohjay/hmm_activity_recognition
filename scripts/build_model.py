from hmmlearn import hmm

# TODO make config file (?)
N_COMPONENTS = 3

# estimate model parameters from observed features
# TODO get features first
model = hmm.GaussianHMM(n_components=N_COMPONENTS, covariance_type='full')
