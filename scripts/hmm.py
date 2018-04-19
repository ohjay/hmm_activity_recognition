import numpy as np
from collections import deque
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn import cluster
from sklearn.mixture import gmm
from skleran.preprocessing import normalize
from hmmlearn import _hmmc

class GaussianHMM:
    """Hidden Markov Model with Gaussian emissions.
       Created as an exercise and uses code from the
       hmmlearn library.

       The Viterbi, Forward, and Backward algorithms
       are custom implementations.

    Parameters
    ----------
    n_components: int
        the number of states (hidden) in the hmm

    startprob_prior: array, shape (n_components, )
        prior distribution (state probabilities)

    transmat_prior: array, shape(n_components, n_components)
        prior transition matrix

    n_iter: int
        maximum number of iterations to perform

    tolerance: float
        tolerance for EM (EM will stop when the log-likelihood
        is within the tolerance)

    verbose: bool
        whether to print convergence reports to stderr

    params: string
        controls which parameters are updated when training
        's': startprob
        't': transmat
        'm': means
        'c': covariances

        defaults to all

    init_params: string
        controls which parameters are already supplied for
        training
        's': prob_prior
        't': transmat_prior
        'm': means_prior
        'c': covars_prior

        if a parameter is unsupplied, it defaults to a uniform prior

    Attributes
    ----------
    startprob_: array, shape (n_components, )
        initial state distribution

    transmat_: array, shape (n_components, n_components)
        transition matrix
    """
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=0.01, covars_weight=1,
                 min_covar=0.001, covariance_type='diag',
                 n_iter=10, tolerance=0.01, verbose=False,
                 params = 'stmc', init_params=''):
        self.n_components = n_components
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.min_covar = min_covar
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.verbose = verbose
        self.params = params
        self.init_params = init_params
    
    def fit(self, X, lengths=None):
        """Infer the model parameters
           (i.e., initial distribution and transition matrix)
           using EM algorithm

        Parameters
        ----------
        X: array-like, shape(n_samples, n_features)
            feature matrix --- each row is a feature vector
            corresponding to a sample

        lengths: array-like of ints, shape(n_sequences, )
            lengths of the sequences in X; each sequence
            is an enumeration of related samples
        """
        X = check_array(X)
        self._init_params(X, lengths)
        self._check()
        history = deque(maxlen=2)

        for _ in range(self.n_iter):
            stats = self._init_m_stats()
            cur_logprob = 0
            for i, j in _gen_sequences(X, lengths):
                framelogprob = self._compute_log_likelihood(X[i:j])
                logprob, fwdlattice = self._forward_pass(framelogprob)
                cur_logprob += logprob
                bwdlattice = self._backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_m_statistics(
                    stats, X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice)
            self._m_step

            # Check convergence tolerance
            # TODO verbose reports
            history.append(cur_logprob)
            if len(history) == 2 and history[1] - history[0] < self.tolerance:
                break

        return self

    def score(self, X, lengths=None):
        """Returns the log probability of X under the model

        Parameters
        ----------
        X: array-like, shape(n_samples, n_features)
            feature matrix --- each row is a feature vector
            corresponding to a sample
 
        lengths: array-like of ints, shape(n_sequences, )
            lengths of the sequences in X; each sequence
            is an enumeration of related samples
        """
        check_is_fitted(self, 'startprob_')
        self._check()
        X = check_array(X)

        logprob = 0
        for i, j in _gen_sequences(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij = self._forward_pass(framelogprob)[0]
            logprob += logprobij
        return logprob

    def _viterbi_pass(self, framelogprob):
        # TODO

    def _forward_pass(self, framelogprob):
        # TODO

    def _backwardpass(self, framelogprob):
        # TODO

    def _compute_posteriors(self, fdwlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        log_normalize(log_gamma, axis=1)
        return np.exp(log_gamma)
    
    def _compute_log_likelihood(self, X):
        """Compute the log likelihood of feature matrix X"""
        return gmm.log_multivariate_normal_density(X, self.means_, self.covars_,
                                                   self.covariance_type)

    def _init_m_stats(self):
        """Initialize dict of stats for M step of EM algorithm"""
        # start[i] = P(first sample generated by ith state)
        # trans = transition matrix
        stats = {'n_samples': 0,
                'start': np.zeros(self.n_components),
                'trans': np.zeros((self.n_components, self.n_components)),
                'post': np.zeros(self.n_components),
                'obs': np.zeros((self.n_components, self.n_components)),
                'obs**2': np.zeros((self.n_components, self.n_features))}
        if self.covariance_type == 'full':
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                          self.n_features))
        return stats

    def _accumulate_m_stats(self, stats, X, framelogprob, posteriors,
                            fwdlattice, bwdlattice):
        stats['n_samples'] += 1
        if 's' in self.params:
            stats['start'] += posteriors[0]
        if 't' in self.params:
            n_samples, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            lneta = np.zeros((n_samples - 1, n_components, n_components))
            _hmmc._compute_lneta(n_samples, n_components, fwdlattice,
                                 np.log(self.transmat_),
                                 bwdlattice, framelogprob, lneta)
            stats['trans'] += np.exp(logsumexp(lneta, axis=0))


        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)

    def _m_step(self, stats):
        """Performs the M step of the EM Algorithm"""
        if 's' in self.params:
            startprob_ = self.startprob_prior - 1.0 + stats['start']
            self.startprob_ = np.where(self.startprob_ == 0.0,
                                       self.startprob_, startprob_)
            normalize(self.startprob_)
        if 't' in self.params:
            transmat_ = self.transmat_prior - 1.0 + stats['trans']
            self.transmat_ = np.where(self.transmat_ == 0.0,
                                      self.transmat_, transmat_)
            normalize(self.transmat_, axis=1)

        means_prior = self.means_prior
        means_weight = self.means_weight

        denom = stats['post'][:, np.newaxis]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type == 'diag':
                cv_num = (means_weight * meandiff ** 2
                          + stats['obs**2']
                          - 2 * self.means_ * stats['obs']
                          + self.means_ ** 2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = \
                    (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
            elif self.covariance_type == 'full':
                cv_num = np.empty((self.n_components, self.n_features,
                                  self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self.means_[c])

                    cv_num[c] = (means_weight * np.outer(meandiff[c],
                                                         meandiff[c])
                                 + stats['obs*obs.T'][c]
                                 - obsmean - obsmean.T
                                 + np.outer(self.means_[c], self.means_[c])
                                 * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                self._covars_ = ((covars_prior + cv_num) /
                                 (cvweight + stats['post'][:, None, None]))

    def _init_params(self, X, lengths):
        """Initialize parameters for training"""
        self.n_features = X.shape[1]
        uniform = 1.0 / self.n_components
        # start distribution
        if 's' in self.init_params:
            self.startprob_ = self.startprob_prior
        else:
            self.startprob_ = np.full(self.n_components, uniform)

        # transition matrix
        if 't' in self.init_params:
            self.transmat_ = self.transmat_prior
        else:
            self.transmat_ = np.full((self.n_samples, self.n_components), uniform)

        # means
        if 'm' in self.init_parms:
            self.means_ = self.means_prior
        else:
            kmeans = cluster.KMeans(n_clusters=n_components)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_

        # covariances
        if 'c' in self.init_params:
            self.covars_ = self.covars_prior
        else:
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if self.covariance_type == 'diag':
                self.covars_ = np.tile(np.diag(cv), (self.n_components, 1))
            else: # full covariance matrix
                self.covars_ = np.tile(cv, (n_components, 1, 1))

    def _check(self):
        """Sanity check for model parameters

        Raises
        ------
        ValueError
            if parameters (startprob, transmat) are invalid

        """
        if self.startprob_.shape[0] != self.n_components:
            raise ValueError("startprob_ length doesn't match n_components")
        if not np.allclose(self.startprob_.sum(), 1.0):
            raise ValueError("startprob_ doesn't sum to 1.0")

        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError("transmat_ is not size (n_components x n_components)")
        if not np.allclose(self.transmat_.sum(axis=1), 1.0):
            raise ValueError("rows of transmat_ don't sum to 1.0")

    def _gen_sequences(X, lengths):
        """Generator for sequence bounds in X"""
        if lengths is None:
            yield 0, len(X)
        else:
            ends = np.cumsum(lengths)
            starts = ends - lengths
            if ends[-1] > X.shape[0]:
                raise ValueError("lengths describes {0} samples and X \
                                  describes {1} samples".format(X.shape[0],
                                                                ends[-1]))
            for i in range(len(lengths)):
                yield starts[i], ends[i]


