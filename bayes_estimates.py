import numpy as np
import scipy.misc as sm
import scipy.special as sp

from scipy.optimize import minimize


class BayesEstimates(object):
    """A class of functions to perform Bayesian estimation."""

    def __init__(self, low_init=1, high_init=150, x_var=None, y_var=None):
        """Instantiation of the class.

        :param low_init: A guess at the alpha.
        :param high_init: A guess at the beta.
        :param x_var: The x-variable to run.
        :param y_var: The y-variable to run.
        :return: None
        """
        self.low_init = low_init
        self.high_init = high_init
        self.x_var = x_var
        self.y_var = y_var

    def binomial_beta(self, alpha, beta, n, k):
        """A function to calculate the binomial beta distribution.

        :param alpha: The alpha parameter.
        :param beta: The beta parameter.
        :param n: The N parameter.
        :param k: The K parameter.
        :return: A vector of binomial-beta transformed results.
        """
        choices = sm.comb(n, k)
        top = sp.betaln(k+alpha, n-k+beta)
        top = top.replace(np.inf, np.nan)
        top.fillna(0, inplace=True)
        bottom = sp.betaln(n, k)
        bottom = bottom.replace(np.inf, np.nan)
        bottom.fillna(0, inplace=True)
        result = np.log(choices) + top - bottom
        result = result.replace([np.inf, -np.inf], np.nan)
        result.fillna(0, inplace=True)
        return result

    def compute_ll_min(self, params):
        """A function to minimize the log-likelihood.

        :param params: A list of the alpha and beta parameters.
        :return: The minimum log-likelihood.
        """
        x = self.x_var
        y = self.y_var
        alpha = params[0]
        beta = params[1]
        return -abs(sum(self.binomial_beta(alpha, beta, x, y)))

    def run_mle(self):
        """A function to run the maximum likelihood estimation.

        :return: An estimated alpha and beta.
        """
        init_params = [[self.low_init, self.high_init]]
        min_results = minimize(self.compute_ll_min, init_params, method='nelder-mead')
        est_alpha = min_results.x[0]
        est_beta = min_results.x[1]
        print("Alpha: {alpha}, Beta: {beta}".format(alpha=est_alpha, beta=est_beta))
        return est_alpha, est_beta
