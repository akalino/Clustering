import random
import numpy as np
import pandas as pd


class ClaraClustering(object):
    """The clara clustering algorithm.

    Basically an iterative guessing version of k-mediods that makes things a lot faster
    for bigger data sets.
    """

    def __init__(self, max_iter=100000):
        """Class initialization.

        :param max_iter: The default number of max iterations
        """
        self.max_iter = max_iter
        self.dist_cache = dict()

    def clara(self, _df, _k, _fn):
        """The main clara clustering iterative algorithm.

        :param _df: Input data frame.
        :param _k: Number of medoids.
        :param _fn: The distance function to use.
        :return: The minimized cost, the best medoid choices and the final configuration.
        """
        size = len(_df)
        if size > 100000:
            niter = 1000
            runs = 1
        else:
            niter = self.max_iter
            runs = 5

        min_avg_cost = np.inf
        best_choices = []
        best_results = {}

        for j in range(runs):
            sampling_idx = random.sample([i for i in range(size)], (40+_k*2))
            sampling_data = []
            for idx in sampling_idx:
                sampling_data.append(_df.iloc[idx])

            sampled_df = pd.DataFrame(sampling_data)
            pre_cost, pre_choice, pre_medoids = self.k_medoids(sampled_df, _k, _fn, niter)
            tmp_avg_cost, tmp_medoids = self.average_cost(_df, _fn, pre_choice)
            if tmp_avg_cost <= min_avg_cost:
                min_avg_cost = tmp_avg_cost
                best_choices = list(pre_choice)
                best_results = dict(tmp_medoids)

        return min_avg_cost, best_choices, best_results

    def k_medoids(self, _df, _k, _fn, _niter):
        """The original k-mediods algorithm.

        :param _df: Input data frame.
        :param _k: Number of medoids.
        :param _fn: The distance function to use.
        :param _niter: The number of iterations.
        :return: Cluster label.

        Pseudo-code for the k-mediods algorithm.
        1. Sample k of the n data points as the medoids.
        2. Associate each data point to the closest medoid.
        3. While the cost of the data point space configuration is decreasing.
            1. For each medoid m and each non-medoid point o:
                1. Swap m and o, recompute cost.
                2. If global cost increased, swap back.
        """
        print('K-medoids starting')
        # Do some smarter setting of initial cost configuration
        pc1, medoids_sample = self.cheat_at_sampling(_df, _k, _fn, 17)
        prior_cost, medoids = self.compute_cost(_df, _fn, medoids_sample)
        current_cost = prior_cost
        iter_count = 0
        best_choices = []
        best_results = {}

        print('Running with {m} iterations'.format(m=_niter))
        while iter_count < _niter:
            for m in medoids:
                clust_iter = 0
                for item in medoids[m]:
                    if item != m:
                        idx = medoids_sample.index(m)
                        swap_temp = medoids_sample[idx]
                        medoids_sample[idx] = item
                        tmp_cost, tmp_medoids = self.compute_cost(_df, _fn, medoids_sample, True)
                        if (tmp_cost < current_cost) & (clust_iter < 1):
                            best_choices = list(medoids_sample)
                            best_results = dict(tmp_medoids)
                            current_cost = tmp_cost
                            clust_iter += 1
                        else:
                            best_choices = best_choices
                            best_results = best_results
                            current_cost = current_cost
                        medoids_sample[idx] = swap_temp

            iter_count += 1
            if best_choices == medoids_sample:
                print('Best configuration found!')
                break

            if current_cost <= prior_cost:
                prior_cost = current_cost
                medoids = best_results
                medoids_sample = best_choices

        return current_cost, best_choices, best_results

    def compute_cost(self, _df, _fn, _cur_choice, cache_on=True):
        """A function to compute the configuration cost.

        :param _df: The input data frame.
        :param _fn: The distance function.
        :param _cur_choice: The current set of medoid choices.
        :param cache_on: Binary flag to turn caching.
        :return: The total configuration cost, the mediods.
        """
        size = len(_df)
        total_cost = 0.0
        medoids = {}
        for idx in _cur_choice:
            medoids[idx] = []

        for i in range(size):
            choice = -1
            min_cost = np.inf
            for m in medoids:
                if cache_on:
                    tmp = self.dist_cache.get((m, i), None)

                if not cache_on or tmp is None:
                    if _fn == 'manhattan':
                        tmp = self.manhattan_distance(_df.iloc[m], _df.iloc[i])
                    elif _fn == 'cosine':
                        tmp = self.cosine_distance(_df.iloc[m], _df.iloc[i])
                    elif _fn == 'euclidean':
                        tmp = self.euclidean_distance(_df.iloc[m], _df.iloc[i])
                    elif _fn == 'fast_euclidean':
                        tmp = self.fast_euclidean(_df.iloc[m], _df.iloc[i])
                    else:
                        print('You need to input a distance function.')

                if cache_on:
                    self.dist_cache[(m, i)] = tmp

                if tmp < min_cost:
                    choice = m
                    min_cost = tmp

            medoids[choice].append(i)
            total_cost += min_cost

        return total_cost, medoids

    def average_cost(self, _df, _fn, _cur_choice):
        """A function to compute the average cost.

        :param _df: The input data frame.
        :param _fn: The distance function.
        :param _cur_choice: The current medoid candidates.
        :return: The average cost, the new medoids.
        """
        _tc, _m = self.compute_cost(_df, _fn, _cur_choice)
        avg_cost = _tc / len(_m)
        return avg_cost, _m

    def cheat_at_sampling(self, _df, _k, _fn, _nsamp):
        """A function to cheat at sampling for speed ups.

        :param _df: The input data frame.
        :param _k: The number of mediods.
        :param _fn: The distance function.
        :param _nsamp: The number of samples.
        :return: The best score, the medoids.
        """
        size = len(_df)
        score_holder = []
        medoid_holder = []
        for _ in range(_nsamp):
            medoids_sample = random.sample([i for i in range(size)], _k)
            prior_cost, medoids = self.compute_cost(_df, _fn, medoids_sample, True)
            score_holder.append(prior_cost)
            medoid_holder.append(medoids)

        idx = score_holder.index(min(score_holder))
        ms = medoid_holder[idx].keys()
        return score_holder[idx], ms

    def euclidean_distance(self, v1, v2):
        """Slow function for euclidean distance.

        :param v1: The first vector.
        :param v2: The second vector.
        :return: The euclidean distance between v1 and v2.
        """
        dist = 0
        for a1, a2 in zip(v1, v2):
            dist += abs(a1 - a2)**2
        return dist

    def fast_euclidean(self, v1, v2):
        """Faster function for euclidean distance.

        :param v1: The first vector.
        :param v2: The second vector.
        :return: The euclidean distance between v1 and v2.
        """
        return np.linalg.norm(v1 - v2)

    def manhattan_distance(self, v1, v2):
        """Function for manhattan distance.

        :param v1: The first vector.
        :param v2: The second vector.
        :return: The manhattan distance between v1 and v2.
        """
        dist = 0
        for a1, a2 in zip(v1, v2):
            dist += abs(a1 - a2)
        return dist

    def cosine_distance(self, v1, v2):
        """Function for cosine distance.

        :param v1: The first vector.
        :param v2: The second vector.
        :return: The cosine distance between v1 and v2.
        """
        xx, yy, xy = 0, 0, 0
        for a1, a2 in zip(v1, v2):
            xx += a1*a1
            yy += a2*a2
            xy += a1*a2
        return float(xy) / np.sqrt(xx*yy)
