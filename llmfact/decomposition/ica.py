from sklearn.decomposition import fastica
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_random_state
from joblib import delayed, Parallel
from scipy.stats import scoreatpercentile
from tqdm import trange
import numpy as np
import torch

from operator import itemgetter
import warnings
warnings.filterwarnings("ignore")

class CanICA:
    def __init__(self,
                 n_components=20,
                 n_jobs=-1,
                 threshold="auto",
                 random_state=666):
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.mixing_ = None
        self.original_mixing_ = None
        self.normal_mixing_ = None
        self.components_ = None
        self.threshold = threshold

    def fit(self, signals, n_iter=5, max_iter=300, norm=True):
        if type(signals) == str:
            if signals.endswith(".npy"):
                signals = np.load(signals)
            else:
                raise ValueError("signals must be numpy.ndarray."
                                 "You provided {}".format(signals))

        if norm:
            S = np.sqrt(np.sum(signals ** 2, axis=1))
            S[S == 0] = 1
            signals /= S[:, np.newaxis]
        components_, variance_, _ = randomized_svd(signals.T,
                                                   n_components=self.n_components,
                                                   transpose=True,
                                                   random_state=self.random_state,
                                                   n_iter=n_iter)
        if norm:
            signals *= S[:, np.newaxis]
        self.components_ = components_.T
        seeds = check_random_state(self.random_state).randint(np.iinfo(np.int32).max, size=10)

        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(fastica)(components_.astype(np.float64), whiten='unit-variance', fun='cube', random_state=seed, max_iter=max_iter)
            for seed in seeds)
        ica_maps_gen_ = (result[2].T for result in results)

        ica_maps_and_sparsities = ((ica_map,
                                    np.sum(np.abs(ica_map), axis=1).max())
                                   for ica_map in ica_maps_gen_)

        ica_maps, _ = min(ica_maps_and_sparsities, key=itemgetter(-1))

        self.original_mixing_ = np.array(ica_maps)

        mean = np.mean(ica_maps, axis=1, keepdims=True)
        std = np.std(ica_maps, axis=1, keepdims=True)

        z_score_ica_maps = (ica_maps - mean) / std

        self.normal_mixing_ = z_score_ica_maps

        # auto thresholding
        if self.threshold == "auto":
            abs_ica_maps = np.abs(ica_maps)
            percentile = 100. - (100. / len(ica_maps))
            threshold = scoreatpercentile(abs_ica_maps, percentile)
            ica_maps[abs_ica_maps < threshold] = 0.
        else:
            abs_ica_maps = np.abs(ica_maps)
            ica_maps[abs_ica_maps < self.threshold] = 0

        self.mixing_ = ica_maps.astype(self.components_.dtype)
        for component in self.mixing_:
            if component.max() < -component.min():
                component *= -1

        return self

if __name__ == '__main__':
    signals = np.random.rand(100, 2854)
    ica = CanICA()
    ica.fit(signals)
    print(ica.components_.shape)