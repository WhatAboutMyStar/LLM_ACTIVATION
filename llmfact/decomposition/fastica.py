import torch
import warnings

from llmfact.utils import setup_seed

def _logcosh(x):
    alpha = 1

    x *= alpha
    gx = torch.tanh(x)
    x = gx
    g_x = torch.empty(x.shape[0], dtype=x.dtype)
    # XXX compute in chunks to avoid extra allocation
    for i, gx_i in enumerate(gx):  # please don't vectorize.
        g_x[i] = (alpha * (1 - gx_i**2)).mean()
    return gx, g_x


def _exp(x):
    exp = torch.exp(-(x**2) / 2)
    gx = x * exp
    g_x = (1 - x**2) * exp
    return gx, g_x.mean(dim=-1)

def _cube(x):
    return x**3, (3 * x**2).mean(dim=-1)


def _sym_decorrelation(W):
    """Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    WT_W = W @ W.T
    s, u = torch.linalg.eigh(WT_W)

    tiny = torch.finfo(W.dtype).tiny
    s = torch.clamp(s, min=tiny)

    inv_sqrt_s = 1.0 / torch.sqrt(s)
    inv_sqrt_matrix = (u * inv_sqrt_s) @ u.T

    return inv_sqrt_matrix @ W

def _ica_par(X, tol, g, max_iter, w_init):
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in range(max_iter):
        gwtx, g_wtx = g(W @ X)
        W1 = _sym_decorrelation((gwtx @ X.T) / p_ - g_wtx[:, torch.newaxis] * W)
        lim = max(abs(abs(torch.einsum("ij,ij->i", W1, W)) - 1))
        W = W1
        if lim < tol:
            break
    else:
        warnings.warn(
            (
                "FastICA did not converge. Consider increasing "
                "tolerance or the maximum number of iterations."
            )
        )

    return W, ii + 1

def fastica(X,
            n_components=None,
            fun="cube",
            max_iter=300,
            tol=1e-04,
            random_state=None):
    est = FastICA(
        n_components=n_components,
        fun=fun,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    S = est.fit(X)

    K = est.whitening_

    return [K, est._unmixing, S]

class FastICA:
    def __init__(self,
                 n_components=None,
                 fun="cube",
                 max_iter=200,
                 tol=1e-4,
                 random_state=None):
        self.n_components = n_components
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state


    def fit(self, X):
        """
        :param X: array-like of shape (n_samples, n_features)
                  Training data, where `n_samples` is the number of samples
                  and `n_features` is the number of features.
        :return:
        S: ndarray of shape (n_samples, n_components) or None
           Sources matrix.
        """

        if self.random_state:
            setup_seed(self.random_state)

        if self.fun == "logcosh":
            g = _logcosh
        elif self.fun == "exp":
            g = _exp
        elif self.fun == "cube":
            g = _cube

        XT = X.T

        n_features, n_samples = XT.shape
        n_components = self.n_components

        if n_components is None:
            n_components = min(n_samples, n_features)
        if n_components > min(n_samples, n_features):
            n_components = min(n_samples, n_features)
            warnings.warn(
                "n_components is too large: it will be set to %s" % n_components
            )


        # whiten
        X_mean = XT.mean(dim=-1)
        XT -= X_mean[:, torch.newaxis]
        u, d = torch.linalg.svd(XT, full_matrices=False)[:2]

        u *= torch.sign(u[0])
        K = (u / d).T[:n_components]

        del u, d

        X1 = K @ XT

        X1 *= torch.sqrt(torch.tensor(n_samples))

        w_init = torch.randn((n_components, n_components), dtype=X1.dtype)

        w_init = w_init.to(X1.device)

        kwargs = {
            "tol": self.tol,
            "g": g,
            "max_iter": self.max_iter,
            "w_init": w_init
        }

        W, n_iter = _ica_par(X1, **kwargs)

        self.n_iter_ = n_iter

        S = ((W @ K) @ XT).T

        S_std = torch.std(S, dim=0, keepdim=True)
        S /= S_std
        W /= S_std.T

        self.components_ = (W @ K).detach().cpu()
        self.mean_ = X_mean.detach().cpu()
        self.whitening_ = K.detach().cpu()

        self.mixing_ = torch.linalg.pinv(self.components_).detach().cpu()
        self._unmixing = W.detach().cpu()

        return S