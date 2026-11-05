"""Joint path-entropy (Jaynes / Maximum Caliber) model of order K=1:

    P(x_1..x_T) = exp( h Σ_t x_t + J Σ_t x_t x_{t+1} ) / Z_T ,   x_t ∈ [0, L]

A 1D chain of continuous spins with nearest-neighbor interactions in time: Z_T is evaluated
exactly via the transfer operator discretized on Gauss–Legendre nodes (geometric convergence).
The bulk process is a reversible Markov chain with Doob h-transformed kernel
p(y|x) = K(x,y) ψ(y) / (λ ψ(x)), sharing the same sufficient statistics as the causal chain
p(y|x) ∝ e^{(h+Jx) y}.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_legendre


class JointK1:
    def __init__(self, L, n=96):
        t, w = roots_legendre(n)
        self.L = L
        self.x = (t + 1) * L / 2
        self.w = w * L / 2

    def _spectrum(self, h, J):
        x, sw = self.x, np.sqrt(self.w)
        M = sw[:, None] * np.exp(h * (x[:, None] + x[None, :]) / 2 + J * np.outer(x, x)) * sw[None, :]
        v = sw * np.exp(h * x / 2)
        lam, U = np.linalg.eigh(M)
        return lam, U, v

    def logZ(self, h, J, T):
        lam, U, v = self._spectrum(h, J)
        lmax = lam[-1]
        a = (U.T @ v) ** 2
        r = lam / lmax
        s = np.sum(a * np.sign(r) ** (T - 1) * np.abs(r) ** (T - 1))
        return (T - 1) * np.log(lmax) + np.log(s)

    def nll(self, p, seqs):
        h, J = p
        tot = 0.0
        for x in seqs:
            tot -= h * x.sum() + J * np.sum(x[1:] * x[:-1]) - self.logZ(h, J, len(x))
        return tot

    def fit(self, seqs, init=(-0.5, 0.0)):
        seqs = [np.asarray(s, dtype=float) for s in seqs]
        r = minimize(self.nll, np.array(init), args=(seqs,), method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-7, maxiter=4000))
        r = minimize(self.nll, r.x, args=(seqs,), method="BFGS")
        N = sum(len(s) for s in seqs)
        return dict(params=r.x, nll=r.fun, aic=2 * r.fun + 4, bic=2 * r.fun + 2 * np.log(N))

    def sample(self, h, J, T, rng, n_grid=4000):
        """Sample the stationary bulk chain via Nyström interpolation and inverse-CDF on a fine grid."""
        lam, U, _ = self._spectrum(h, J)
        psi_nodes = U[:, -1] / np.sqrt(self.w)        # Perron eigenfunction at quadrature nodes
        psi_nodes *= np.sign(psi_nodes.sum())
        g = np.linspace(0, self.L, n_grid)
        # Nyström extension: ψ(y) = (1/λ) Σ_j w_j K(y, x_j) ψ(x_j)
        Kyx = np.exp(h * (g[:, None] + self.x[None, :]) / 2 + J * np.outer(g, self.x))
        psi = (Kyx @ (self.w * psi_nodes)) / lam[-1]
        # stationary marginal ∝ ψ(x)² ; transition kernel p(y|x) ∝ K(x,y) ψ(y)
        logpsi = np.log(np.maximum(psi, 1e-300))
        out = np.empty(T)
        pi = psi**2
        cdf = np.cumsum(pi); cdf /= cdf[-1]
        xcur = g[np.searchsorted(cdf, rng.random())]
        for t in range(T):
            out[t] = xcur
            lp = h * g / 2 + J * xcur * g + logpsi
            p = np.exp(lp - lp.max())
            cdf = np.cumsum(p); cdf /= cdf[-1]
            xcur = g[min(np.searchsorted(cdf, rng.random()), n_grid - 1)] + (rng.random() - 0.5) * (g[1] - g[0])
            xcur = min(max(xcur, 0.0), self.L)
        return out
