"""Baseline comparison models:
 (i)  M-state Hidden Markov Model (composite random walk) with emissions in x-space;
 (ii) Causal MaxEnt model with coupling in LINEAR space ℓ_t ℓ_{t-τ}
      (the naive sufficient statistic), compared against the scale-free log statistic x_t x_{t-τ}.
All densities are expressed with respect to dx (x = ln ℓ/ℓ_min) so log-likelihoods are directly comparable.
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import logsumexp, roots_legendre
from maxent_levy import pl_logZ


# ----------------------------------------------------------------------------------
# Hidden Markov Model (HMM)
# ----------------------------------------------------------------------------------

def emis_logpdf(x, fam, par, L):
    """Log density (with respect to dx) of one emission.
    fam='pl' : truncated power law, par = h   (density ∝ e^{h x})
    fam='exp': ℓ/ℓ_min truncated exponential with rate e^{par}  (par = log λ)
    """
    if fam == "pl":
        return par * x - pl_logZ(np.array([par]), L)[0]
    lam = np.exp(par)
    ell = np.exp(x)
    # density in ℓ: λ e^{-λℓ} / (e^{-λ} - e^{-λ e^L});  Jacobian dℓ = ℓ dx
    lognorm = -lam + np.log(-np.expm1(-lam * (np.exp(L) - 1.0)))
    return np.log(lam) - lam * ell - lognorm + x


@njit(cache=True)
def _forward_scaled(logB, A, pi):
    """Scaled forward algorithm; returns cumulative log P(x_1..x_t) for each t."""
    T, M = logB.shape
    out = np.empty(T)
    alpha = np.empty(M)
    new = np.empty(M)
    acc = 0.0
    for t in range(T):
        bmax = logB[t].max()
        for j in range(M):
            if t == 0:
                s = pi[j]
            else:
                s = 0.0
                for i in range(M):
                    s += alpha[i] * A[i, j]
            new[j] = s * np.exp(logB[t, j] - bmax)
        c = new.sum()
        if not (c > 0.0) or not np.isfinite(bmax):
            for k in range(t, T):
                out[k] = -np.inf
            return out
        for j in range(M):
            alpha[j] = new[j] / c
        acc += np.log(c) + bmax
        out[t] = acc
    return out


def hmm_forward(logB, logA, logpi):
    """Return cumulative log P(x_1..x_t) for each t (array of length T)."""
    return _forward_scaled(np.ascontiguousarray(logB), np.exp(logA), np.exp(logpi))


def _unpack(theta, M):
    """Emissions + transition matrix: for row i, free logits for states j ≠ i
    and logit 0 for self-transition i→i."""
    em = theta[:M]
    logits = theta[M:].reshape(M, M - 1)
    A = np.empty((M, M))
    for i in range(M):
        row = np.zeros(M)
        others = [j for j in range(M) if j != i]
        row[others] = logits[i]
        A[i] = row - logsumexp(row)
    return em, A


def fit_hmm(seqs_x, L, M=2, fam="pl", Kcond=0, n_starts=6, rng=None):
    """MLE of an M-state HMM; NLL conditioned on the first Kcond steps of each sequence
    (for comparison against causal MaxEnt of order ≤ Kcond on the exact same steps)."""
    rng = np.random.default_rng(0) if rng is None else rng
    seqs_x = [np.asarray(s, dtype=float) for s in seqs_x]

    def nll(theta):
        em, logA = _unpack(theta, M)
        # initial distribution = stationary distribution of the hidden chain
        A = np.exp(logA)
        w, v = np.linalg.eig(A.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1))])
        pi = np.abs(pi) / np.abs(pi.sum())
        logpi = np.log(pi + 1e-300)
        tot = 0.0
        for x in seqs_x:
            logB = np.column_stack([emis_logpdf(x, fam, em[m], L) for m in range(M)])
            cum = hmm_forward(logB, logA, logpi)
            tot += cum[-1] - (cum[Kcond - 1] if Kcond > 0 else 0.0)
        return -tot

    best = None
    allx = np.concatenate(seqs_x)
    for s in range(n_starts):
        if fam == "pl":
            em0 = np.sort(rng.uniform(-3, 0.5, M))
        else:
            em0 = np.sort(rng.uniform(-np.log(np.exp(allx).max()), 0.5, M))
        th0 = np.concatenate([em0, rng.uniform(-4, -1, M * (M - 1))])
        bnds = [(-60.0, 60.0)] * M + [(-25.0, 25.0)] * (M * (M - 1))
        def nll_safe(th):
            v = nll(th)
            return v if np.isfinite(v) else 1e300
        r = minimize(nll_safe, th0, method="L-BFGS-B", bounds=bnds)
        if best is None or r.fun < best.fun:
            best = r
    em, logA = _unpack(best.x, M)
    k = M + M * (M - 1)
    N = sum(len(x) - Kcond for x in seqs_x)
    return dict(nll=best.fun, k=k, N=N, aic=2 * best.fun + 2 * k, bic=2 * best.fun + k * np.log(N),
                em=em, A=np.exp(logA))


# ----------------------------------------------------------------------------------
# Linear-space coupling model: p(x_t|·) ∝ exp(h x_t + φ_t u_t),
# u = ℓ/ℓ_max = e^{x-L} ∈ (0,1],  φ_t = Σ_τ G_τ u_{t-τ}
# ----------------------------------------------------------------------------------

class LinearCoupling:
    def __init__(self, L, n=128):
        t, w = roots_legendre(n)
        self.L = L
        self.x = (t + 1) * L / 2
        self.u = np.exp(self.x - L)
        self.logw = np.log(w * L / 2)

    def _stats(self, h, phi):
        a = h * self.x[None, :] + np.outer(phi, self.u) + self.logw
        amax = a.max(axis=1, keepdims=True)
        p = np.exp(a - amax)
        s = p.sum(axis=1, keepdims=True)
        logZ = (amax + np.log(s))[:, 0]
        p /= s
        ex, eu = p @ self.x, p @ self.u
        vxx = p @ self.x**2 - ex**2
        vuu = p @ self.u**2 - eu**2
        cxu = p @ (self.x * self.u) - ex * eu
        return logZ, ex, eu, vxx, vuu, cxu

    def design(self, seqs_x, Kmax):
        ys, Us = [], []
        for x in seqs_x:
            x = np.asarray(x, dtype=float)
            if len(x) <= Kmax:
                continue
            idx = np.arange(Kmax, len(x))
            U = np.column_stack([np.exp(x[idx - tau] - self.L) for tau in range(1, Kmax + 1)]) if Kmax else np.zeros((len(idx), 0))
            ys.append(x[idx]); Us.append(U)
        return np.concatenate(ys), np.vstack(Us)

    def fit(self, y, U, K, maxiter=100):
        Uk = U[:, :K]
        uy = np.exp(y - self.L)
        p = np.zeros(K + 1)
        p[0] = -1.0

        def terms(p):
            h, G = p[0], p[1:]
            phi = Uk @ G if K else np.zeros(len(y))
            logZ, ex, eu, vxx, vuu, cxu = self._stats(h, phi)
            nll = np.sum(-h * y - phi * uy + logZ)
            gh = np.sum(ex - y)
            gG = Uk.T @ (eu - uy) if K else np.zeros(0)
            g = np.concatenate([[gh], gG])
            H = np.empty((K + 1, K + 1))
            H[0, 0] = vxx.sum()
            if K:
                H[0, 1:] = H[1:, 0] = Uk.T @ cxu
                H[1:, 1:] = (Uk * vuu[:, None]).T @ Uk
            return nll, g, H

        f, g, H = terms(p)
        for _ in range(maxiter):
            step = np.linalg.solve(H + 1e-10 * np.eye(K + 1), g)
            dec = g @ step
            t = 1.0
            while True:
                fn, gn, Hn = terms(p - t * step)
                if np.isfinite(fn) and fn <= f - 1e-4 * t * dec:
                    break
                t *= 0.5
                if t < 1e-12:
                    break
            p, f, g, H = p - t * step, fn, gn, Hn
            if dec / 2 < 1e-10 * max(1, abs(f)):
                break
        return dict(params=p, nll=f, k=K + 1, N=len(y), aic=2 * f + 2 * (K + 1),
                    bic=2 * f + (K + 1) * np.log(len(y)))
