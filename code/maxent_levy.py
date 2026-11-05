"""
maxent_levy — Causal maximum-entropy inference of memory in Lévy-walk step sequences.

Scale-invariant working variable:
    x_t = ln(ℓ_t / ℓ_min) ∈ [0, L],      L = ln(ℓ_max / ℓ_min)
The reference measure dx = dℓ/ℓ is invariant under rescaling of units; fixing <ln ℓ>
yields a linear field h·x, which in ℓ is a truncated power law p(ℓ) ∝ ℓ^{h-1} (μ = 1 - h).

Order-K causal (kinetic) MaxEnt chain:
    p(x_t | x_{t-1..t-K}) = exp(θ_t x_t - β x_t² - A(θ_t, β))
    θ_t = h + Σ_{τ=1..K} J_τ (x_{t-τ} - c)
with A the single-site log-partition function on the compact interval [0, L].
  * family='powerlaw'  : β = 0. A(θ) = ln[(e^{θL}-1)/θ], mean = Langevin response.
  * family='lognormal' : free β (additional constraint <(ln ℓ)²>), Gauss–Legendre quadrature.

The negative log-likelihood is strictly convex in the parameters, so damped Newton's method
converges to the unique global optimum in a few iterations.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.special import roots_legendre

# ----------------------------------------------------------------------------------
# Single-site functions, power-law family (β = 0): numerically stable closed forms
# ----------------------------------------------------------------------------------

_SMALL = 1e-4


def _log_expm1_over_u(u):
    """g(u) = ln[(e^u - 1)/u], numerically stable across all real u."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    small = np.abs(u) < _SMALL
    pos = (u >= _SMALL)
    neg = (u <= -_SMALL)
    us = u[small]
    out[small] = us / 2 + us**2 / 24 - us**4 / 2880
    up = u[pos]
    out[pos] = up + np.log(-np.expm1(-up)) - np.log(up)
    un = u[neg]
    out[neg] = np.log(-np.expm1(un)) - np.log(-un)
    return out


def langevin(y):
    """Langevin function 𝓛(y) = coth(y) - 1/y."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    small = np.abs(y) < 1e-3
    ys = y[small]
    out[small] = ys / 3 - ys**3 / 45 + 2 * ys**5 / 945
    yl = y[~small]
    out[~small] = 1.0 / np.tanh(yl) - 1.0 / yl
    return out


def langevin_prime(y):
    """Derivative 𝓛'(y) = 1/y² - 1/sinh²(y)."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    small = np.abs(y) < 1e-2
    ys = y[small]
    out[small] = 1 / 3 - ys**2 / 15 + 2 * ys**4 / 189
    yl = np.abs(y[~small])
    e = np.exp(-2 * yl)
    out[~small] = 1.0 / yl**2 - 4 * e / (1 - e) ** 2
    return out


def pl_logZ(theta, L):
    """Log-partition function ln ∫_0^L e^{θx} dx."""
    return np.log(L) + _log_expm1_over_u(np.asarray(theta) * L)


def pl_mean_var(theta, L):
    """Mean and variance of x under p(x) ∝ e^{θx} on [0, L] (Langevin response)."""
    y = np.asarray(theta) * L / 2
    m = L / 2 * (1 + langevin(y))
    v = L**2 / 4 * langevin_prime(y)
    return m, v


def pl_sample(theta, L, rng):
    """Sample x ∈ [0, L] with density ∝ e^{θx} (exact inverse-CDF, vectorized)."""
    theta = np.asarray(theta, dtype=float)
    u = rng.random(theta.shape)
    x = np.empty_like(theta)
    small = np.abs(theta * L) < 1e-8
    x[small] = u[small] * L
    pos = theta > 0
    pos &= ~small
    tp = theta[pos]
    x[pos] = L + np.log(u[pos] + (1 - u[pos]) * np.exp(-tp * L)) / tp
    neg = (theta < 0) & ~small
    tn = theta[neg]
    x[neg] = np.log1p(u[neg] * np.expm1(tn * L)) / tn
    return np.clip(x, 0.0, L)


# ----------------------------------------------------------------------------------
# Gauss–Legendre quadrature for the general case (β ≠ 0)
# ----------------------------------------------------------------------------------

class GLQuad:
    """Integrals ∫_0^L exp(θx - βx²) f(x) dx via Gauss–Legendre quadrature in log-space.

    For entire integrands such as exp(θx - βx²), convergence is geometric: with n=96 nodes
    accuracy is at machine precision for |θ|L, |β|L² ≲ 100.
    """

    def __init__(self, L, n=96):
        t, w = roots_legendre(n)
        self.L = L
        self.x = (t + 1) * L / 2
        self.logw = np.log(w * L / 2)

    def stats(self, theta, beta):
        """Return logZ, E[x], E[x²], Var x, Cov(x,x²), Var x² for each θ_t."""
        x = self.x
        a = np.outer(theta, x) - beta * x**2 + self.logw  # (N, n)
        amax = a.max(axis=1, keepdims=True)
        p = np.exp(a - amax)
        s = p.sum(axis=1, keepdims=True)
        logZ = (amax + np.log(s))[:, 0]
        p /= s
        m1 = p @ x
        m2 = p @ x**2
        m3 = p @ x**3
        m4 = p @ x**4
        return logZ, m1, m2, m2 - m1**2, m3 - m1 * m2, m4 - m2**2


# ----------------------------------------------------------------------------------
# Data transformation and design matrices
# ----------------------------------------------------------------------------------

@dataclass
class LogSpace:
    lmin: float
    lmax: float

    @property
    def L(self):
        return float(np.log(self.lmax / self.lmin))

    def to_x(self, ell):
        """Map ℓ → x ∈ [0, L], returning clipped x and the fraction of clipped flights."""
        ell = np.asarray(ell, dtype=float)
        x = np.log(ell / self.lmin)
        frac = np.mean((x < 0) | (x > self.L))
        return np.clip(x, 0.0, self.L), frac

    def to_ell(self, x):
        return self.lmin * np.exp(x)


def build_design(seqs, Kmax, c):
    """For each sequence x, extract predicted steps t = Kmax..len-1 (common rows for all K ≤ Kmax).

    Returns y (N,) and Z (N, Kmax+1) with columns [1, x_{t-1}-c, ..., x_{t-Kmax}-c].
    """
    ys, Zs = [], []
    for x in seqs:
        x = np.asarray(x, dtype=float)
        T = len(x)
        if T <= Kmax:
            continue
        idx = np.arange(Kmax, T)
        Z = np.ones((len(idx), Kmax + 1))
        for tau in range(1, Kmax + 1):
            Z[:, tau] = x[idx - tau] - c
        ys.append(x[idx])
        Zs.append(Z)
    return np.concatenate(ys), np.vstack(Zs)


# ----------------------------------------------------------------------------------
# Maximum-likelihood estimation (damped Newton on convex NLL)
# ----------------------------------------------------------------------------------

@dataclass
class FitResult:
    K: int
    family: str
    params: np.ndarray          # [h, J_1..J_K] (+ β if lognormal)
    nll: float                  # total NLL (without ridge penalty)
    N: int                      # number of predicted steps
    H: np.ndarray               # Hessian of NLL (= observed Fisher information)
    G: np.ndarray               # Σ score score^T (for TIC / sandwich covariance)
    converged: bool
    n_iter: int
    c: float
    L: float
    extra: dict = field(default_factory=dict)

    @property
    def k(self):
        return len(self.params)

    @property
    def h(self):
        return self.params[0]

    @property
    def J(self):
        return self.params[1:self.K + 1]

    @property
    def beta(self):
        return self.params[-1] if self.family == "lognormal" else 0.0

    @property
    def aic(self):
        return 2 * self.nll + 2 * self.k

    @property
    def bic(self):
        return 2 * self.nll + self.k * np.log(self.N)

    @property
    def tic(self):
        try:
            pen = np.trace(np.linalg.solve(self.H, self.G))
        except np.linalg.LinAlgError:
            pen = np.nan
        return 2 * self.nll + 2 * pen

    @property
    def se(self):
        return np.sqrt(np.diag(np.linalg.inv(self.H)))

    @property
    def se_sandwich(self):
        Hi = np.linalg.inv(self.H)
        return np.sqrt(np.diag(Hi @ self.G @ Hi))

    @property
    def h_uncentered(self):
        """Field referenced to x=0 (ℓ=ℓ_min): h' = h - c Σ J."""
        return self.h - self.c * np.sum(self.J)

    @property
    def mu_bare(self):
        """Single-site power-law exponent at the centering point, μ = 1 - h."""
        return 1.0 - self.h


def _terms(params, y, Z, K, family, L, quad):
    """NLL, gradient, Hessian, and per-observation score matrix."""
    w = params[:K + 1]
    Zk = Z[:, :K + 1]
    theta = Zk @ w
    if family == "powerlaw":
        logZ = pl_logZ(theta, L)
        m, v = pl_mean_var(theta, L)
        nll_i = -(theta * y) + logZ
        r = (m - y)                                # dNLL_i/dθ_i
        scores = Zk * r[:, None]
        g = scores.sum(axis=0)
        H = (Zk * v[:, None]).T @ Zk
        return nll_i.sum(), g, H, scores
    else:
        beta = params[K + 1]
        logZ, m1, m2, vx, cx, vx2 = quad.stats(theta, beta)
        nll_i = -(theta * y) + beta * y**2 + logZ
        r_th = m1 - y                              # dNLL_i/dθ
        r_be = y**2 - m2                           # dNLL_i/dβ
        scores = np.hstack([Zk * r_th[:, None], r_be[:, None]])
        g = scores.sum(axis=0)
        Hww = (Zk * vx[:, None]).T @ Zk
        Hwb = -(Zk * cx[:, None]).sum(axis=0)      # d²A/dθdβ = -Cov(x,x²)
        Hbb = vx2.sum()
        k = K + 2
        H = np.empty((k, k))
        H[:K + 1, :K + 1] = Hww
        H[:K + 1, K + 1] = Hwb
        H[K + 1, :K + 1] = Hwb
        H[K + 1, K + 1] = Hbb
        return nll_i.sum(), g, H, scores


def fit(y, Z, K, L, c, family="powerlaw", ridge=0.0, init=None, tol=1e-9,
        maxiter=200, quad=None):
    """MLE of the order-K causal chain on common rows y.

    ridge: L2 penalty ridge * Σ J_τ² applied to the couplings only.
    """
    k = K + 1 + (1 if family == "lognormal" else 0)
    if family == "lognormal" and quad is None:
        quad = GLQuad(L)
    if init is None:
        params = np.zeros(k)
        # initialize the field from the i.i.d. renewal fit (bisection on Langevin response)
        params[0] = _invert_mean(y.mean(), L)
    else:
        params = np.array(init, dtype=float)
    pen_mask = np.zeros(k)
    pen_mask[1:K + 1] = 1.0

    def obj(p):
        nll, g, H, S = _terms(p, y, Z, K, family, L, quad)
        nll_p = nll + ridge * np.sum(pen_mask * p**2)
        g_p = g + 2 * ridge * pen_mask * p
        H_p = H + 2 * ridge * np.diag(pen_mask)
        return nll_p, g_p, H_p, nll, H, S

    f, g, Hp, nll, H, S = obj(params)
    converged = False
    it = 0
    for it in range(1, maxiter + 1):
        try:
            step = np.linalg.solve(Hp + 1e-12 * np.eye(k), g)
        except np.linalg.LinAlgError:
            step = g / (np.abs(np.diag(Hp)) + 1e-8)
        dec = g @ step
        t = 1.0
        while True:
            pn = params - t * step
            fn, gn, Hpn, nlln, Hn, Sn = obj(pn)
            if np.isfinite(fn) and fn <= f - 1e-4 * t * dec:
                break
            t *= 0.5
            if t < 1e-12:
                break
        params, f, g, Hp, nll, H, S = pn, fn, gn, Hpn, nlln, Hn, Sn
        if dec / 2 < tol * max(1.0, abs(f)) or np.max(np.abs(g)) < 1e-8 * len(y):
            converged = True
            break
    G = S.T @ S
    return FitResult(K=K, family=family, params=params, nll=nll, N=len(y), H=H + 2 * ridge * np.diag(pen_mask),
                     G=G, converged=converged, n_iter=it, c=c, L=L)


def _invert_mean(target, L, lo=-200.0, hi=200.0):
    """Find θ such that E_θ[x] = target by bisection on the monotone Langevin response."""
    target = np.clip(target, 1e-6 * L, (1 - 1e-6) * L)
    for _ in range(200):
        mid = (lo + hi) / 2
        m, _ = pl_mean_var(np.array([mid / L]), L)
        if m[0] < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 / L


# ----------------------------------------------------------------------------------
# Memory order selection, fixed effects, covariates, and cross-validation
# ----------------------------------------------------------------------------------

def scan_orders(seqs_x, L, Kmax, family="powerlaw", ridge=0.0, c=None):
    """Fit K = 0..Kmax on the same predicted steps; returns list of FitResult."""
    if c is None:
        c = float(np.mean(np.concatenate([np.asarray(s) for s in seqs_x])))
    y, Z = build_design(seqs_x, Kmax, c)
    quad = GLQuad(L) if family == "lognormal" else None
    fits = []
    init = None
    for K in range(Kmax + 1):
        if init is not None:
            init = np.insert(init, K, 0.0) if family == "lognormal" else np.append(init, 0.0)
        r = fit(y, Z, K, L, c, family=family, ridge=ridge, init=init, quad=quad)
        fits.append(r)
        init = r.params.copy()
    return fits


def build_design_fe(seqs, Kmax, block=None):
    """Design for the fixed-effects model: one field h_i per sequence (or per contiguous block
    of `block` flights if given) and lags centered on each sequence's mean.
    Returns y, LAG (N, Kmax), and field index tid."""
    ys, lags, tid = [], [], []
    i = 0
    for x in seqs:
        x = np.asarray(x, dtype=float)
        if len(x) <= Kmax + 1:
            continue
        c = x.mean()
        idx = np.arange(Kmax, len(x))
        Lg = np.column_stack([x[idx - tau] - c for tau in range(1, Kmax + 1)]) if Kmax else np.zeros((len(idx), 0))
        ys.append(x[idx]); lags.append(Lg)
        if block is None:
            tid.append(np.full(len(idx), i)); i += 1
        else:
            b = np.arange(len(idx)) // block
            nb = b.max() + 1
            if nb > 1 and np.sum(b == nb - 1) < block // 2:   # merge trailing short block into previous
                b[b == nb - 1] = nb - 2
            tid.append(i + b); i += int(b.max()) + 1
    return np.concatenate(ys), np.vstack(lags), np.concatenate(tid)


def scan_orders_fe(seqs_x, L, Kmax, family="powerlaw", block=None):
    """Scan K = 0..Kmax with individual field h_i per sequence (quenched heterogeneity)
    and shared couplings J_τ. Returns list of FitResult with params = [h_1..h_n, J_1..J_K(, β)]
    and extra['n_seq']. With block=B, fields vary every B flights (non-stationarity control)."""
    y, LAG, tid = build_design_fe(seqs_x, Kmax, block=block)
    n = int(tid.max()) + 1
    OH = np.zeros((len(y), n))
    OH[np.arange(len(y)), tid] = 1.0
    quad = GLQuad(L) if family == "lognormal" else None
    h0 = np.array([_invert_mean(y[tid == i].mean(), L) for i in range(n)])
    fits = []
    prevJ = np.zeros(0)
    beta0 = []
    for K in range(Kmax + 1):
        Z = np.hstack([OH, LAG[:, :K]])
        init = np.concatenate([h0, prevJ, np.zeros(K - len(prevJ))])
        if family == "lognormal":
            init = np.concatenate([init, beta0 or [0.0]])
        r = fit(y, Z, Z.shape[1] - 1, L, 0.0, family=family, init=init, quad=quad)
        r.K = K
        r.extra["n_seq"] = n
        r.extra["J"] = r.params[n:n + K].copy()
        r.extra["se_J"] = r.se_sandwich[n:n + K].copy() if K else np.zeros(0)
        r.extra["h_i"] = r.params[:n].copy()
        fits.append(r)
        h0 = r.params[:n].copy()
        prevJ = r.params[n:n + K].copy()
        if family == "lognormal":
            beta0 = [r.params[-1]]
    return fits


def scan_orders_cov(seqs_x, covs, L, Kmax, family="powerlaw"):
    """Scan K = 0..Kmax with known exogenous covariates (external fields):
        θ_t = h + Σ_p g_p w_{t,p} + Σ_τ J_τ (x_{t-τ} - c)
    covs: list of arrays (len(x), p) aligned with seqs_x. Returns list of FitResult
    with params = [h, g_1..g_p, J_1..J_K(, β)] and extra['g'], extra['J']."""
    c = float(np.mean(np.concatenate([np.asarray(s) for s in seqs_x])))
    ys, Zs = [], []
    for x, w in zip(seqs_x, covs):
        x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float).reshape(len(x), -1)
        if len(x) <= Kmax:
            continue
        idx = np.arange(Kmax, len(x))
        Z = np.column_stack([np.ones(len(idx)), w[idx]] + [x[idx - tau] - c for tau in range(1, Kmax + 1)])
        ys.append(x[idx]); Zs.append(Z)
    y, Zall = np.concatenate(ys), np.vstack(Zs)
    p = Zall.shape[1] - 1 - Kmax
    quad = GLQuad(L) if family == "lognormal" else None
    fits, init = [], None
    for K in range(Kmax + 1):
        Z = Zall[:, :1 + p + K]
        if init is not None:
            init = np.insert(init, p + K, 0.0) if family == "lognormal" else np.append(init, 0.0)
        r = fit(y, Z, p + K, L, c, family=family, init=init, quad=quad)
        r.K = K
        r.extra["g"] = r.params[1:1 + p].copy()
        r.extra["J"] = r.params[1 + p:1 + p + K].copy()
        r.extra["se_J"] = r.se_sandwich[1 + p:1 + p + K].copy() if K else np.zeros(0)
        fits.append(r)
        init = r.params.copy()
    return fits


def select(fits, criterion="bic"):
    vals = np.array([getattr(f, criterion) for f in fits])
    return int(np.nanargmin(vals)), vals


def blocked_cv(seqs_x, L, Kmax, n_blocks=5, family="powerlaw", ridge=0.0, gap=None, c=None):
    """Contiguous hv-blocked cross-validation: each sequence is split into n_blocks folds;
    for fold b, the model is trained on all steps except fold b ± gap and evaluated on fold b.
    Returns total out-of-sample NLL for each K = 0..Kmax.
    """
    if gap is None:
        gap = Kmax
    if c is None:
        c = float(np.mean(np.concatenate([np.asarray(s) for s in seqs_x])))
    # predicted steps per sequence, with fold label
    ys, Zs, blk = [], [], []
    for x in seqs_x:
        x = np.asarray(x, dtype=float)
        y1, Z1 = build_design([x], Kmax, c)
        n = len(y1)
        if n == 0:
            continue
        b = np.minimum((np.arange(n) * n_blocks) // n, n_blocks - 1)
        ys.append(y1)
        Zs.append(Z1)
        blk.append(b)
    quad = GLQuad(L) if family == "lognormal" else None
    out = np.zeros(Kmax + 1)
    for b in range(n_blocks):
        tr_y, tr_Z, te_y, te_Z = [], [], [], []
        for y1, Z1, bb in zip(ys, Zs, blk):
            test = bb == b
            if not test.any():
                tr_y.append(y1); tr_Z.append(Z1)
                continue
            idx = np.where(test)[0]
            lo, hi = idx.min() - gap, idx.max() + gap
            train = (np.arange(len(y1)) < lo) | (np.arange(len(y1)) > hi)
            tr_y.append(y1[train]); tr_Z.append(Z1[train])
            te_y.append(y1[test]); te_Z.append(Z1[test])
        tr_y, tr_Z = np.concatenate(tr_y), np.vstack(tr_Z)
        te_y, te_Z = np.concatenate(te_y), np.vstack(te_Z)
        init = None
        for K in range(Kmax + 1):
            if init is not None:
                init = np.insert(init, K, 0.0) if family == "lognormal" else np.append(init, 0.0)
            r = fit(tr_y, tr_Z, K, L, c, family=family, ridge=ridge, init=init, quad=quad)
            init = r.params.copy()
            nll_te, _, _, _ = _terms(r.params, te_y, te_Z, K, family, L, quad)
            out[K] += nll_te
    return out


# ----------------------------------------------------------------------------------
# Simulation from the causal chain and exact K=1 stationary properties
# ----------------------------------------------------------------------------------

def simulate_model(h, J, L, c, T, rng, n_seq=1, burn=500, x0=None):
    """Sample n_seq chains of length T from the power-law causal MaxEnt model."""
    J = np.atleast_1d(np.asarray(J, dtype=float))
    K = len(J)
    tot = T + burn
    X = np.empty((n_seq, tot))
    X[:, :max(K, 1)] = (L / 2) if x0 is None else x0
    for t in range(K, tot):
        theta = h + sum(J[tau - 1] * (X[:, t - tau] - c) for tau in range(1, K + 1)) if K else np.full(n_seq, h)
        X[:, t] = pl_sample(np.asarray(theta, dtype=float), L, rng)
    return X[:, burn:]


def stationary_marginal_K1(h, J, L, c, n=2000):
    """Exact stationary density of the K=1 causal chain (reversible Markov chain):
    π(x) ∝ exp(h' x) Z(h' + J x),  h' = h - J c.
    """
    hp = h - J * c
    x = np.linspace(0, L, n)
    logpi = hp * x + pl_logZ(hp + J * x, L)
    logpi -= logpi.max()
    pi = np.exp(logpi)
    pi /= np.trapezoid(pi, x)
    return x, pi
