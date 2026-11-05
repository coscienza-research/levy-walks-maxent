"""Collective MaxEnt on step moduli with self and social delayed couplings.

Variables: x_i^t = ln(ell_i^t / ell_min) in [0, L] for unit i at a common frame t
(fixed clock). Causal (kinetic) MaxEnt model, conditionally independent across units
given the past of everybody:

    p(x_i^t | past) = exp(theta_i^t x_i^t - beta (x_i^t)^2) / Z(theta_i^t, beta)
    theta_i^t = h_i + g_t + sum_{tau<=Ks} a_tau (x_i^{t-tau} - c)
                          + sum_{tau<=Kc} J_tau (s_i^{t-tau} - c)

where s_i^{t-tau} is a social regressor:
  'global'  : leave-one-out mean of the other units present at t-tau
  'global+' : mean of all units at t-tau including i (the PRE choice)
  'local'   : mean of the k nearest other units (in space, at t-tau)
h_i are unit fields (fixed effects), g_t optional time fields (fixed effects absorbing any
common drive). The negative log-likelihood is convex; Newton with a sparse Hessian.

Also provided: a faithful re-implementation of the PRE (Agliari et al. 2024) pipeline adapted
to moduli: G_t = Lambda^{-1}(m_t), Gaussian AR on m_{t-tau}, least-squares AIC.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass, field
from maxent_levy import pl_logZ, pl_mean_var, GLQuad


# ----------------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------------

@dataclass
class Group:
    """One co-recorded group (a trial / a fish). X[i, t] = log step (NaN if missing);
    P[i, t, :] = position at frame t (optional; NaN if missing)."""
    X: np.ndarray
    P: np.ndarray | None = None
    box: float | None = None          # periodic box size (None = open space)
    name: str = ""


def _pairwise_dist(p, box):
    d = p[:, None, :] - p[None, :, :]
    if box is not None:
        d -= box * np.round(d / box)
    return np.hypot(d[..., 0], d[..., 1])


def social_regressors(g: Group, Kc: int, mode: str, k_nn: int = 3, n_min: int = 2):
    """Return S[tau-1, i, t] = social regressor of unit i built at frame t - tau
    (NaN where undefined). Built once per group for tau = 1..Kc."""
    U, T = g.X.shape
    S = np.full((Kc, U, T), np.nan)
    pres = ~np.isnan(g.X)
    Xz = np.where(pres, g.X, 0.0)
    n = pres.sum(axis=0)                        # units present per frame
    tot = Xz.sum(axis=0)
    if mode in ("global", "global+"):
        for s in range(T):
            if mode == "global":
                cnt = n[s] - pres[:, s]
                val = (tot[s] - Xz[:, s]) / np.maximum(cnt, 1)
                ok = cnt >= max(n_min - 1, 1)          # at least n_min - 1 other units
            else:
                cnt = np.full(U, n[s])
                val = np.full(U, tot[s] / max(n[s], 1))
                ok = cnt >= n_min
            for tau in range(1, Kc + 1):
                t = s + tau
                if t < T:
                    S[tau - 1, ok, t] = val[ok]
    elif mode == "local":
        assert g.P is not None
        for s in range(T):
            idx = np.where(pres[:, s] & ~np.isnan(g.P[:, s, 0]))[0]
            if len(idx) < 2:
                continue
            D = _pairwise_dist(g.P[idx, s, :], g.box)
            np.fill_diagonal(D, np.inf)
            k = min(k_nn, len(idx) - 1)
            nn = np.argpartition(D, k - 1, axis=1)[:, :k] if k < len(idx) - 1 else np.argsort(D, axis=1)[:, :k]
            val = g.X[idx[nn], s].mean(axis=1)
            for tau in range(1, Kc + 1):
                t = s + tau
                if t < T:
                    S[tau - 1, idx, t] = val
    else:
        raise ValueError(mode)
    return S


@dataclass
class Design:
    y: np.ndarray            # (N,)
    Xd: np.ndarray           # (N, Kmax_s + Kmax_c) lag regressors (centred): [self..., social...]
    unit: np.ndarray         # (N,) global unit index
    time: np.ndarray         # (N,) global time index
    group: np.ndarray        # (N,) group index
    Ks_max: int
    Kc_max: int
    n_units: int
    n_times: int
    c: float


def build_design(groups, Ks_max, Kc_max, mode="global", k_nn=3, n_min=2, c=None):
    """Rows (i, t) with x_i^t present, all self lags present (tau<=Ks_max) and all social
    regressors defined (tau<=Kc_max). Same rows for every (Ks, Kc) <= maxima."""
    if c is None:
        c = float(np.nanmean(np.concatenate([g.X.ravel() for g in groups])))
    ys, Xs, us, ts, gs = [], [], [], [], []
    u_off = t_off = 0
    K = max(Ks_max, Kc_max)
    for gi, g in enumerate(groups):
        U, T = g.X.shape
        if T <= K:
            u_off += U; t_off += T
            continue
        S = social_regressors(g, Kc_max, mode, k_nn, n_min) if Kc_max > 0 else np.zeros((0, U, T))
        tt = np.arange(K, T)
        cur = g.X[:, K:]                                            # (U, T-K)
        self_l = np.stack([g.X[:, K - tau:T - tau] for tau in range(1, Ks_max + 1)], axis=-1) \
            if Ks_max else np.zeros((U, T - K, 0))
        soc_l = np.moveaxis(S[:, :, K:], 0, -1) if Kc_max else np.zeros((U, T - K, 0))
        ok = ~np.isnan(cur)
        if Ks_max:
            ok &= ~np.isnan(self_l).any(axis=-1)
        if Kc_max:
            ok &= ~np.isnan(soc_l).any(axis=-1)
        ii, jj = np.nonzero(ok)
        ys.append(cur[ii, jj])
        Xs.append(np.concatenate([self_l[ii, jj] - c, soc_l[ii, jj] - c], axis=1))
        us.append(u_off + ii)
        ts.append(t_off + tt[jj])
        gs.append(np.full(len(ii), gi))
        u_off += U
        t_off += T
    y = np.concatenate(ys) if ys else np.zeros(0)
    Xd = np.vstack(Xs) if Xs else np.zeros((0, Ks_max + Kc_max))
    us, ts = np.concatenate(us), np.concatenate(ts)
    # re-index units/times to the ones actually used
    uu, uinv = np.unique(us, return_inverse=True)
    tt, tinv = np.unique(ts, return_inverse=True)
    return Design(y, Xd, uinv, tinv, np.concatenate(gs), Ks_max, Kc_max, len(uu), len(tt), c)


# ----------------------------------------------------------------------------------
# Fitter with fixed effects (sparse Newton)
# ----------------------------------------------------------------------------------

def schur_solve(H, G, lo, hi, bs=1, shared_cov=0):
    """Solve H X = G when the parameters lo:hi (time fields) form a block-diagonal block of H with
    blocks of size bs (1 for moduli, 2 for directions): the time block is eliminated exactly and the
    remaining dense system (units + shared couplings, a few hundred) is solved directly.  Same
    solution as spsolve(H, G), without the fill-in of a sparse factorization with ~1e4 time fields.
    shared_cov > 0 also returns the last shared_cov x shared_cov block of H^{-1}."""
    H = H.tocsr()
    n = H.shape[0]
    r = np.r_[np.arange(0, lo), np.arange(hi, n)]
    Htt = H[lo:hi, lo:hi]
    if bs == 1:
        Tinv = sp.diags(1.0 / Htt.diagonal())
    else:
        dg, up = Htt.diagonal(), Htt.diagonal(1)
        a, c, b = dg[0::2], dg[1::2], up[0::2]
        det = a * c - b * b
        nb = len(a)
        i0 = 2 * np.arange(nb)
        Tinv = sp.csr_matrix((np.concatenate([c / det, -b / det, -b / det, a / det]),
                              (np.concatenate([i0, i0, i0 + 1, i0 + 1]), np.concatenate([i0, i0 + 1, i0, i0 + 1]))),
                             shape=(hi - lo, hi - lo))
    Hr = H[r]
    Hrr = Hr[:, r].toarray()
    Hrt = Hr[:, lo:hi].tocsr()
    S = Hrr - (Hrt @ Tinv @ Hrt.T).toarray()
    G2 = np.asarray(G, dtype=float).reshape(n, -1)
    xr = np.linalg.solve(S, G2[r] - Hrt @ (Tinv @ G2[lo:hi]))
    xt = Tinv @ (G2[lo:hi] - Hrt.T @ xr)
    X = np.empty_like(G2)
    X[r] = xr
    X[lo:hi] = xt
    X = X.reshape(np.shape(G))
    if shared_cov:
        return X, np.linalg.inv(S)[-shared_cov:, -shared_cov:]
    return X


@dataclass
class CFit:
    Ks: int
    Kc: int
    family: str
    unit_fe: bool
    time_fe: bool
    b: np.ndarray            # shared coefficients [intercept?] + a_1..a_Ks + J_1..J_Kc (+ beta)
    names: list
    fe_unit: np.ndarray
    fe_time: np.ndarray
    nll: float
    N: int
    k: int
    H: object                # sparse Hessian at the optimum (incl. FE ridge); covariance computed lazily
    off: int                 # offset of the shared block in the parameter vector
    converged: bool
    n_iter: int
    _cov: np.ndarray = None

    @property
    def cov_b(self):
        """Shared block of the inverse Hessian (computed on first use only)."""
        if self._cov is None:
            npar = self.H.shape[0]
            p = npar - self.off
            E = sp.csc_matrix((np.ones(p), (np.arange(self.off, npar), np.arange(p))), shape=(npar, p))
            try:
                Y = spla.spsolve(self.H.tocsc(), E)
                Y = Y.toarray() if sp.issparse(Y) else np.asarray(Y).reshape(npar, p)
                self._cov = Y[self.off:, :]
            except Exception:
                self._cov = np.full((p, p), np.nan)
        return self._cov

    @property
    def aic(self):
        return 2 * self.nll + 2 * self.k

    @property
    def bic(self):
        return 2 * self.nll + self.k * np.log(self.N)

    def coef(self, name):
        return self.b[self.names.index(name)]

    def se(self, name):
        j = self.names.index(name)
        return float(np.sqrt(self.cov_b[j, j]))

    @property
    def a(self):
        return np.array([self.coef(f"a{t}") for t in range(1, self.Ks + 1)])

    @property
    def J(self):
        return np.array([self.coef(f"J{t}") for t in range(1, self.Kc + 1)])

    @property
    def se_J(self):
        return np.array([self.se(f"J{t}") for t in range(1, self.Kc + 1)])

    @property
    def se_a(self):
        return np.array([self.se(f"a{t}") for t in range(1, self.Ks + 1)])


def _row_stats(theta, beta, family, L, quad):
    """Per row: logZ, mean, var (and for lognormal: E[x^2], Cov(x,x^2), Var(x^2))."""
    if family == "powerlaw":
        lz = pl_logZ(theta, L)
        m, v = pl_mean_var(theta, L)
        return lz, m, v, None, None, None
    lz, m1, m2, vx, cx, vx2 = quad.stats(theta, beta)
    return lz, m1, vx, m2, cx, vx2


def fit_collective(D: Design, Ks, Kc, family="powerlaw", unit_fe=True, time_fe=False, L=None,
                   ridge_fe=1e-6, maxiter=100, tol=1e-10, init=None, quad=None, schur=False):
    """Exact MLE of the causal collective model on the rows of design D.
    schur=True (with time fields): Newton steps and covariance by elimination of the diagonal
    time-field block (schur_solve); same optimum, much faster for thousands of frames."""
    schur = schur and time_fe
    y = D.y
    N = len(y)
    cols = list(range(Ks)) + list(range(D.Ks_max, D.Ks_max + Kc))
    X = D.Xd[:, cols]
    names = [f"a{t}" for t in range(1, Ks + 1)] + [f"J{t}" for t in range(1, Kc + 1)]
    if not unit_fe:
        X = np.column_stack([np.ones(N), X])
        names = ["h"] + names
    lognormal = family == "lognormal"
    if lognormal:
        names = names + ["beta"]
        if quad is None:
            quad = GLQuad(L)
    p = X.shape[1] + (1 if lognormal else 0)
    nu = D.n_units if unit_fe else 0
    nt = D.n_times if time_fe else 0
    npar = nu + nt + p
    # sparse indicator matrices
    Eu = sp.csr_matrix((np.ones(N), (np.arange(N), D.unit)), shape=(N, D.n_units)) if unit_fe else None
    Et = sp.csr_matrix((np.ones(N), (np.arange(N), D.time)), shape=(N, D.n_times)) if time_fe else None

    def unpack(par):
        hu = par[:nu]
        gt = par[nu:nu + nt]
        bb = par[nu + nt:]
        return hu, gt, bb

    def theta_of(par):
        hu, gt, bb = unpack(par)
        th = X @ bb[:X.shape[1]]
        if unit_fe:
            th = th + hu[D.unit]
        if time_fe:
            th = th + gt[D.time]
        beta = bb[-1] if lognormal else 0.0
        return th, beta

    def terms(par):
        th, beta = theta_of(par)
        lz, m, v, m2, cx, vx2 = _row_stats(th, beta, family, L, quad)
        nll = float(np.sum(-th * y + beta * y**2 + lz))
        r = m - y                                  # dNLL/dtheta per row
        # gradient
        g_parts = []
        if unit_fe:
            g_parts.append(np.bincount(D.unit, weights=r, minlength=D.n_units))
        if time_fe:
            g_parts.append(np.bincount(D.time, weights=r, minlength=D.n_times))
        g_b = X.T @ r
        if lognormal:
            g_beta = np.sum(y**2 - m2)
            g_parts.append(np.concatenate([g_b, [g_beta]]))
        else:
            g_parts.append(g_b)
        grad = np.concatenate(g_parts)
        # Hessian blocks (sparse)
        blocks = []
        Wv = sp.diags(v)
        Zth = []                                   # columns acting on theta
        if unit_fe:
            Zth.append(Eu)
        if time_fe:
            Zth.append(Et)
        Zth.append(sp.csr_matrix(X))
        Z = sp.hstack(Zth, format="csr")
        Hth = (Z.T @ Wv @ Z).tocsr()
        if lognormal:
            # beta enters with feature -x^2: d2A/dtheta dbeta = -Cov(x,x^2), d2A/dbeta2 = Var(x^2)
            col = -(Z.T @ cx)
            Hbb = np.sum(vx2)
            H = sp.bmat([[Hth, sp.csr_matrix(col.reshape(-1, 1))],
                         [sp.csr_matrix(col.reshape(1, -1)), sp.csr_matrix([[Hbb]])]], format="csr")
        else:
            H = Hth
        return nll, grad, H, r

    par = np.zeros(npar) if init is None else np.array(init, dtype=float)
    if init is None:
        from maxent_levy import _invert_mean
        th0 = _invert_mean(float(np.mean(y)), L)
        if unit_fe:
            par[:nu] = th0
        else:
            par[nu + nt] = th0
    R = np.zeros(npar)
    if unit_fe and time_fe:
        R[nu:nu + nt] = ridge_fe                   # removes the (h_i + c, g_t - c) null direction
    Rm = sp.diags(R)

    def total(par):
        nll, g, H, r = terms(par)
        return nll + 0.5 * np.sum(R * par**2), g + R * par, (H + Rm).tocsc(), nll, H, r

    f, g, H, nll, Hraw, r = total(par)
    conv = False
    it = 0
    for it in range(1, maxiter + 1):
        try:
            step = schur_solve(H, g, nu, nu + nt) if schur else spla.spsolve(H, g)
        except Exception:
            step = g / (H.diagonal() + 1e-8)
        if not np.all(np.isfinite(step)):
            step = g / (np.abs(H.diagonal()) + 1e-8)
        dec = float(g @ step)
        t = 1.0
        while True:
            pn = par - t * step
            fn, gn, Hn, nlln, Hrawn, rn = total(pn)
            if np.isfinite(fn) and fn <= f - 1e-4 * t * dec:
                break
            t *= 0.5
            if t < 1e-12:
                break
        par, f, g, H, nll, Hraw, r = pn, fn, gn, Hn, nlln, Hrawn, rn
        if dec / 2 < tol * max(1.0, abs(f)):
            conv = True
            break
    # the covariance of the shared block is computed lazily (CFit.cov_b) from the stored Hessian
    hu, gt, bb = unpack(par)
    kpar = nu + max(nt - (1 if (unit_fe and nt) else 0), 0) + p
    fit = CFit(Ks, Kc, family, unit_fe, time_fe, bb, names, hu, gt, nll, N, kpar, H, nu + nt, conv, it)
    if schur and p:
        fit._cov = schur_solve(H, np.zeros(npar), nu, nu + nt, shared_cov=p)[1]
    return fit


def scan(D: Design, Ks_list, Kc_list, **kw):
    """Fit every (Ks, Kc) on the same rows; returns dict {(Ks,Kc): CFit}."""
    out = {}
    for Ks in Ks_list:
        for Kc in Kc_list:
            out[(Ks, Kc)] = fit_collective(D, Ks, Kc, **kw)
    return out


def best(fits, crit="bic"):
    k = min(fits, key=lambda kk: getattr(fits[kk], crit))
    return k, fits[k]


# ----------------------------------------------------------------------------------
# PRE pipeline (Agliari et al. 2024) adapted to moduli
# ----------------------------------------------------------------------------------

def Lambda(theta, L):
    """Single-site mean E[x] for p(x) ∝ e^{θx} on [0, L] (Langevin)."""
    return pl_mean_var(np.asarray(theta, dtype=float), L)[0]


def Lambda_inv(m, L, lo=-500.0, hi=500.0, n_iter=200):
    """Vectorised inverse of Lambda by bisection on u = θL."""
    m = np.clip(np.asarray(m, dtype=float), 1e-9 * L, (1 - 1e-9) * L)
    a = np.full(m.shape, lo)
    b = np.full(m.shape, hi)
    for _ in range(n_iter):
        mid = (a + b) / 2
        val = Lambda(mid / L, L)
        lt = val < m
        a = np.where(lt, mid, a)
        b = np.where(lt, b, mid)
    return (a + b) / 2 / L


def collective_series(g: Group, n_min=2):
    """m_t = mean of x over units present at t (NaN if fewer than n_min)."""
    pres = ~np.isnan(g.X)
    n = pres.sum(axis=0)
    m = np.where(n >= n_min, np.nansum(g.X, axis=0) / np.maximum(n, 1), np.nan)
    return m, n


def pre_pipeline(groups, L, tauM_max, n_min=2):
    """Faithful PRE procedure (Eqs. A35-A61), D=1: for each tau_M, OLS of
    G_t = Lambda^{-1}(m_t) on [1, m_{t-1}, ..., m_{t-tau_M}], sigma^2 from residuals,
    AIC = 2 (tau_M + D + 1)/(D (N_T - tau_M)) + log(N_T sigma^2 / (N_T - tau_M)).
    Rows: frames with m defined at t and all lags (pooled over groups, same rows for all tau_M)."""
    rows_G, rows_M = [], []
    for g in groups:
        m, n = collective_series(g, n_min)
        G = Lambda_inv(m, L)
        T = len(m)
        for t in range(tauM_max, T):
            lagm = m[t - tauM_max:t][::-1]
            if np.isnan(m[t]) or np.isnan(lagm).any():
                continue
            rows_G.append(G[t])
            rows_M.append(lagm)
    G = np.array(rows_G)
    M = np.array(rows_M)
    NT = len(G)
    out = {}
    for tauM in range(0, tauM_max + 1):
        A = np.column_stack([np.ones(NT), M[:, :tauM]])
        coef, *_ = np.linalg.lstsq(A, G, rcond=None)
        res = G - A @ coef
        sigma2 = float(np.mean(res**2))
        n_o = NT - tauM
        n_p = tauM + 1 + 1
        eps2 = NT * sigma2 / max(NT - tauM, 1)
        aic = 2 * n_p / n_o + np.log(eps2)
        # standard errors from the Gaussian Fisher information (Eq. A57)
        cov = sigma2 * np.linalg.pinv(A.T @ A)
        out[tauM] = dict(H=float(coef[0]), J=coef[1:].copy(), se=np.sqrt(np.diag(cov))[1:], sigma2=sigma2,
                         aic=float(aic), NT=NT)
    tau_sel = min(out, key=lambda k: out[k]["aic"])
    J = out[tau_sel]["J"]
    delay = int(np.argmax(np.abs(J)) + 1) if tau_sel > 0 else 0
    return out, tau_sel, delay
