"""Synthetic collective movement with self and social delayed couplings on step moduli.

'Speed-Vicsek' model in a periodic 2D box (fixed clock, delta t = 1):
  headings : Vicsek alignment with the k_align nearest neighbours (incl. self) + angular noise eta
  log-steps: x_i^t ~ p(x) ∝ exp(theta_i^t x) on [0, L], with
             theta_i^t = h_i + sum_tau a_tau (x_i^{t-tau} - c)
                              + sum_tau Jloc_tau (mloc_i^{t-tau} - c)      (k nearest others at t-tau)
                              + sum_tau Jglob_tau (mglob_i^{t-tau} - c)    (leave-one-out mean)
                              + u_t                                         (common drive, OU)
  positions: r_i^{t+1} = r_i^t + ell_min e^{x_i^t} s_i^t   (minimum image)
The observed data are positions; steps are recomputed from them as in a real experiment.
"""
import numpy as np
from maxent_levy import pl_sample
from collective import Group


def _knn(P, box, k):
    d = P[:, None, :] - P[None, :, :]
    d -= box * np.round(d / box)
    D = np.hypot(d[..., 0], d[..., 1])
    np.fill_diagonal(D, np.inf)
    return np.argpartition(D, k - 1, axis=1)[:, :k]


def speed_vicsek(N, T, rng, L=np.log(100.0), box=30.0, h=None, a=(), Jloc=(), Jglob=(), k_nn=3,
                 k_align=6, eta=0.3, drive_sd=0.0, drive_tau=20.0, c=None, burn=300, ell_min=0.05,
                 loc_noise=0.0):
    a = np.asarray(a, dtype=float)
    Jloc = np.asarray(Jloc, dtype=float)
    Jglob = np.asarray(Jglob, dtype=float)
    K = max(len(a), len(Jloc), len(Jglob), 1)
    tot = T + burn
    if h is None:
        h = np.full(N, -1.0)
    h = np.asarray(h, dtype=float)
    if c is None:
        c = L / 2
    X = np.empty((N, tot))
    P = np.empty((N, tot + 1, 2))
    P[:, 0] = rng.uniform(0, box, (N, 2))
    ang = rng.uniform(-np.pi, np.pi, N)
    mloc = np.zeros((N, tot))
    mglob = np.zeros((N, tot))
    u = np.zeros(tot)
    phi = np.exp(-1.0 / drive_tau)
    for t in range(tot):
        if t > 0:
            u[t] = phi * u[t - 1] + drive_sd * np.sqrt(1 - phi**2) * rng.standard_normal()
        theta = h + u[t]
        for tau in range(1, K + 1):
            s = t - tau
            xs = X[:, s] if s >= 0 else np.full(N, c)
            if tau <= len(a):
                theta = theta + a[tau - 1] * (xs - c)
            if tau <= len(Jloc):
                theta = theta + Jloc[tau - 1] * ((mloc[:, s] if s >= 0 else c) - c)
            if tau <= len(Jglob):
                theta = theta + Jglob[tau - 1] * ((mglob[:, s] if s >= 0 else c) - c)
        X[:, t] = pl_sample(theta, L, rng)
        # social summaries at time t (used by later frames)
        nn = _knn(P[:, t], box, k_nn)
        mloc[:, t] = X[nn, t].mean(axis=1)
        mglob[:, t] = (X[:, t].sum() - X[:, t]) / (N - 1)
        # headings (Vicsek) and positions
        na = _knn(P[:, t], box, k_align - 1)
        vx = np.cos(ang) + np.cos(ang[na]).sum(axis=1)
        vy = np.sin(ang) + np.sin(ang[na]).sum(axis=1)
        ang = np.arctan2(vy, vx) + eta * rng.uniform(-np.pi, np.pi, N)
        step = ell_min * np.exp(X[:, t])
        P[:, t + 1] = (P[:, t] + step[:, None] * np.c_[np.cos(ang), np.sin(ang)]) % box
    P = P[:, burn:]
    Pobs = P + loc_noise * rng.standard_normal(P.shape) if loc_noise > 0 else P
    return dict(X_true=X[:, burn:], P=Pobs, u=u[burn:], box=box, L=L, ell_min=ell_min, c=c)


def pre_model(N, T, H, J, L, rng, burn=300):
    """The kinetic model behind the PRE mean-field measure, for moduli: common field H, global
    coupling to the full mean (including self), no individual memory, homogeneous units:
    theta^t = H + sum_tau J_tau m_{t-tau}. Returns X (N x T)."""
    J = np.atleast_1d(np.asarray(J, dtype=float))
    K = len(J)
    tot = T + burn
    X = np.empty((N, tot))
    m = np.full(tot, L / 2)
    for t in range(tot):
        th = H + sum(J[tau - 1] * (m[t - tau] if t - tau >= 0 else L / 2) for tau in range(1, K + 1))
        X[:, t] = pl_sample(np.full(N, th), L, rng)
        m[t] = X[:, t].mean()
    return X[:, burn:]


def group_from_positions(sim, lmin=None, lmax=None):
    """Recompute log steps from (observed) positions with the minimum-image convention,
    exactly as in an experiment. Returns a Group and the (L, ell_min) used."""
    P, box = sim["P"], sim["box"]
    d = np.diff(P, axis=1)
    d -= box * np.round(d / box)
    ell = np.hypot(d[..., 0], d[..., 1])
    lmin = sim["ell_min"] if lmin is None else lmin
    lmax = sim["ell_min"] * np.exp(sim["L"]) if lmax is None else lmax
    L = float(np.log(lmax / lmin))
    X = np.clip(np.log(np.maximum(ell, 1e-300) / lmin), 0, L)
    return Group(X=X, P=P[:, :-1, :], box=box), L
