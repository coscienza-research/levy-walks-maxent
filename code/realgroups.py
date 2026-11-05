"""Self + social MaxEnt on co-recorded groups given by positions on a common clock (open data:
wild baboons, homing-pigeon flocks).  For moduli and directions,

    theta_i^t = h_i + g_t + sum_tau a_tau z_i^{t-tau} + sum_tau J_tau q_i^{t-tau}

z = own log-step (moduli) or own unit heading (directions; zero when the unit does not move);
q = mean of the same variable over i's neighbours at the emission frame t - tau.
Neighbours ('then', the classical definition): ranks r0..r1-1 by distance from i at frame t - tau.
Reading 'now' (directions only): the units that at t - tau were nearest to where i is NOW; it also
contains any field fixed in space (trails, landmarks), which time fields cannot absorb.
Split (directions only): neighbours ahead of / behind i along the group heading at t - tau.
Units = (group, individual); frames = (group, t).  Fits eliminate the time fields exactly (schur=True).
"""
import numpy as np

import collective as co
import collective_dir as cd


class Grp:
    """One co-recorded group: positions P_full (U, T+1, 2) on the clock (NaN = no fix)."""

    def __init__(self, name, P_full):
        self.name = name
        self.P = P_full[:, :-1]
        self.V = np.diff(P_full, axis=1)

    def shifted(self, rng):
        """every unit shifted circularly by an independent random offset (time-shift null)"""
        U, T = self.V.shape[:2]
        g = Grp.__new__(Grp)
        g.name = self.name
        g.P, g.V = np.empty_like(self.P), np.empty_like(self.V)
        for i in range(U):
            s = int(rng.integers(T))
            g.P[i] = np.roll(self.P[i], s, axis=0)
            g.V[i] = np.roll(self.V[i], s, axis=0)
        return g


def headings(V, min_step):
    ell = np.hypot(V[..., 0], V[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        h = V / ell[..., None]
    h = np.where((ell >= min_step)[..., None], h, 0.0)
    return np.where(np.isnan(ell)[..., None], np.nan, h), ell >= min_step


def logsteps(V, lmin, lmax):
    ell = np.hypot(V[..., 0], V[..., 1])
    L = np.log(lmax / lmin)
    x = np.clip(np.log(np.maximum(ell, 1e-12) / lmin), 0, L)
    return np.where(np.isnan(ell) | (ell > lmax), np.nan, x)


def cutoffs(groups, q_min=5, q_max=99.95):
    ell = np.concatenate([np.hypot(g.V[..., 0], g.V[..., 1]).ravel() for g in groups])
    ell = ell[np.isfinite(ell)]
    lmin = float(np.percentile(ell[ell > 0], q_min)); lmax = float(np.percentile(ell, q_max))
    return lmin, lmax, float(np.log(lmax / lmin))


def _group_heading(V):
    m = np.nanmean(V, axis=0)                                   # (T, 2)
    n = np.hypot(m[:, 0], m[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return m / n[:, None]


def emitted(P, Z, ranks, split=False, V=None):
    """Q[i, s] = mean of Z over the neighbours of i of distance ranks [r0, r1) at frame s.
    Z (U, T) or (U, T, 2).  split=True (directions): returns (Q_ahead, Q_behind), neighbours ahead
    of / behind i along the group heading at s (zero when there is none)."""
    r0, r1 = ranks                      # r1 = None: all the other units present (global coupling)
    U, T = Z.shape[:2]
    vec = Z.ndim == 3
    shp = (U, T, 2) if vec else (U, T)
    Q = np.full(shp, np.nan)
    if split:
        Q2 = np.full(shp, np.nan)
        c = _group_heading(V)
    for s in range(T):
        idx = np.where(~np.isnan(P[:, s, 0]))[0]
        if len(idx) - 1 < (r1 if r1 is not None else 1):
            continue
        d = P[idx, s][:, None, :] - P[idx, s][None, :, :]
        D = np.hypot(d[..., 0], d[..., 1]); np.fill_diagonal(D, np.inf)
        nb = idx[np.argsort(D, axis=1)[:, r0:(r1 if r1 is not None else len(idx) - 1)]]   # (n, k)
        Zn = Z[nb, s]
        if not split:
            with np.errstate(invalid="ignore"):
                Q[idx, s] = np.nanmean(Zn, axis=1)
            continue
        if not np.all(np.isfinite(c[s])):
            continue
        ahead = ((P[nb, s] - P[idx, s][:, None, :]) @ c[s]) > 0     # (n, k)
        Zn = np.where(np.isnan(Zn), 0.0, Zn)
        for out, m in ((Q, ahead), (Q2, ~ahead)):
            cnt = m.sum(axis=1)
            val = (Zn * m[..., None]).sum(axis=1) / np.maximum(cnt, 1)[:, None]
            out[idx, s] = val
    return (Q, Q2) if split else Q


def emitted_now(P, Z, k, Kc):
    """Q[tau-1, i, t] = mean of Z at frame t - tau over the k units nearest, at t - tau, to the
    position of i at frame t (directions)."""
    U, T = Z.shape[:2]
    Q = np.full((Kc, U, T, 2), np.nan)
    for t in range(T):
        tgt = np.where(~np.isnan(P[:, t, 0]))[0]
        for tau in range(1, Kc + 1):
            s = t - tau
            if s < 0 or len(tgt) == 0:
                continue
            src = np.where(~np.isnan(P[:, s, 0]))[0]
            if len(src) < k + 1:
                continue
            D = np.hypot(*(P[tgt, t][:, None, :] - P[src, s][None, :, :]).transpose(2, 0, 1))
            D[tgt[:, None] == src[None, :]] = np.inf
            nb = src[np.argsort(D, axis=1)[:, :k]]
            with np.errstate(invalid="ignore"):
                Q[tau - 1, tgt, t] = np.nanmean(Z[nb, s], axis=1)
    return Q


def rows(groups, variable, Ks, Kc, social=(("then", (0, 3)),), min_step=1.0, cut=None):
    """Design for moduli (co.Design) or directions (dict for cd.fit_dir_fe); same rows for every
    (Ks' <= Ks, Kc' <= Kc).  social: families of social regressors, each (kind, ranks) with kind
    'then' (neighbours of ranks r0..r1-1 at the emission frame), 'now' (directions: units nearest, at
    the emission frame, to where i is now), 'ahead' / 'behind' (directions: the 'then' neighbours
    ahead of / behind i along the group heading).  With several families the social columns are
    [family 1 lags 1..Kc, family 2 lags 1..Kc, ...] and the fit takes Kc * len(social) couplings.
    Moduli: cut = (lmin, lmax)."""
    import warnings
    K = max(Ks, Kc)
    Y, SL, QL, UN, TI = [], [], [], [], []
    u_off = t_off = 0
    for g in groups:
        U, T = g.V.shape[:2]
        if variable == "dir":
            Z, moving = headings(g.V, min_step)
            target_ok = moving
        else:
            Z = logsteps(g.V, *cut)
            target_ok = ~np.isnan(Z)
        soc = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for kind, ranks in (social if Kc else ()):
                if kind == "now":
                    Qt = emitted_now(g.P, Z, ranks[1], Kc)
                    soc += [Qt[tau - 1] for tau in range(1, Kc + 1)]
                    continue
                if kind in ("ahead", "behind"):
                    Qa, Qb = emitted(g.P, Z, ranks, split=True, V=g.V)
                    Qe = Qa if kind == "ahead" else Qb
                else:
                    Qe = emitted(g.P, Z, ranks)
                soc += [np.roll(Qe, tau, axis=1) for tau in range(1, Kc + 1)]
        selfl = [np.roll(Z, tau, axis=1) for tau in range(1, Ks + 1)]
        ok = target_ok.copy()
        ok[:, :K] = False
        for a in selfl + soc:
            ok &= ~(np.isnan(a[..., 0]) if a.ndim == 3 else np.isnan(a))
        ii, tt = np.nonzero(ok)
        Y.append(Z[ii, tt])
        SL.append(np.stack([a[ii, tt] for a in selfl], axis=1) if Ks else np.zeros((len(ii), 0) + Z.shape[2:]))
        QL.append(np.stack([a[ii, tt] for a in soc], axis=1) if soc else np.zeros((len(ii), 0) + Z.shape[2:]))
        UN.append(u_off + ii); TI.append(t_off + tt)
        u_off += U; t_off += T
    y = np.concatenate(Y); sl = np.concatenate(SL); ql = np.concatenate(QL)
    uu, unit = np.unique(np.concatenate(UN), return_inverse=True)
    tt, time = np.unique(np.concatenate(TI), return_inverse=True)
    if variable == "dir":
        return dict(y=y, selfl=sl, socl=ql, unit=unit, time=time, n_units=len(uu), n_times=len(tt),
                    Ks_max=Ks, Kc_max=ql.shape[1])
    c = float(np.mean(y))
    Xd = np.concatenate([sl - c, ql - c], axis=1)
    return co.Design(y, Xd, unit, time, np.zeros(len(y), int), Ks, ql.shape[1], len(uu), len(tt), c)


# ------------------------------------------------------------------------------------------- fits
def fit(D, variable, Ks, Kc, time_fe=True, L=None, unit_fe=True):
    if variable == "dir":
        f = cd.fit_dir_fe(D, Ks, Kc, unit_fe=unit_fe, time_fe=time_fe, schur=True)
        return dict(a=np.asarray(f["a"]).tolist(), J=np.asarray(f["J"]).tolist(), se_a=np.asarray(f["se_a"]).tolist(),
                    se_J=np.asarray(f["se_J"]).tolist(), nll=float(f["nll"]), k=int(f["k"]), N=int(f["N"]),
                    bic=float(f["bic"]), conv=bool(f["conv"]))
    f = co.fit_collective(D, Ks, Kc, L=L, unit_fe=unit_fe, time_fe=time_fe, schur=True)
    return dict(a=f.a.tolist(), J=f.J.tolist(), se_a=f.se_a.tolist(), se_J=f.se_J.tolist(), nll=float(f.nll),
                k=int(f.k), N=int(f.N), bic=float(f.bic), conv=bool(f.converged))


def bic_path(D, variable, Ks_max, Kc_list, L=None):
    """depth selection with time fields along the axes: Ks at Kc = 0, then Kc at the selected Ks"""
    ks_b = {ks: fit(D, variable, ks, 0, L=L)["bic"] for ks in range(Ks_max + 1)}
    ks = min(ks_b, key=ks_b.get)
    kc_b = {kc: fit(D, variable, ks, kc, L=L)["bic"] for kc in Kc_list}
    return int(ks), int(min(kc_b, key=kc_b.get)), ks_b, kc_b


def group_average(groups, variable, tmax, min_step=1.0, cut=None):
    """the analysis of the group average (mean-field reading): selected depth and delay"""
    if variable == "mod":
        L = float(np.log(cut[1] / cut[0]))
        gs = [co.Group(X=logsteps(g.V, *cut), P=g.P) for g in groups]
        _, tsel, delay = co.pre_pipeline(gs, L, tmax)
        return dict(tauM=int(tsel), delay=int(delay))
    out = []
    for g in groups:
        ell = np.hypot(g.V[..., 0], g.V[..., 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where((ell > 0)[..., None], g.V / ell[..., None], np.nan)
        ok = ~np.all(np.isnan(s[..., 0]), axis=0)
        if ok.sum() <= 3 * tmax:
            continue
        s = s[:, np.argmax(ok):len(ok) - np.argmax(ok[::-1])]
        s = s[:, ~np.all(np.isnan(s[..., 0]), axis=0)]
        _, tsel, delay = cd.pre_pipeline_dir(s, tmax, rows="fixed")
        out.append(dict(tauM=int(tsel), delay=int(delay)))
    return out
