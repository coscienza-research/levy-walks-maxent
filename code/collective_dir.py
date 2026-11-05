"""Self + social decomposition for DIRECTIONS (planar unit vectors), i.e. the variables of the PRE.

Causal von Mises model:
    p(s_i^t | past) = exp(theta_i^t . s_i^t) / (2 pi I0(|theta_i^t|)),
    theta_i^t = H + sum_{tau<=Ks} a_tau s_i^{t-tau} + sum_{tau<=Kc} J_tau q_i^{t-tau}   (vectors)
with q the leave-one-out mean direction ('global') or the mean of the k nearest ('local').
Scalar couplings a_tau, J_tau; common field H (2-vector). NLL convex (log I0(|theta|) is convex).
Also: the PRE pipeline for D = 2 (Eqs. A35-A61), and simulators.
"""
import numpy as np
from scipy.special import i0e, i1e


def A(k):
    k = np.asarray(k, dtype=float)
    out = np.where(k < 1e-8, k / 2, i1e(k) / np.maximum(i0e(k), 1e-300))
    return out


def Ainv(r, lo=0.0, hi=1e4, n=200):
    r = np.clip(np.asarray(r, dtype=float), 0, 1 - 1e-12)
    a = np.full(r.shape, lo); b = np.full(r.shape, hi)
    for _ in range(n):
        mid = (a + b) / 2
        lt = A(mid) < r
        a = np.where(lt, mid, a); b = np.where(lt, b, mid)
    return (a + b) / 2


def logI0(k):
    return np.log(i0e(k)) + k


# ----------------------------------------------------------------------------------
def social_dir(S, P, Kc, mode, box=None, k_nn=3):
    """Q[tau-1, i, t, :] = social mean direction of unit i built at frame t - tau."""
    U, T, _ = S.shape
    Q = np.full((Kc, U, T, 2), np.nan)
    pres = ~np.isnan(S[..., 0])
    for s in range(T):
        idx = np.where(pres[:, s])[0]
        if len(idx) < 2:
            continue
        if mode == "global":
            tot = S[idx, s].sum(axis=0)
            val = (tot[None, :] - S[idx, s]) / (len(idx) - 1)
        elif mode == "global+":
            val = np.repeat(S[idx, s].mean(axis=0)[None, :], len(idx), axis=0)
        else:
            d = P[idx, s][:, None, :] - P[idx, s][None, :, :]
            if box is not None:
                d -= box * np.round(d / box)
            D = np.hypot(d[..., 0], d[..., 1]); np.fill_diagonal(D, np.inf)
            k = min(k_nn, len(idx) - 1)
            nn = np.argsort(D, axis=1)[:, :k]
            val = S[idx[nn], s].mean(axis=1)
        for tau in range(1, Kc + 1):
            if s + tau < T:
                Q[tau - 1, idx, s + tau] = val
    return Q


def build_dir(S, P, Ks, Kc, mode="global", box=None, k_nn=3):
    """Rows (i,t): target s_i^t, self lags (Ks,2), social lags (Kc,2)."""
    U, T, _ = S.shape
    K = max(Ks, Kc)
    Q = social_dir(S, P, Kc, mode, box, k_nn) if Kc else np.zeros((0, U, T, 2))
    ys, sl, ql = [], [], []
    cur = S[:, K:]
    selfl = np.stack([S[:, K - tau:T - tau] for tau in range(1, Ks + 1)], axis=2) if Ks else np.zeros((U, T - K, 0, 2))
    socl = np.moveaxis(Q[:, :, K:], 0, 2) if Kc else np.zeros((U, T - K, 0, 2))
    ok = ~np.isnan(cur[..., 0])
    if Ks:
        ok &= ~np.isnan(selfl[..., 0]).any(axis=2)
    if Kc:
        ok &= ~np.isnan(socl[..., 0]).any(axis=2)
    ii, jj = np.nonzero(ok)
    return cur[ii, jj], selfl[ii, jj], socl[ii, jj]


def fit_dir(y, selfl, socl, Ks, Kc, maxiter=60, tol=1e-10):
    """MLE of (H_x, H_y, a_1..a_Ks, J_1..J_Kc). Rows: y (N,2), selfl (N,Ks*,2), socl (N,Kc*,2)."""
    N = len(y)
    # design: theta = B p, B (N, 2, P)
    cols = [np.broadcast_to(np.array([1.0, 0.0]), (N, 2)), np.broadcast_to(np.array([0.0, 1.0]), (N, 2))]
    cols += [selfl[:, t] for t in range(Ks)] + [socl[:, t] for t in range(Kc)]
    B = np.stack(cols, axis=2)
    P = B.shape[2]
    p = np.zeros(P)
    m0 = y.mean(axis=0); r0 = np.linalg.norm(m0)
    if r0 > 0:
        p[:2] = m0 / r0 * Ainv(np.array([r0]))[0]

    def terms(p):
        th = np.einsum("nkp,p->nk", B, p)
        k = np.hypot(th[:, 0], th[:, 1])
        nll = float(np.sum(-np.sum(th * y, axis=1) + logI0(k)))
        Ak = A(k)
        u = th / np.maximum(k, 1e-12)[:, None]
        r = Ak[:, None] * u - y                                     # dNLL/dtheta
        g = np.einsum("nk,nkp->p", r, B)
        dA = 1 - Ak / np.maximum(k, 1e-12) - Ak**2                  # A'(k)
        small = k < 1e-6
        Mpar = np.where(small, 0.5, dA)
        Mperp = np.where(small, 0.5, Ak / np.maximum(k, 1e-12))
        Mx = Mperp[:, None, None] * np.eye(2)[None] + (Mpar - Mperp)[:, None, None] * np.einsum("ni,nj->nij", u, u)
        H = np.einsum("nkp,nkl,nlq->pq", B, Mx, B)
        return nll, g, H

    f, g, H = terms(p)
    conv = False
    for it in range(maxiter):
        step = np.linalg.solve(H + 1e-12 * np.eye(P), g)
        dec = g @ step
        t = 1.0
        while True:
            pn = p - t * step
            fn, gn, Hn = terms(pn)
            if fn <= f - 1e-4 * t * dec or t < 1e-12:
                break
            t *= 0.5
        p, f, g, H = pn, fn, gn, Hn
        if dec / 2 < tol * max(1, abs(f)):
            conv = True
            break
    cov = np.linalg.inv(H)
    k = P
    return dict(H=p[:2], a=p[2:2 + Ks], J=p[2 + Ks:], se_J=np.sqrt(np.diag(cov))[2 + Ks:],
                se_a=np.sqrt(np.diag(cov))[2:2 + Ks], nll=f, k=k, N=N,
                bic=2 * f + k * np.log(N), aic=2 * f + 2 * k, conv=conv)


def scan_dir(S, P, Ks_max, Kc_max, mode="global", box=None, k_nn=3):
    y, sl, ql = build_dir(S, P, Ks_max, Kc_max, mode, box, k_nn)
    out = {}
    for Ks in range(Ks_max + 1):
        for Kc in range(Kc_max + 1):
            out[(Ks, Kc)] = fit_dir(y, sl[:, :Ks], ql[:, :Kc], Ks, Kc)
    return out


# ----------------------------------------------------------------------------------
# Fixed-effects version: unit fields h_i and time fields g_t (2-vectors), sparse Newton
# ----------------------------------------------------------------------------------
def social_dir_now(S, P, Kc, mode="local", k_nn=3):
    """Q[tau-1, i, t] = mean direction at frame t - tau of the k cells (j != i) nearest to the
    CURRENT position P[i, t] ('local'), or of all other cells present at t - tau ('global').
    Only i's present position is needed: a delayed signal emitted at t - tau near where i is now."""
    from scipy.spatial import cKDTree
    U, T, _ = S.shape
    Q = np.full((Kc, U, T, 2), np.nan)
    pres = ~np.isnan(S[..., 0]) & ~np.isnan(P[..., 0])
    for t in range(T):
        tgt = np.where(~np.isnan(S[:, t, 0]) & ~np.isnan(P[:, t, 0]))[0]
        if len(tgt) == 0:
            continue
        for tau in range(1, Kc + 1):
            s = t - tau
            if s < 0:
                continue
            src = np.where(pres[:, s])[0]
            if len(src) < k_nn + 1:
                continue
            if mode == "global":
                tot = S[src, s].sum(axis=0)
                ins = np.isin(tgt, src)
                cnt = len(src) - ins
                val = (tot[None, :] - np.where(ins[:, None], S[tgt, s], 0.0)) / cnt[:, None]
            else:
                tree = cKDTree(P[src, s])
                kq = min(k_nn + 1, len(src))
                _, nb = tree.query(P[tgt, t], k=kq)
                nb = src[nb]                                         # (n_tgt, kq) unit ids
                keep = nb != tgt[:, None]
                # first k_nn neighbours that are not the cell itself
                order = np.cumsum(keep, axis=1)
                use = keep & (order <= k_nn)
                val = (S[nb, s] * use[..., None]).sum(axis=1) / use.sum(axis=1, keepdims=True)
            Q[tau - 1, tgt, t] = val
    return Q


def build_dir_fe(S, P, Ks_max, Kc_max, mode="local", k_nn=3, min_rows=8):
    """Rows (i, t): target s_i^t with all self lags (<= Ks_max) and social lags (<= Kc_max) defined.
    Units with fewer than min_rows rows are dropped (their unit field would be unidentified)."""
    U, T, _ = S.shape
    K = max(Ks_max, Kc_max)
    Q = social_dir_now(S, P, Kc_max, mode, k_nn) if Kc_max else None
    rows_i, rows_t = np.nonzero(~np.isnan(S[:, K:, 0]))
    rows_t = rows_t + K
    sl = np.stack([S[rows_i, rows_t - tau] for tau in range(1, Ks_max + 1)], axis=1) if Ks_max else np.zeros((len(rows_i), 0, 2))
    ql = np.stack([Q[tau - 1, rows_i, rows_t] for tau in range(1, Kc_max + 1)], axis=1) if Kc_max else np.zeros((len(rows_i), 0, 2))
    ok = ~np.isnan(sl[..., 0]).any(axis=1) & ~np.isnan(ql[..., 0]).any(axis=1)
    rows_i, rows_t, sl, ql = rows_i[ok], rows_t[ok], sl[ok], ql[ok]
    cnt = np.bincount(rows_i, minlength=U)
    ok = cnt[rows_i] >= min_rows
    rows_i, rows_t, sl, ql = rows_i[ok], rows_t[ok], sl[ok], ql[ok]
    uu, unit = np.unique(rows_i, return_inverse=True)
    tt, time = np.unique(rows_t, return_inverse=True)
    return dict(y=S[rows_i, rows_t], selfl=sl, socl=ql, unit=unit, time=time, n_units=len(uu), n_times=len(tt),
                units=uu, times=tt, Ks_max=Ks_max, Kc_max=Kc_max)


def fit_dir_fe(Dd, Ks, Kc, unit_fe=True, time_fe=False, ridge_t=1e-4, ridge_u=1e-6, maxiter=80, tol=1e-10, schur=False):
    """Exact MLE of the causal von Mises model
        theta_i^t = h_i + g_t + sum_{tau<=Ks} a_tau s_i^{t-tau} + sum_{tau<=Kc} J_tau q_i^{t,tau}
    (h_i, g_t in R^2; scalar a, J) on the rows of Dd, by Newton with a sparse Hessian.
    Without unit fields a common field H is fitted instead.  ridge_t removes the (h + c, g - c)
    gauge direction; ridge_u (tiny) keeps unit fields finite for nearly straight tracks."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from collective import schur_solve
    schur = schur and time_fe
    y = Dd["y"]; N = len(y)
    sl = Dd["selfl"][:, :Ks]; ql = Dd["socl"][:, :Kc]
    nu = 2 * Dd["n_units"] if unit_fe else 2
    nt = 2 * Dd["n_times"] if time_fe else 0
    p_sh = Ks + Kc
    P = nu + nt + p_sh
    rows = np.arange(2 * N)
    comp = np.tile([0, 1], N)
    n_of = np.repeat(np.arange(N), 2)
    # fixed-effect columns (one per row)
    col_u = (2 * Dd["unit"][n_of] + comp) if unit_fe else comp
    blocks_r, blocks_c, blocks_v = [rows], [col_u], [np.ones(2 * N)]
    if time_fe:
        blocks_r.append(rows); blocks_c.append(nu + 2 * Dd["time"][n_of] + comp); blocks_v.append(np.ones(2 * N))
    for k in range(Ks):
        blocks_r.append(rows); blocks_c.append(np.full(2 * N, nu + nt + k)); blocks_v.append(sl[:, k].reshape(-1))
    for k in range(Kc):
        blocks_r.append(rows); blocks_c.append(np.full(2 * N, nu + nt + Ks + k)); blocks_v.append(ql[:, k].reshape(-1))
    Z = sp.csr_matrix((np.concatenate(blocks_v), (np.concatenate(blocks_r), np.concatenate(blocks_c))), shape=(2 * N, P))
    yv = y.reshape(-1)
    R = np.zeros(P)
    if unit_fe:
        R[:nu] = ridge_u
    if time_fe:
        R[nu:nu + nt] = ridge_t if unit_fe else 0.0
    # Hessian-block index pattern (2x2 blocks on the diagonal of the 2N x 2N row space)
    bi = np.concatenate([2 * np.arange(N), 2 * np.arange(N), 2 * np.arange(N) + 1, 2 * np.arange(N) + 1])
    bj = np.concatenate([2 * np.arange(N), 2 * np.arange(N) + 1, 2 * np.arange(N), 2 * np.arange(N) + 1])

    def terms(par):
        th = (Z @ par).reshape(N, 2)
        k = np.hypot(th[:, 0], th[:, 1])
        nll = float(np.sum(-np.sum(th * y, axis=1) + logI0(k)))
        Ak = A(k)
        u = th / np.maximum(k, 1e-12)[:, None]
        small = k < 1e-6
        u = np.where(small[:, None], np.array([1.0, 0.0]), u)
        r = (Ak[:, None] * u - y).reshape(-1)
        g = Z.T @ r
        dA = 1 - Ak / np.maximum(k, 1e-12) - Ak**2
        Mpar = np.where(small, 0.5, dA)
        Mperp = np.where(small, 0.5, Ak / np.maximum(k, 1e-12))
        m00 = Mperp + (Mpar - Mperp) * u[:, 0] ** 2
        m01 = (Mpar - Mperp) * u[:, 0] * u[:, 1]
        m11 = Mperp + (Mpar - Mperp) * u[:, 1] ** 2
        M = sp.csr_matrix((np.concatenate([m00, m01, m01, m11]), (bi, bj)), shape=(2 * N, 2 * N))
        H = (Z.T @ M @ Z).tocsc()
        return nll + 0.5 * np.sum(R * par**2), g + R * par, H + sp.diags(R), nll

    par = np.zeros(P)
    m0 = y.mean(axis=0); r0 = np.linalg.norm(m0)
    h0 = m0 / max(r0, 1e-12) * Ainv(np.array([r0]))[0] if r0 > 0 else np.zeros(2)
    if unit_fe:
        par[:nu] = np.tile(h0, Dd["n_units"])
    else:
        par[:2] = h0
    f, g, H, nll = terms(par)
    conv = False
    it = 0
    for it in range(1, maxiter + 1):
        step = schur_solve(H, g, nu, nu + nt, bs=2) if schur else spla.spsolve(H, g)
        dec = float(g @ step)
        t = 1.0
        while True:
            pn = par - t * step
            fn, gn, Hn, nlln = terms(pn)
            if np.isfinite(fn) and fn <= f - 1e-4 * t * dec:
                break
            t *= 0.5
            if t < 1e-12:
                break
        par, f, g, H, nll = pn, fn, gn, Hn, nlln
        if dec / 2 < tol * max(1.0, abs(f)):
            conv = True
            break
    # covariance of the shared block
    E = sp.csc_matrix((np.ones(p_sh), (np.arange(nu + nt, P), np.arange(p_sh))), shape=(P, p_sh))
    if p_sh and schur:
        cov = schur_solve(H, np.zeros(P), nu, nu + nt, bs=2, shared_cov=p_sh)[1]
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    elif p_sh:
        Y = spla.spsolve(H, E)
        Y = Y.toarray() if sp.issparse(Y) else np.asarray(Y).reshape(P, p_sh)
        cov = Y[nu + nt:]
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    else:
        se = np.zeros(0)
    b = par[nu + nt:]
    k_par = nu + max(nt - (2 if (unit_fe and time_fe) else 0), 0) + p_sh
    return dict(a=b[:Ks], J=b[Ks:], se_a=se[:Ks], se_J=se[Ks:], h=par[:nu].reshape(-1, 2), g=par[nu:nu + nt].reshape(-1, 2),
                nll=nll, k=k_par, N=N, bic=2 * nll + k_par * np.log(N), aic=2 * nll + 2 * k_par, conv=conv, n_iter=it,
                grad_norm=float(np.max(np.abs(g))))


# ----------------------------------------------------------------------------------
def pre_pipeline_dir(S, tauM_max, rows="fixed", m=None):
    """PRE for D = 2: m_t, G_t = M^{-1}(m_t) = m_hat A^{-1}(|m_t|), OLS G_t = H + sum J m_{t-tau},
    AIC (A61) with D = 2.
    rows = 'fixed'   : the same rows t > tauM_max for every tau_M (comparable likelihoods);
           'variable': rows t = tau_M+1..N_T for each tau_M, as written in the PRE (Eqs. A36-A61).
    m: optional precomputed magnetisation series (T, 2) instead of the mean of S over units."""
    if rows == "variable":
        return _pre_pipeline_dir_variable(S, tauM_max, m)
    D = 2
    m = np.nanmean(S, axis=0) if m is None else m                    # (T, 2)
    r = np.hypot(m[:, 0], m[:, 1])
    G = m / np.maximum(r, 1e-12)[:, None] * Ainv(r)[:, None]
    T = len(m)
    rows_G, rows_M = [], []
    for t in range(tauM_max, T):
        rows_G.append(G[t]); rows_M.append(m[t - tauM_max:t][::-1])
    Gt = np.array(rows_G); Mt = np.array(rows_M)                    # (NT,2), (NT,tauM_max,2)
    NT = len(Gt)
    out = {}
    for tauM in range(0, tauM_max + 1):
        # stack components: rows (t, comp); columns H_x, H_y, J_1..J_tauM
        X = np.zeros((2 * NT, 2 + tauM))
        X[0::2, 0] = 1; X[1::2, 1] = 1
        for tau in range(tauM):
            X[0::2, 2 + tau] = Mt[:, tau, 0]; X[1::2, 2 + tau] = Mt[:, tau, 1]
        yv = Gt.reshape(-1)
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        res = yv - X @ coef
        sigma2 = float(np.sum(res**2) / (D * NT))
        eps2 = NT * sigma2 / max(NT - tauM, 1)
        aic = 2 * (tauM + D + 1) / (D * (NT - tauM)) + np.log(eps2)
        out[tauM] = dict(J=coef[2:], H=coef[:2], sigma2=sigma2, aic=float(aic), NT=NT)
    tsel = min(out, key=lambda k: out[k]["aic"])
    delay = int(np.argmax(np.abs(out[tsel]["J"])) + 1) if tsel else 0
    return out, tsel, delay


def _pre_pipeline_dir_variable(S, tauM_max, m=None):
    """Literal PRE protocol: for each tau_M, rows t = tau_M+1..N_T (A36), H and J from (A45-A48),
    sigma^2 = <|residual|^2>_t / D (A49), AIC (A61) with N_T the number of frames of m."""
    D = 2
    m = np.nanmean(S, axis=0) if m is None else m
    r = np.hypot(m[:, 0], m[:, 1])
    G = m / np.maximum(r, 1e-12)[:, None] * Ainv(r)[:, None]
    NT = len(m)
    out = {}
    for tauM in range(0, tauM_max + 1):
        t = np.arange(tauM, NT)
        X = np.zeros((2 * len(t), 2 + tauM))
        X[0::2, 0] = 1; X[1::2, 1] = 1
        for tau in range(1, tauM + 1):
            X[0::2, 1 + tau] = m[t - tau, 0]; X[1::2, 1 + tau] = m[t - tau, 1]
        yv = G[t].reshape(-1)
        ok = np.isfinite(yv) & np.isfinite(X).all(axis=1)
        ok = ok.reshape(-1, 2).all(axis=1).repeat(2)
        coef, *_ = np.linalg.lstsq(X[ok], yv[ok], rcond=None)
        res = yv[ok] - X[ok] @ coef
        n_rows = int(ok.sum() // 2)
        sigma2 = float(np.sum(res**2) / (D * n_rows))
        aic = 2 * (tauM + D + 1) / (D * (NT - tauM)) + np.log(NT * sigma2 / (NT - tauM))
        out[tauM] = dict(J=coef[2:], H=coef[:2], sigma2=sigma2, aic=float(aic), NT=n_rows)
    tsel = min(out, key=lambda k: out[k]["aic"])
    delay = int(np.argmax(np.abs(out[tsel]["J"])) + 1) if tsel else 0
    return out, tsel, delay


# ----------------------------------------------------------------------------------
def sim_dir(N, T, rng, H=(0.0, 0.0), a=(), Jloc=(), Jglob=(), box=20.0, v0=0.3, k_nn=3, burn=300,
            turn_corr=0.0, turn_sd=0.0):
    """Units with von Mises self + social dynamics, optionally plus correlated turning noise
    (an OU 'angular velocity' with correlation turn_corr) that gives long self memory outside the
    model family. Positions move at constant speed v0 in a periodic box."""
    a = np.asarray(a, float); Jloc = np.asarray(Jloc, float); Jglob = np.asarray(Jglob, float)
    K = max(len(a), len(Jloc), len(Jglob), 1)
    tot = T + burn
    S = np.zeros((N, tot, 2)); P = np.zeros((N, tot, 2))
    P[:, 0] = rng.uniform(0, box, (N, 2))
    phi0 = rng.uniform(-np.pi, np.pi, N)
    S[:, 0] = np.c_[np.cos(phi0), np.sin(phi0)]
    qloc = np.zeros((N, tot, 2)); qglob = np.zeros((N, tot, 2))
    omega = np.zeros(N)
    H = np.asarray(H, float)
    for t in range(tot):
        if t > 0:
            th = np.repeat(H[None, :], N, axis=0)
            for tau in range(1, K + 1):
                s = t - tau
                if s < 0:
                    continue
                if tau <= len(a):
                    th = th + a[tau - 1] * S[:, s]
                if tau <= len(Jloc):
                    th = th + Jloc[tau - 1] * qloc[:, s]
                if tau <= len(Jglob):
                    th = th + Jglob[tau - 1] * qglob[:, s]
            k = np.hypot(th[:, 0], th[:, 1])
            mu = np.arctan2(th[:, 1], th[:, 0])
            ang = rng.vonmises(mu, np.maximum(k, 1e-9))
            if turn_sd > 0:
                omega = turn_corr * omega + turn_sd * np.sqrt(1 - turn_corr**2) * rng.standard_normal(N)
                prev = np.arctan2(S[:, t - 1, 1], S[:, t - 1, 0])
                ang = prev + omega if len(a) == 0 and len(Jloc) == 0 and len(Jglob) == 0 else ang + omega
            S[:, t] = np.c_[np.cos(ang), np.sin(ang)]
            P[:, t] = (P[:, t - 1] + v0 * S[:, t - 1]) % box
        d = P[:, t][:, None, :] - P[:, t][None, :, :]
        d -= box * np.round(d / box)
        D = np.hypot(d[..., 0], d[..., 1]); np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1)[:, :k_nn]
        qloc[:, t] = S[nn, t].mean(axis=1)
        qglob[:, t] = (S[:, t].sum(axis=0)[None, :] - S[:, t]) / (N - 1)
    return S[:, burn:], P[:, burn:]
