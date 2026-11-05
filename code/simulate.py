"""Synthetic flight-sequence generators and 2D trajectory segmentation routines.

All flight sequences are returned in x = ln(ℓ/ℓ_min) ∈ [0, L] unless stated otherwise.
"""
import numpy as np
from scipy.stats import norm
from maxent_levy import pl_sample


def trunc_pareto_ppf(u, mu, L):
    """Quantile of x for ℓ ~ truncated Pareto (exponent μ) on [ℓ_min, ℓ_min e^L]."""
    a = mu - 1.0
    if abs(a) < 1e-12:
        return u * L
    return -np.log1p(-u * (-np.expm1(-a * L))) / a


def iid_pareto(mu, L, T, rng, n_seq=1):
    return pl_sample(np.full((n_seq, T), 1.0 - mu), L, rng)


def ar_gaussian(phi, T, rng, n_seq=1, burn=1000):
    """Gaussian AR(p) standardized to unit marginal variance."""
    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    tot = T + burn
    z = np.zeros((n_seq, tot))
    e = rng.standard_normal((n_seq, tot))
    for t in range(p, tot):
        z[:, t] = z[:, t - p:t][:, ::-1] @ phi + e[:, t]
    z = z[:, burn:]
    return (z - z.mean(axis=1, keepdims=True)) / z.std(axis=1, keepdims=True)


def copula_ar_pareto(phi, mu, L, T, rng, n_seq=1):
    """Gaussian AR(p) copula with truncated-Pareto marginals: finite Markov order p
    outside the causal MaxEnt exponential family (misspecification benchmark)."""
    z = ar_gaussian(phi, T, rng, n_seq)
    return trunc_pareto_ppf(norm.cdf(z), mu, L)


def fgn(H, T, rng, n_seq=1):
    """Fractional Gaussian noise (Davies–Harte circulant embedding)."""
    k = np.arange(T)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    lam = np.real(np.fft.fft(row))
    lam = np.clip(lam, 0, None)
    M = len(row)
    out = np.empty((n_seq, T))
    for i in range(n_seq):
        W = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        f = np.fft.fft(np.sqrt(lam / M) * W)
        out[i] = np.real(f[:T])
    return out / out.std(axis=1, keepdims=True)


def copula_fgn_pareto(H, mu, L, T, rng, n_seq=1):
    """Long-range memory (no finite Markov order) with truncated-Pareto marginals."""
    return trunc_pareto_ppf(norm.cdf(fgn(H, T, rng, n_seq)), mu, L)


def trunc_exp_ell_sample(lam, L, rng, size):
    """Sample ℓ/ℓ_min ~ exponential(lam) truncated to [1, e^L]; returns x = ln(ℓ/ℓ_min)."""
    u = rng.random(size)
    a, b = np.exp(-lam * 1.0), np.exp(-lam * np.exp(L))
    ell = -np.log(a - u * (a - b)) / lam
    return np.log(ell)


def ccrw_hmm(lams, P, L, T, rng, n_seq=1):
    """Composite correlated random walk: hidden Markov model switching between
    'intensive/extensive' exponential step scales. Geometrically decaying dependence."""
    lams = np.asarray(lams, dtype=float)
    P = np.asarray(P, dtype=float)
    M = len(lams)
    out = np.empty((n_seq, T))
    states = np.empty((n_seq, T), dtype=int)
    for i in range(n_seq):
        s = rng.integers(M)
        for t in range(T):
            states[i, t] = s
            s = rng.choice(M, p=P[s])
        for m in range(M):
            idx = states[i] == m
            out[i, idx] = trunc_exp_ell_sample(lams[m], L, rng, idx.sum())
    return out, states


# ----------------------------------------------------------------------------------
# 2D trajectories and flight segmentation
# ----------------------------------------------------------------------------------

def levy_walk_2d(flight_x, v, dt, noise, rng, lmin=1.0):
    """Constant-speed 2D Lévy walk: flights of length ℓ = lmin e^x with isotropic directions,
    sampled at interval dt with additive Gaussian localization noise.
    Returns positions (n, 2)."""
    ell = lmin * np.exp(np.asarray(flight_x))
    ang = rng.uniform(0, 2 * np.pi, len(ell))
    durations = ell / v
    t_turn = np.concatenate([[0], np.cumsum(durations)])
    pos_turn = np.vstack([[0, 0], np.cumsum(np.c_[ell * np.cos(ang), ell * np.sin(ang)], axis=0)])
    ts = np.arange(0, t_turn[-1], dt)
    k = np.searchsorted(t_turn, ts, side="right") - 1
    k = np.clip(k, 0, len(ell) - 1)
    frac = (ts - t_turn[k]) * v
    pos = pos_turn[k] + np.c_[frac * np.cos(ang[k]), frac * np.sin(ang[k])]
    return pos + noise * rng.standard_normal(pos.shape)


def segment_flights(pos, eps):
    """Rectangular corridor model (Rhee et al. 2011; Humphries et al. 2013): a flight is the
    longest stretch of track whose intermediate points stay within distance eps of the chord
    joining its endpoints. Greedy scan. Returns flight lengths and start indices."""
    n = len(pos)
    lengths, starts = [], []
    i = 0
    while i < n - 1:
        j = i + 1
        while j + 1 < n:
            a, b = pos[i], pos[j + 1]
            d = b - a
            nd = np.hypot(*d)
            seg = pos[i + 1:j + 1] - a
            if nd == 0:
                dev = np.hypot(seg[:, 0], seg[:, 1])
            else:
                dev = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / nd
                # intermediate points must also project inside the chord (no backtracking)
                proj = (seg @ d) / nd
                dev = np.where((proj < -eps) | (proj > nd + eps), np.inf, dev)
            if np.all(dev < eps):
                j += 1
            else:
                break
        lengths.append(np.hypot(*(pos[j] - pos[i])))
        starts.append(i)
        i = j
    return np.array(lengths), np.array(starts)


def merge_collinear(pos, starts, theta_c):
    """Merge consecutive segmented pieces whose heading change |Δθ| < theta_c.

    Noise-induced splits have Δθ ≈ 0 and manufacture spurious positive lag-1 correlation;
    merging nearly collinear true flights is an i.i.d. thinning that preserves renewal.
    Returns merged flight lengths and start indices.
    """
    starts = list(starts)
    ends = starts[1:] + [len(pos) - 1]
    segs = [[s, e] for s, e in zip(starts, ends)]
    out = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = out[-1]
        d1 = pos[pe] - pos[ps]
        d2 = pos[e] - pos[s]
        a = np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2)
        if abs(a) < theta_c:
            out[-1][1] = e
        else:
            out.append([s, e])
    lengths = np.array([np.hypot(*(pos[e] - pos[s])) for s, e in out])
    return lengths, np.array([s for s, _ in out])
