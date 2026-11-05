"""Validation of the fixed-effects von Mises self + social fitter (fit_dir_fe) on synthetic
multi-agent tracks with heterogeneous individual fields h_i, self memory, common per-frame
drive, and optional local delayed directional coupling."""
import safe  # noqa: F401
import time

import numpy as np
from scipy.spatial import cKDTree

import collective_dir as cd


def simulate(rng, N=220, T=150, a=(0.6, 0.2), J=0.0, k_nn=3, h_mean=(0.4, 0.0), h_sd=0.4, drive_sd=0.5,
             drive_rho=0.7, box=1331.0, speed=2.5, jitter_sd=0.0, life=40):
    """Agents live ~life frames (finite track durations); returns directions S and positions P
    of the measured steps (true positions + optional rigid frame jitter)."""
    U = N * T // life                                # total distinct tracks over the recording
    start = rng.integers(-life, T, U)
    dur = rng.geometric(1 / life, U) + 8
    h = np.asarray(h_mean) + h_sd * rng.standard_normal((U, 2))
    a = np.asarray(a, float)
    g = np.zeros((T, 2))
    for t in range(1, T):
        g[t] = drive_rho * g[t - 1] + drive_sd * np.sqrt(1 - drive_rho**2) * rng.standard_normal(2)
    Xp = np.full((U, T + 1, 2), np.nan)              # true positions
    St = np.full((U, T, 2), np.nan)                  # true directions
    for t in range(T):
        alive = np.where((start <= t) & (start + dur > t))[0]
        new = alive[np.isnan(Xp[alive, t, 0])]
        Xp[new, t] = rng.uniform(0, box, (len(new), 2))
        th = h[alive] + g[t]
        for k, ak in enumerate(a, 1):
            if t - k >= 0:
                prev = St[alive, t - k]
                th = th + ak * np.nan_to_num(prev)
        if J and t >= 1:
            src = np.where(~np.isnan(St[:, t - 1, 0]))[0]
            if len(src) > k_nn:
                tree = cKDTree(Xp[src, t - 1])
                _, nb = tree.query(Xp[alive, t], k=k_nn + 1)
                nb = src[nb]
                keep = nb != alive[:, None]
                use = keep & (np.cumsum(keep, axis=1) <= k_nn)
                q = (St[nb, t - 1] * use[..., None]).sum(1) / use.sum(1, keepdims=True)
                th = th + J * q
        kap = np.hypot(th[:, 0], th[:, 1])
        ang = rng.vonmises(np.arctan2(th[:, 1], th[:, 0]), np.maximum(kap, 1e-9))
        s = np.c_[np.cos(ang), np.sin(ang)]
        St[alive, t] = s
        Xp[alive, t + 1] = np.clip(Xp[alive, t] + speed * rng.gamma(2.0, 0.5, len(alive))[:, None] * s, 0, box)
    eps = jitter_sd * rng.standard_normal((T + 1, 2))
    Xm = Xp + eps[None]                              # measured positions
    V = Xm[:, 1:] - Xm[:, :-1]
    ell = np.hypot(V[..., 0], V[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where((ell > 0)[..., None], V / ell[..., None], np.nan)
    return S, Xm[:, :-1], V


def register(V):
    from scipy.ndimage import uniform_filter1d
    d = np.nanmedian(V, axis=0)
    tr = np.c_[uniform_filter1d(d[:, 0], 21, mode="nearest"), uniform_filter1d(d[:, 1], 21, mode="nearest")]
    Vr = V - (d - tr)[None]
    ell = np.hypot(Vr[..., 0], Vr[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((ell > 0)[..., None], Vr / ell[..., None], np.nan)


def test_dir_fe_equivalence_without_fixed_effects():
    """Without fixed effects, fit_dir_fe must coincide with fit_dir (common external field H)."""
    rng = np.random.default_rng(11)
    S, P, _ = simulate(rng, N=80, T=80, J=0.3)
    Dd = cd.build_dir_fe(S, P, 2, 2, mode="local", min_rows=1)
    f1 = cd.fit_dir_fe(Dd, 2, 2, unit_fe=False, time_fe=False)
    f2 = cd.fit_dir(Dd["y"], Dd["selfl"], Dd["socl"], 2, 2)
    print("fit_dir_fe vs fit_dir (no FE): dNLL", abs(f1["nll"] - f2["nll"]),
          " max|dJ|", np.max(np.abs(f1["J"] - f2["J"])), " max|da|", np.max(np.abs(f1["a"] - f2["a"])))
    assert abs(f1["nll"] - f2["nll"]) < 1e-4
    assert np.max(np.abs(f1["J"] - f2["J"])) < 1e-4
    assert np.max(np.abs(f1["a"] - f2["a"])) < 1e-4


def test_dir_fe_common_drive_control():
    """Verify that time fixed effects absorb a common frame drive while preserving true local coupling."""
    rng = np.random.default_rng(12)
    S, P, _ = simulate(rng, N=90, T=90, J=0.0, drive_sd=0.6)
    Dd = cd.build_dir_fe(S, P, 2, 1, mode="local", k_nn=3)
    f_no_tfe = cd.fit_dir_fe(Dd, 2, 1, unit_fe=True, time_fe=False)
    f_tfe = cd.fit_dir_fe(Dd, 2, 1, unit_fe=True, time_fe=True)
    print(f"J=0 + common drive: without time FE J1={f_no_tfe['J'][0]:+.3f}, with time FE J1={f_tfe['J'][0]:+.3f}")
    assert f_tfe["conv"]
    assert abs(f_tfe["J"][0]) < abs(f_no_tfe["J"][0])


def main():
    rng = np.random.default_rng(11)
    cases = [("A  local J=0.4", dict(J=0.4)), ("B  no coupling, common drive", dict(J=0.0)),
             ("C  no coupling + image jitter", dict(J=0.0, jitter_sd=0.4))]
    for name, kw in cases:
        S, P, V = simulate(rng, **kw)
        variants = [("raw", S)] + ([("registered", register(V))] if kw.get("jitter_sd") else [])
        for vname, SS in variants:
            Dd = cd.build_dir_fe(SS, P, 2, 2, mode="local", k_nn=3)
            for ufe, tfe in ((True, False), (True, True)):
                t0 = time.time()
                f = cd.fit_dir_fe(Dd, 2, 2, unit_fe=ufe, time_fe=tfe)
                print(f"{name:32s} {vname:10s} unitFE={ufe} timeFE={tfe}: a={np.round(f['a'], 3)} J={np.round(f['J'], 3)} "
                      f"se_J={np.round(f['se_J'], 3)} conv={f['conv']} it={f['n_iter']} |grad|={f['grad_norm']:.1e} "
                      f"N={f['N']} ({time.time() - t0:.1f}s)")
    test_dir_fe_equivalence_without_fixed_effects()


if __name__ == "__main__":
    main()
