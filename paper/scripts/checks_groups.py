"""checks_groups.py -- finite-group effect of the time fields (Appendix G, Sec. V.C).

    python checks_groups.py             # a few minutes, one process

Synthetic groups in which every unit is coupled to one other unit, drawn at random in each frame, with
unit fields and a common drive. Headings (von Mises): pigeon-like and weakly concentrated couplings,
with and without time fields, and a null coupling. Log-speeds: self coupling 0.4, social 0.3.
Prints mean and s.d. of the estimates over replicates; writes ../data/checks_groups.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
import safe  # noqa: E402,F401

import numpy as np  # noqa: E402

import collective as co  # noqa: E402
import collective_dir as cd  # noqa: E402
from maxent_levy import pl_sample  # noqa: E402

rng = np.random.default_rng(1)
OUT = {}


def sim_headings(N, T, a, J, gmag, hmag, burn=200):
    """Von Mises units: theta_i^t = h_i + g_t + sum a_tau n_i^{t-tau} + sum J_tau n_{nb(i)}^{t-tau}."""
    K = max(len(a), len(J))
    TT = T + burn
    ang = np.zeros((N, TT)); ang[:, :K] = rng.uniform(0, 2 * np.pi, N)[:, None]
    u = rng.uniform(0, 2 * np.pi, N)
    h = hmag * np.stack([np.cos(u), np.sin(u)], 1)
    phi_g = np.cumsum(rng.normal(0, 0.05, TT)); gm = gmag * (1 + 0.5 * np.sin(np.arange(TT) / 37.0))
    g = np.stack([gm * np.cos(phi_g), gm * np.sin(phi_g)], 1)
    nb = (np.arange(N)[:, None] + rng.integers(1, N, (N, TT))) % N          # one other unit per frame
    n = lambda tt: np.stack([np.cos(ang[:, tt]), np.sin(ang[:, tt])], 1)
    for t in range(K, TT):
        th = h + g[t]
        for k, ak in enumerate(a, 1):
            th = th + ak * n(t - k)
        for k, jk in enumerate(J, 1):
            th = th + jk * n(t - k)[nb[:, t - k]]
        ang[:, t] = rng.vonmises(np.arctan2(th[:, 1], th[:, 0]), np.hypot(th[:, 0], th[:, 1]))
    ang, nb = ang[:, burn:], nb[:, burn:]
    U = np.stack([np.cos(ang), np.sin(ang)], -1)
    tt, ii = np.meshgrid(np.arange(K, T), np.arange(N), indexing="ij"); ii = ii.ravel(); tt = tt.ravel()
    return dict(y=U[ii, tt], selfl=np.stack([U[ii, tt - k] for k in range(1, len(a) + 1)], 1),
                socl=np.stack([U[nb[ii, tt - k], tt - k] for k in range(1, len(J) + 1)], 1),
                unit=ii, time=tt - K, n_units=N, n_times=T - K, Ks_max=len(a), Kc_max=len(J))


def headings(tag, a, J, gmag, hmag, Ns, reps=8, T=3000, time_fe=(True,)):
    for N in Ns:
        for tfe in time_fe:
            est = []
            for _ in range(reps):
                f = cd.fit_dir_fe(sim_headings(N, T, a, J, gmag, hmag), len(a), len(J), unit_fe=True,
                                  time_fe=tfe, schur=tfe)
                est.append(np.concatenate([f["a"], f["J"]]))
            est = np.array(est)
            OUT[f"{tag}_N{N}_tfe{int(tfe)}"] = dict(planted=list(a) + list(J), mean=est.mean(0).tolist(), sd=est.std(0).tolist())
            print(f"{tag:15s} N={N:2d} time fields={tfe!s:5s} planted {list(a) + list(J)}  mean {np.round(est.mean(0), 3)}  "
                  f"ratio {np.round(est.mean(0) / np.where(np.array(list(a) + list(J)) == 0, np.nan, list(a) + list(J)), 2)}",
                  flush=True)


def sim_speeds(N, T, a, J, L, gsd, burn=200):
    TT = T + burn
    X = np.full((N, TT), L / 2)
    h = rng.uniform(-1.3, -0.5, N)
    g = np.zeros(TT)
    for t in range(1, TT):
        g[t] = 0.9 * g[t - 1] + gsd * rng.normal()
    nb = (np.arange(N)[:, None] + rng.integers(1, N, (N, TT))) % N
    for t in range(1, TT):
        X[:, t] = pl_sample(h + g[t] + a * (X[:, t - 1] - L / 2) + J * (X[nb[:, t - 1], t - 1] - L / 2), L, rng)
    X, nb = X[:, burn:], nb[:, burn:]
    ii, tt = np.meshgrid(np.arange(N), np.arange(1, T), indexing="ij"); ii = ii.ravel(); tt = tt.ravel()
    y = X[ii, tt]; c = float(y.mean())
    Xd = np.column_stack([X[ii, tt - 1] - c, X[nb[ii, tt - 1], tt - 1] - c])
    return co.Design(y, Xd, ii, tt - 1, np.zeros(len(y), int), 1, 1, N, T - 1, c)


def speeds(Ns, reps=8, T=3000, a=0.4, J=0.3):
    L = float(np.log(100))
    for N in Ns:
        for tfe in (False, True):
            est = np.array([[q.a[0], q.J[0]] for q in
                            (co.fit_collective(sim_speeds(N, T, a, J, L, 0.15 if tfe else 0.0), 1, 1, L=L, unit_fe=True,
                                               time_fe=tfe, schur=tfe) for _ in range(reps))])
            OUT[f"speeds_N{N}_tfe{int(tfe)}"] = dict(planted=[a, J], mean=est.mean(0).tolist(), sd=est.std(0).tolist())
            print(f"log-speeds      N={N:2d} time fields={tfe!s:5s} planted [{a}, {J}]  mean {np.round(est.mean(0), 3)}", flush=True)


if __name__ == "__main__":
    PIG_A, PIG_J = (185.0, -82.0, 21.0), (17.6, -11.0, 3.8)
    headings("pigeon-like", PIG_A, PIG_J, 5.0, 1.0, (3, 5, 8))
    headings("pigeon, no drive", PIG_A, PIG_J, 0.0, 1.0, (5,), time_fe=(False, True))
    headings("pigeon, null", PIG_A, (0.0, 0.0, 0.0), 5.0, 1.0, (3, 5))
    headings("weak", (0.7,), (0.45,), 0.5, 0.3, (3, 5, 8))
    headings("weak", (0.7,), (0.45,), 0.5, 0.3, (26,), reps=6, T=1500)
    speeds((3, 5, 8, 26))
    json.dump(OUT, open(os.path.join(HERE, "..", "data", "checks_groups.json"), "w"))
