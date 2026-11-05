"""
compute.py -- the few new computations needed by the paper (everything else is read
from ../../results/*.json, written by the experiment scripts in ../../code).

    python compute.py validation     # planted vs inferred, 200 random single walkers
    python compute.py phase          # K = 1 phase-diagram coordinates of larvae and termites
    python compute.py all

Outputs go to ../data/.  Resource guard: single-threaded BLAS, below-normal priority
(code/safe.py), one process.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "code")
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "..", "results")
sys.path.insert(0, CODE)
import safe  # noqa: E402,F401

import numpy as np  # noqa: E402

from maxent_levy import pl_sample, scan_orders, select, _invert_mean, scan_orders_fe  # noqa: E402


# ----------------------------------------------------------------------------- validation
def simulate_many(hp, J, L, T, rng, burn=500):
    """Vectorised causal chains, one per row: theta_t = h'_r + sum_tau J_{r,tau} x_{t-tau}."""
    R, K = J.shape
    X = np.empty((R, T + burn))
    X[:, :K] = L / 2
    for t in range(K, T + burn):
        theta = hp + np.einsum("rk,rk->r", J, X[:, t - K:t][:, ::-1])
        X[:, t] = pl_sample(theta, L, rng)
    return X[:, burn:]


def validation(n_rep=200, T=3000, K=3, Kmax=8, seed=20261024):
    """Planted vs inferred, in the spirit of Fig. 1 of Agliari et al. PRE 110, 024301 (2024).

    Planted: bare exponent mu ~ U(1.3, 2.5) (h' = 1 - mu), K = 3 couplings with random sign and
    |J_tau| ~ U(0.04, 0.20), dynamic range l_max/l_min = 10^2 (L = ln 100, as in the larvae).
    Inferred: causal MaxEnt fit at the planted K (scatter, z-scores) and depth selection by
    BIC and AIC over K = 0..Kmax on common rows."""
    rng = np.random.default_rng(seed)
    L = float(np.log(100.0))
    mu = rng.uniform(1.3, 2.5, n_rep)
    J = rng.uniform(0.04, 0.20, (n_rep, K)) * rng.choice([-1.0, 1.0], (n_rep, K))
    hp = 1.0 - mu
    X = simulate_many(hp, J, L, T, rng)
    rows = []
    for r in range(n_rep):
        x = X[r]
        fits = scan_orders([x], L, Kmax)
        kb, _ = select(fits, "bic")
        ka, _ = select(fits, "aic")
        f = fits[K]
        c = f.c
        g = np.concatenate([[1.0], -c * np.ones(K)])          # h' = h - c sum J
        cov = np.linalg.inv(f.H)
        hp_hat = float(f.h - c * f.J.sum())
        se_hp = float(np.sqrt(g @ cov @ g))
        rows.append(dict(mu=float(mu[r]), hp=float(hp[r]), J=J[r].tolist(), hp_hat=hp_hat, se_hp=se_hp,
                         J_hat=f.J.tolist(), se_J=f.se[1:].tolist(), K_bic=int(kb), K_aic=int(ka),
                         phi=float(J[r].sum() * L / (mu[r] - 1)),
                         mu_iid=float(1 - _invert_mean(x.mean(), L))))
    out = dict(L=L, T=T, K=K, Kmax=Kmax, n_rep=n_rep, rows=rows)
    with open(os.path.join(DATA, "validation.json"), "w") as fh:
        json.dump(out, fh)
    kb = np.array([q["K_bic"] for q in rows]); ka = np.array([q["K_aic"] for q in rows])
    z = np.concatenate([(np.array(q["J_hat"]) - np.array(q["J"])) / np.array(q["se_J"]) for q in rows])
    print(f"validation: K*=K (BIC) {np.mean(kb == K):.2f}, (AIC) {np.mean(ka == K):.2f}; "
          f"z mean {z.mean():+.3f} sd {z.std():.3f}")


# ----------------------------------------------------------------------------- phase coordinates
def phase():
    """K = 1 coordinates (mu_bare, J_1 L) of every larva (fixed effects per condition, as in
    exp_drosophila.phase_index) and of every termite (one fit per track)."""
    from exp_drosophila import get_flights, to_x_list, COND_ORDER
    key, flights, _, _ = get_flights(0.5)
    larvae = []
    for cond in COND_ORDER:
        seqs = [flights[f] for f in key.csv[key.Treatment == cond] if len(flights[f]) > 16]
        lmax = np.concatenate(seqs).max() * 1.001
        xs, L, _ = to_x_list(seqs, 1.0, lmax)
        fe = scan_orders_fe(xs, L, 1)
        J1 = float(fe[1].extra["J"][0]); se = float(fe[1].extra["se_J"][0])
        c = np.array([x.mean() for x in xs])
        mub = 1 - (fe[1].extra["h_i"] - J1 * c)
        mui = np.array([1 - _invert_mean(x.mean(), L) for x in xs])
        larvae.append(dict(cond=cond, L=L, J1=J1, seJ1=se, mu_bare=mub.tolist(), mu_iid=mui.tolist(),
                           n=[len(x) for x in xs]))
    with open(os.path.join(DATA, "larvae_K1.json"), "w") as fh:
        json.dump(larvae, fh)
    R = json.load(open(os.path.join(RES, "termites.json")))
    term = []
    for r in R:
        x = np.array(r["x"]); L = r["L"]
        f = scan_orders([x], L, 1)[1]
        Jv = float(f.J[0])
        term.append(dict(id=r["id"], density=r["density"], L=L, J1=Jv, seJ1=float(f.se[1]),
                         mu_bare=float(1 - (f.h - f.c * Jv)), mu_iid=float(1 - _invert_mean(x.mean(), L)),
                         K=r["K_bic"]))
    with open(os.path.join(DATA, "termites_K1.json"), "w") as fh:
        json.dump(term, fh)
    print("phase coordinates written")


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    t0 = time.time()
    if "validation" in what or "all" in what:
        validation()
    if "phase" in what or "all" in what:
        phase()
    print(f"done in {time.time() - t0:.0f} s")
