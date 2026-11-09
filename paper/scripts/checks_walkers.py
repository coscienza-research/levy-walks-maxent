"""checks_walkers.py -- robustness numbers quoted in Secs. V.A-B and Appendix F.

    python checks_walkers.py            # larvae and termites (about 10 min, one process)

Larvae: exact K = 1 stationary law and simulated K* chains against the observed apparent exponent;
fits conditioned on l >= l_min (flights below l_min dropped as responses, kept at x = 0 as lags).
Termites: K = 1 and K* coordinates (J_tot), l_max = 49 mm, densities, conditioning on l >= l_min.
Outputs: ../data/checks_larvae.json, ../data/checks_termites.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "code")
RES = os.path.join(HERE, "..", "..", "results")
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, CODE)
import safe  # noqa: E402,F401

import numpy as np  # noqa: E402

from maxent_levy import scan_orders, scan_orders_fe, fit, _invert_mean, simulate_model, pl_logZ, build_design_fe  # noqa: E402

rng = np.random.default_rng(7)
KSTAR = {"BL/+ 22 Deg": 0, "shi/+ 22 Deg": 0, "New Blsens-rprhid_control": 1, "BL/+ 33 Deg": 1, "shi/+ 33 Deg": 1,
         "MB247/+": 2, "MB247-shi": 3, "BL-shi": 4, "BLsens-shi": 0, "New BLsens-rprhid": 2}     # Table II


def mu_hat_K1(mub, J, L, n=4001):
    """Apparent exponent of the exact stationary law of the K = 1 chain."""
    hp = 1 - mub
    x = np.linspace(0, L, n)
    lp = hp * x + pl_logZ(hp + J * x, L); lp -= lp.max(); p = np.exp(lp); p /= np.trapezoid(p, x)
    return 1 - _invert_mean(np.trapezoid(p * x, x), L)


def termites():
    R = json.load(open(os.path.join(RES, "termites.json")))
    out = []
    for r in R:
        x = np.array(r["x"]); L = r["L"]; Ks = r["K_bic"]
        f1 = scan_orders([x], L, 1)[1]
        J1 = float(f1.params[1])
        mub1 = 1 - (f1.params[0] - f1.c * J1)
        fk = scan_orders([x], L, Ks)[Ks]
        Jk = fk.params[1:Ks + 1]
        mubk = 1 - (fk.params[0] - fk.c * Jk.sum())
        muh_obs = 1 - _invert_mean(x.mean(), L)
        X = simulate_model(fk.params[0], Jk, L, fk.c, 6000, rng, n_seq=30)
        # l_max = 49 mm, the diameter of the dish
        L49 = float(np.log(49.0))
        g = scan_orders([x], L49, 1)[1]; J49 = float(g.params[1]); mub49 = 1 - (g.params[0] - g.c * J49)
        # conditioning on l_t >= l_min: rows with a clamped response dropped
        m = x[1:] > 0
        c = x.mean()
        fc = fit(x[1:][m], np.column_stack([np.ones(m.sum()), x[:-1][m] - c]), 1, L, c)
        Jc = float(fc.params[1]); mubc = 1 - (fc.params[0] - c * Jc)
        out.append(dict(id=r["id"], dens=r["density"], L=L, Ks=Ks, clip=r["clip"], J1=J1, se1=float(f1.se[1]), mub1=mub1,
                        phi1=J1 * L / (mub1 - 1), J1_Ks=float(Jk[0]), Jtot=float(Jk.sum()), mubk=mubk,
                        phik=float(Jk.sum()) * L / (mubk - 1), muh_obs=muh_obs, muh_K1=mu_hat_K1(mub1, J1, L),
                        muh_Ks=1 - _invert_mean(X.mean(), L), J49=J49, mub49=mub49, phi49=J49 * L49 / (mub49 - 1),
                        Jc=Jc, mubc=mubc, phic=Jc * L / (mubc - 1), muh_cond=1 - _invert_mean(x[x > 0].mean(), L)))
    json.dump(out, open(os.path.join(DATA, "checks_termites.json"), "w"))
    a = lambda k: np.array([t[k] for t in out])
    print(f"termites: J1(K=1) {np.median(a('J1')):.3f}, J_tot {np.median(a('Jtot')):.3f}; "
          f"Phi K=1 {np.median(a('phi1')):.2f}, K* {np.median(a('phik')):.2f} [{a('phik').min():.2f}, {a('phik').max():.2f}], "
          f"lmax=49 {np.median(a('phi49')):.2f} (>2: {np.mean(a('phi49') > 2):.3f}); conditioned J1 {np.median(a('Jc')):.3f} "
          f"mu_bare {np.median(a('mubc')):.2f} Phi {np.median(a('phic')):.2f}")
    for d in sorted(set(a("dens"))):
        s = a("dens") == d
        print(f"  density {d}: mu_bare {np.median(a('mub1')[s]):.2f} Phi {np.median(a('phi1')[s]):.2f} "
              f"mu_hat {np.median(a('muh_obs')[s]):.2f} | conditioned mu_bare {np.median(a('mubc')[s]):.2f} "
              f"Phi {np.median(a('phic')[s]):.2f}")


def larvae():
    from exp_drosophila import get_flights, COND_ORDER
    key, flights, _, _ = get_flights(0.5)
    out = []
    for cond in COND_ORDER:
        raw = [np.asarray(flights[f]) for f in key.csv[key.Treatment == cond] if len(flights[f]) > 16]
        L = float(np.log(np.concatenate(raw).max() * 1.001))          # l_min = 1 mm
        xs = [np.clip(np.log(s), 0, L) for s in raw]
        fe = scan_orders_fe(xs, L, 1)[1]
        n = len(xs)
        J1 = float(fe.extra["J"][0]); se1 = float(fe.extra["se_J"][0])
        c = np.array([x.mean() for x in xs])
        mub = 1 - (fe.extra["h_i"] - J1 * c)
        muh_obs = np.array([1 - _invert_mean(x.mean(), L) for x in xs])
        muh_K1 = np.array([mu_hat_K1(m, J1, L) for m in mub])
        # (b) clamped responses dropped; (c) clamped responses or lags dropped; (d) as (b) with a field
        # for the flights that follow a clamped one
        y, LAG, tid = build_design_fe(xs, 1)
        yr, LAGr, _ = build_design_fe([np.log(s) for s in raw], 1)
        lagraw = LAGr[:, 0] + np.repeat([np.log(s).mean() for s in raw], [len(s) - 1 for s in raw])
        OH = np.zeros((len(y), n)); OH[np.arange(len(y)), tid] = 1
        mresp = yr >= 0
        variants = {}
        for tag, m, extra in [("b", mresp, False), ("c", mresp & (lagraw >= 0), False), ("d", mresp, True)]:
            Z = np.hstack([OH[m], LAG[m, :1]] + ([(lagraw[m] < 0).astype(float)[:, None]] if extra else []))
            f = fit(y[m], Z, Z.shape[1] - 1, L, 0.0, init=np.concatenate([fe.extra["h_i"], [J1], [0.0] * extra]))
            Jv = float(f.params[n]); mubv = 1 - (f.params[:n] - Jv * c)
            variants[tag] = dict(J1=Jv, se=float(np.sqrt(np.diag(np.linalg.pinv(f.H)))[n]), mub_med=float(np.median(mubv)),
                                 phi_med=float(np.median(Jv * L / (mubv[mubv > 1] - 1))))
        Ks = KSTAR[cond]
        muh_Ks = np.full(n, np.nan)
        if Ks > 0:
            fk = scan_orders_fe(xs, L, Ks)[Ks]
            for i in range(n):
                X = simulate_model(fk.extra["h_i"][i], fk.extra["J"], L, c[i], 4000, rng, n_seq=10)
                muh_Ks[i] = 1 - _invert_mean(X.mean(), L)
        o = dict(cond=cond, n=n, L=L, clip=float(np.mean(np.concatenate(raw) < 1.0)), J1=J1, se1=se1,
                 mub_med=float(np.median(mub)), phi_med=float(np.median(J1 * L / (mub[mub > 1] - 1))),
                 muh_obs_med=float(np.median(muh_obs)), muh_K1_med=float(np.median(muh_K1)),
                 muh_Ks_med=float(np.nanmedian(muh_Ks)) if Ks else None, Kstar=Ks,
                 muh_cond_med=float(np.median([1 - _invert_mean(x[x > 0].mean(), L) for x in xs])), variants=variants)
        out.append(o)
        v = variants["b"]
        print(f"{cond:28s} clip {o['clip']:.2f}  mu_hat obs {o['muh_obs_med']:.3f} K=1 law {o['muh_K1_med']:.3f} "
              f"K* sim {o['muh_Ks_med'] if Ks else float('nan'):.3f} | J1 {J1:.3f} -> {v['J1']:.3f}+-{v['se']:.3f}, "
              f"mu_bare {o['mub_med']:.2f} -> {v['mub_med']:.2f}, Phi {o['phi_med']:.2f} -> {v['phi_med']:.2f}", flush=True)
    json.dump(out, open(os.path.join(DATA, "checks_larvae.json"), "w"))


if __name__ == "__main__":
    what = sys.argv[1:] or ["larvae", "termites"]
    if "larvae" in what:
        larvae()
    if "termites" in what:
        termites()
