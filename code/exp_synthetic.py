"""Synthetic single-walker Monte Carlo experiments. Writes outputs to ../results/synthetic_*.json.

S1  showcase: recovery of depth K and couplings J from an order-3 causal chain
S2  null calibration (i.i.d. renewal): false-positive rates of AIC/BIC/TIC/CV,
    log statistic (x x') vs linear statistic (ℓ ℓ')
S3  power: detection probability of lag-1 memory, log vs linear statistic
S4  signatures of dependence mechanisms: K*(T) for copula-AR(2), HMM (CCRW), fGn, i.i.d.,
    and MaxEnt(3), plus MaxEnt vs HMM comparison
S5  2D trajectory pipeline: continuous-time Lévy walk sampled at dt with noise → segmentation → K*
S6  quenched heterogeneity: 40 walkers with individual exponents μ_i ~ U(1.2, 2.2), pooled vs fixed effects
"""
import json
import os
import zlib
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from maxent_levy import scan_orders, scan_orders_fe, select, blocked_cv, simulate_model
from simulate import (iid_pareto, copula_ar_pareto, copula_fgn_pareto, ccrw_hmm,
                      levy_walk_2d, segment_flights, merge_collinear)
from baselines import fit_hmm, LinearCoupling

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
L3 = float(np.log(1e3))
CRITS = ["aic", "bic", "tic"]


def _kstar_all(x_list, L, Kmax, cv=True, linear=False):
    fits = scan_orders(x_list, L, Kmax)
    res = {c: select(fits, c)[0] for c in CRITS}
    if cv:
        res["cv"] = int(np.argmin(blocked_cv(x_list, L, Kmax, n_blocks=5)))
    if linear:
        lc = LinearCoupling(L)
        y, U = lc.design(x_list, Kmax)
        lf = [lc.fit(y, U, K) for K in range(Kmax + 1)]
        res["lin_aic"] = int(np.argmin([f["aic"] for f in lf]))
        res["lin_bic"] = int(np.argmin([f["bic"] for f in lf]))
    return res


# ---------------------------------------------------------------- S2 / S3 / S4 worker
def worker(task):
    kind, T, rep, par = task
    rng = np.random.default_rng(zlib.crc32(repr((kind, T, rep, sorted(par.items()))).encode()))
    L = par.get("L", L3)
    Kmax = par.get("Kmax", 6)
    if kind == "null":
        x = iid_pareto(par["mu"], L, T, rng)[0]
        return task, _kstar_all([x], L, Kmax, cv=par.get("cv", True), linear=par.get("linear", False))
    if kind == "power":
        x = copula_ar_pareto([par["rho"]], 2.0, L, T, rng)[0]
        r = _kstar_all([x], L, Kmax, cv=False, linear=True)
        return task, r
    if kind == "sig":
        g = par["gen"]
        if g == "iid":
            x = iid_pareto(2.0, L, T, rng)[0]
        elif g == "copAR2":
            x = copula_ar_pareto([0.5, 0.2], 2.0, L, T, rng)[0]
        elif g == "ccrw":
            x = ccrw_hmm([1.0, 0.01], [[0.9, 0.1], [0.2, 0.8]], L, T, rng)[0][0]
        elif g == "fgn":
            x = copula_fgn_pareto(0.75, 2.0, L, T, rng)[0]
        elif g == "maxent3":
            x = simulate_model(-0.9, [0.25, 0.10, 0.06], L, 2.0, T, rng)[0]
        r = _kstar_all([x], L, Kmax, cv=True)
        if par.get("hmm", False):
            fits = scan_orders([x], L, Kmax)
            kb = select(fits, "bic")[0]
            fits_ln = scan_orders([x], L, Kmax, family="lognormal")
            kb_ln = select(fits_ln, "bic")[0]
            h_pl = fit_hmm([x], L, M=2, fam="pl", Kcond=Kmax, rng=rng)
            h_ex = fit_hmm([x], L, M=2, fam="exp", Kcond=Kmax, rng=rng)
            r.update(aic_maxent=fits[kb].aic, aic_maxent_ln=fits_ln[kb_ln].aic,
                     aic_hmm_pl=h_pl["aic"], aic_hmm_exp=h_ex["aic"],
                     bic_maxent=fits[kb].bic, bic_maxent_ln=fits_ln[kb_ln].bic,
                     bic_hmm_pl=h_pl["bic"], bic_hmm_exp=h_ex["bic"])
        return task, r
    if kind == "pipe":
        # 2D Lévy walk: i.i.d. flights (renewal) or with memory (AR(1) copula ρ)
        n_fl = T
        Lf = float(np.log(1e3))
        if par["rho"] == 0:
            fx = iid_pareto(2.0, Lf, n_fl, rng)[0]
        else:
            fx = copula_ar_pareto([par["rho"]], 2.0, Lf, n_fl, rng)[0]
        pos = levy_walk_2d(fx, v=1.0, dt=par["dt"], noise=par["noise"], rng=rng, lmin=1.0)
        lens, st = segment_flights(pos, eps=par["eps"])
        lens_m, _ = merge_collinear(pos, st, par["theta_c"])
        lmin = par["lmin"]
        r = {}
        for tag, ll in [("raw", lens), ("merged", lens_m)]:
            Ls = float(np.log(ll.max() * 1.0001 / lmin))
            x = np.clip(np.log(ll / lmin), 0, Ls)
            rr = _kstar_all([x], Ls, Kmax, cv=False)
            r[tag + "_bic"] = rr["bic"]; r[tag + "_aic"] = rr["aic"]
            r[tag + "_n"] = len(ll); r[tag + "_clip"] = float(np.mean(ll < lmin))
        rt = _kstar_all([fx], Lf, Kmax, cv=False)
        r.update(true_bic=rt["bic"], true_aic=rt["aic"])
        return task, r
    if kind == "het":
        # 40 tracks with individual exponents μ_i ~ U(1.2, 2.2); shared memory coupling J1
        mus = rng.uniform(1.2, 2.2, par["n_tr"])
        S = [simulate_model(1 - m, [par["J1"]], L, 1.0, T, rng)[0] if par["J1"] else iid_pareto(m, L, T, rng)[0]
             for m in mus]
        fp = scan_orders(S, L, Kmax)
        ff = scan_orders_fe(S, L, Kmax)
        Sh = [rng.permutation(x) for x in S]
        fs = scan_orders(Sh, L, Kmax)
        return task, dict(pooled_bic=select(fp, "bic")[0], fe_bic=select(ff, "bic")[0],
                          pooled_aic=select(fp, "aic")[0], fe_aic=select(ff, "aic")[0],
                          shuffled_pooled_bic=select(fs, "bic")[0],
                          fe_J1=float(ff[1].extra["J"][0]), fe_seJ1=float(ff[1].extra["se_J"][0]),
                          pooled_J1=float(fp[1].J[0]))
    raise ValueError(kind)


def run_pool(tasks, label):
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, (task, r) in enumerate(ex.map(worker, tasks, chunksize=4)):
            out.append(dict(kind=task[0], T=task[1], rep=task[2], par=task[3], **r))
    print(f"[{label}] {len(tasks)} task in {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, f"synthetic_{label}.json"), "w") as f:
        json.dump(out, f, default=float)
    return out


def s1_showcase():
    rng = np.random.default_rng(42)
    h, J, c = -0.9, np.array([0.25, 0.10, 0.06]), 2.0
    T = 5000
    x = simulate_model(h, J, L3, c, T, rng)[0]
    Kmax = 10
    fits = scan_orders([x], L3, Kmax, c=c)
    cv = blocked_cv([x], L3, Kmax, n_blocks=5, c=c)
    fits_iid = fits[0]
    res = dict(
        x=x.tolist(), h_true=h, J_true=J.tolist(), c=c, L=L3, Kmax=Kmax,
        aic=[f.aic for f in fits], bic=[f.bic for f in fits], tic=[f.tic for f in fits], cv=cv.tolist(),
        params={K: fits[K].params.tolist() for K in range(Kmax + 1)},
        se={K: fits[K].se.tolist() for K in range(Kmax + 1)},
        se_sw={K: fits[K].se_sandwich.tolist() for K in range(Kmax + 1)},
        mu_iid=float(1 - fits_iid.h), mu_bare_K3=float(1 - fits[3].h_uncentered),
    )
    with open(os.path.join(OUT, "synthetic_S1.json"), "w") as f:
        json.dump(res, f)
    print("[S1] K* aic/bic/tic/cv:", select(fits, "aic")[0], select(fits, "bic")[0], select(fits, "tic")[0], int(np.argmin(cv)))


if __name__ == "__main__":
    which = sys.argv[1:] or ["S1", "S2", "S3", "S4", "S5"]
    if "S1" in which:
        s1_showcase()
    if "S2" in which:
        tasks = []
        for T in [250, 1000, 4000, 16000]:
            for rep in range(300):
                tasks.append(("null", T, rep, dict(mu=2.0, linear=True, cv=(T <= 4000))))
        # dipendenza da μ e dal range dinamico (code più pesanti, range più ampio)
        for mu in [1.5, 2.5]:
            for rep in range(200):
                tasks.append(("null", 4000, rep, dict(mu=mu, linear=True, cv=False)))
        for rep in range(200):
            tasks.append(("null", 4000, rep, dict(mu=1.5, L=float(np.log(1e5)), linear=True, cv=False)))
        run_pool(tasks, "S2")
    if "S3" in which:
        tasks = []
        for T in [100, 250, 1000, 4000]:
            for rho in [0.05, 0.1, 0.2, 0.4]:
                for rep in range(200):
                    tasks.append(("power", T, rep, dict(rho=rho, Kmax=4)))
        run_pool(tasks, "S3")
    if "S4" in which:
        tasks = []
        for g in ["iid", "copAR2", "ccrw", "fgn", "maxent3"]:
            for T in [250, 1000, 4000, 16000]:
                for rep in range(100 if T < 16000 else 50):
                    tasks.append(("sig", T, rep, dict(gen=g, Kmax=12)))
            for rep in range(30):
                tasks.append(("sig", 4000, 1000 + rep, dict(gen=g, Kmax=8, hmm=True)))
        run_pool(tasks, "S4")
    if "S5" in which:
        tasks = []
        for rho in [0.0, 0.3]:
            for dt, noise, eps in [(0.25, 0.02, 0.1), (1.0, 0.05, 0.2)]:
                for rep in range(60):
                    tasks.append(("pipe", 3000, rep, dict(rho=rho, dt=dt, noise=noise, eps=eps, lmin=1.0, theta_c=0.35, Kmax=6)))
        run_pool(tasks, "S5")
    if "S6" in which:
        tasks = []
        for J1 in [0.0, 0.05, 0.1]:
            for T in [250, 1000]:
                for rep in range(60):
                    tasks.append(("het", T, rep, dict(J1=J1, n_tr=40, Kmax=6)))
        run_pool(tasks, "S6")
