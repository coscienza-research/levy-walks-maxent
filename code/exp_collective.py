"""Monte Carlo experiments for the collective (self + social) package.

C1  recovery: self memory (2 lags) + local social coupling at delay 3
C2  aggregation trap: independent units with self memory (depth 3), no interaction;
    PRE pipeline vs exact P_A MLE vs self+social model; N = 10, 30, 100
C3  common-drive trap: shared OU field, no interaction (and a positive control with interaction);
    global vs local vs local + time fixed effects
C4  PRE model true (no self memory, global coupling): PRE surrogate vs exact MLE, N = 5..100
C5  delay recovery for Delta = 1, 2, 4, 6 (self memory present)
C6  mean-field checks: susceptibility vs J (sim vs theory), self-memory shift of the critical point
Results: ../results/collective_*.json
"""
import safe  # noqa: F401  (must come first: thread limits + low priority)
import json
import os
import sys
import time
import zlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import collective as co
from collective import Group, build_design, fit_collective, pre_pipeline, Lambda_inv
from simulate_collective import speed_vicsek, group_from_positions, pre_model
from meanfield import simulate_mf, dLam, critical_point, fixed_points

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
L0 = float(np.log(100.0))
BASE = dict(box=30.0, k_nn=3, k_align=6, eta=0.3, ell_min=0.05)


def seed(*a):
    return np.random.default_rng(zlib.crc32(repr(a).encode()))


def hetero_h(N, rng):
    return rng.uniform(-1.3, -0.5, N)


def scan_best(D, Ks_list, Kc_list, L, crit="bic", **kw):
    fits = co.scan(D, Ks_list, Kc_list, L=L, **kw)
    k, f = co.best(fits, crit)
    return k, f, fits


def yw_prediction(groups, L, tauM):
    """Theory for the PRE coefficients under pure self memory (Prop. C2):
    J_pred = phi / Lambda'(theta_bar), phi = Yule-Walker AR(tauM) coefficients of the
    average within-unit autocovariance."""
    acs = []
    for g in groups:
        for x in g.X:
            x = x[~np.isnan(x)]
            if len(x) < 50:
                continue
            x = x - x.mean()
            acs.append([np.mean(x[k:] * x[:len(x) - k]) for k in range(tauM + 1)])
    C = np.mean(acs, axis=0)
    R = np.array([[C[abs(i - j)] for j in range(tauM)] for i in range(tauM)])
    phi = np.linalg.solve(R, C[1:])
    mbar = np.nanmean(np.concatenate([g.X.ravel() for g in groups]))
    th = Lambda_inv(np.array([mbar]), L)[0]
    return phi / dLam(np.array([th]), L)[0]


def worker(task):
    kind, rep, par = task
    rng = seed(kind, rep, sorted(par.items()))
    if kind == "C1":
        N, T = par["N"], par["T"]
        sim = speed_vicsek(N, T, rng, L=L0, h=hetero_h(N, rng), a=par["a"], Jloc=par["Jloc"], **BASE)
        g, L = group_from_positions(sim)
        r = {}
        for mode in ["local", "global"]:
            D = build_design([g], 4, 6, mode=mode, k_nn=3)
            k, f, fits = scan_best(D, range(0, 5), range(0, 7), L)
            full = fits[(2, 6)]
            r[mode] = dict(sel=list(k), a=full.a.tolist(), se_a=full.se_a.tolist(), J=full.J.tolist(),
                           se_J=full.se_J.tolist(),
                           bic=({f"{a},{b}": fits[(a, b)].bic for (a, b) in fits} if rep == 0 else None))
        out, tsel, delay = pre_pipeline([g], L, 8)
        r["pre"] = dict(tauM=tsel, delay=delay, J8=out[8]["J"].tolist(), Jsel=out[tsel]["J"].tolist())
        return task, r
    if kind == "C2":
        N, T = par["N"], par["T"]
        sim = speed_vicsek(N, T, rng, L=L0, h=hetero_h(N, rng), a=par["a"], **BASE)
        g, L = group_from_positions(sim)
        out, tsel, delay = pre_pipeline([g], L, 8)
        pre = dict(tauM=tsel, delay=delay, J8=out[8]["J"].tolist(), se8=out[8]["se"].tolist(),
                   pred8=yw_prediction([g], L, 8).tolist())
        # exact MLE of the PRE measure P_A: common field, full mean incl. self, no self terms
        Dq = build_design([g], 0, 8, mode="global+")
        fq = co.scan(Dq, [0], range(0, 9), L=L, unit_fe=False)
        kq = co.best(fq, "bic")[0]
        ka = co.best(fq, "aic")[0]
        q = dict(Kc_bic=kq[1], Kc_aic=ka[1], J8=fq[(0, 8)].J.tolist(), se8=fq[(0, 8)].se_J.tolist())
        # ours: unit fields + self lags + leave-one-out global social
        Ds = build_design([g], 4, 6, mode="global")
        ks, fs, fits = scan_best(Ds, range(0, 5), range(0, 7), L, unit_fe=True)
        s = dict(sel=list(ks), J6=fits[(4, 6)].J.tolist(), se6=fits[(4, 6)].se_J.tolist(),
                 a4=fits[(4, 6)].a.tolist(), Kc_aic=co.best(fits, "aic")[0][1])
        return task, dict(pre=pre, pa=q, ours=s)
    if kind == "C3":
        N, T = par["N"], par["T"]
        sim = speed_vicsek(N, T, rng, L=L0, h=hetero_h(N, rng), a=par["a"], Jloc=par["Jloc"],
                           drive_sd=par["drive_sd"], drive_tau=par["drive_tau"], **BASE)
        g, L = group_from_positions(sim)
        r = {}
        out, tsel, delay = pre_pipeline([g], L, 6)
        r["pre"] = dict(tauM=tsel, delay=delay, J6=out[6]["J"].tolist())
        for name, mode, tfe in [("global", "global", False), ("local", "local", False), ("local_tfe", "local", True)]:
            D = build_design([g], 2, 4, mode=mode, k_nn=3)
            k, f, fits = scan_best(D, range(0, 3), range(0, 5), L, unit_fe=True, time_fe=tfe)
            full = fits[(1, 4)]
            r[name] = dict(sel=list(k), J4=full.J.tolist(), se4=full.se_J.tolist())
        return task, r
    if kind == "C4":
        N, T = par["N"], par["T"]
        X = pre_model(N, T, par["H"], par["J"], L0, rng)
        g = Group(X=X)
        out, tsel, delay = pre_pipeline([g], L0, 6)
        D = build_design([g], 0, 6, mode="global+")
        fits = co.scan(D, [0], range(0, 7), L=L0, unit_fe=False)
        kb = co.best(fits, "bic")[0][1]
        Jb = fits[(0, kb)].J
        return task, dict(pre=dict(tauM=tsel, delay=delay, J6=out[6]["J"].tolist()),
                          mle=dict(Kc=kb, delay=int(np.argmax(np.abs(Jb)) + 1) if kb else 0,
                                   J6=fits[(0, 6)].J.tolist(), se6=fits[(0, 6)].se_J.tolist()))
    if kind == "C5":
        N, T, Dl = par["N"], par["T"], par["Delta"]
        Jloc = np.zeros(Dl); Jloc[-1] = par["J"]
        sim = speed_vicsek(N, T, rng, L=L0, h=hetero_h(N, rng), a=par["a"], Jloc=Jloc, **BASE)
        g, L = group_from_positions(sim)
        D = build_design([g], 3, 8, mode="local", k_nn=3)
        k, f, fits = scan_best(D, range(0, 4), range(0, 9), L, unit_fe=True)
        Jsel = f.J
        dl = int(np.argmax(np.abs(Jsel)) + 1) if k[1] else 0
        out, tsel, delay = pre_pipeline([g], L, 8)
        return task, dict(sel=list(k), delay=dl, J8=fits[(2, 8)].J.tolist(), pre=dict(tauM=tsel, delay=delay))
    if kind == "C7":
        from collective_dir import sim_dir, scan_dir, pre_pipeline_dir
        N, T = par["N"], par["T"]
        S, P = sim_dir(N, T, rng, H=par["H"], a=par.get("a", []), Jloc=par["Jloc"], turn_corr=par["turn_corr"], turn_sd=par["turn_sd"],
                       box=20.0)
        out, ts, dl = pre_pipeline_dir(S, 10)
        mode = par["mode"]
        sc = scan_dir(S, P, 4, 5, mode, box=20.0)
        kb = min(sc, key=lambda k: sc[k]["bic"])
        full = sc[(4, 5)]
        return task, dict(pre=dict(tauM=ts, delay=dl, J10=out[10]["J"].tolist()),
                          ours=dict(sel=list(kb), J5=full["J"].tolist(), se5=full["se_J"].tolist(),
                                    a4=full["a"].tolist()))
    if kind == "C6":
        Jc, Hc, mc = critical_point(L0)
        N, T, a = par["N"], par["T"], par["a"]
        J = par["frac"] * Jc
        H = -(J + a) * L0 / 2                     # symmetric line (theta* = 0 when x ~ L/2)
        m = simulate_mf(N, T, H, [J], L0, rng, a=[a] if a else ())
        return task, dict(NVar=float(N * m.var()), mean=float(m.mean()), J=J, a=a,
                          ac1=float(np.corrcoef(m[1:], m[:-1])[0, 1]))
    raise ValueError(kind)


def run(tasks, label, workers=None):
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(workers or safe.MAX_WORKERS) as ex:
        for task, r in ex.map(worker, tasks, chunksize=1):
            res.append(dict(kind=task[0], rep=task[1], par=task[2], **r))
    with open(os.path.join(OUT, f"collective_{label}.json"), "w") as f:
        json.dump(res, f, default=float)
    print(f"[{label}] {len(tasks)} tasks in {time.time() - t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    which = sys.argv[1:] or ["C1", "C2", "C3", "C4", "C5", "C6"]
    if "C1" in which:
        run([("C1", r, dict(N=50, T=1500, a=[0.25, 0.10], Jloc=[0, 0, 0.35])) for r in range(40)], "C1")
    if "C2" in which:
        run([("C2", r, dict(N=N, T=1500, a=[0.30, 0.15, 0.10])) for N in [10, 30, 100] for r in range(40)], "C2")
    if "C3" in which:
        tasks = []
        for Jl in [[], [0, 0.30]]:
            for r in range(40):
                tasks.append(("C3", r, dict(N=50, T=1500, a=[0.25], Jloc=Jl, drive_sd=0.4, drive_tau=20.0)))
        run(tasks, "C3")
    if "C4" in which:
        run([("C4", r, dict(N=N, T=1000, H=-1.2, J=[0.25, 0, 0, 0.35])) for N in [5, 10, 30, 100] for r in range(60)],
            "C4")
    if "C5" in which:
        run([("C5", r, dict(N=50, T=1500, a=[0.25, 0.10], Delta=Dl, J=0.30)) for Dl in [1, 2, 4, 6] for r in range(30)],
            "C5")
    if "C7" in which:
        tasks = []
        for cond, Jl, aa, mode in [("none", [], [], "global"), ("local3", [0, 0, 1.5], [1.5, 0.5], "local")]:
            for r in range(30):
                tasks.append(("C7", r, dict(N=50, T=1200, H=[0.3, 0.0], Jloc=Jl, a=aa, turn_corr=0.9, turn_sd=0.25,
                                            mode=mode, cond=cond)))
        run(tasks, "C7")
    if "C6" in which:
        tasks = []
        for a in [0.0, 0.2]:
            for frac in np.linspace(-0.9, 0.9, 13):
                for r in range(3):
                    tasks.append(("C6", r, dict(N=400, T=20000, a=a, frac=float(frac))))
        run(tasks, "C6")
