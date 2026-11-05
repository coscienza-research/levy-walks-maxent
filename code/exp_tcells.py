"""T cells and Dictyostelium (Jerison & Quake 2020, eLife 9:e53933; MIT on GitHub).

Variable: frame displacement ℓ_t = |r(t+Δt) - r(t)| (fixed Δt; tracks split at time gaps).
Purpose: negative control where apparent pooled heavy tails arise from inter-cellular heterogeneity.
Compares pooled model (common field h) against fixed effects (field h_i per cell).
"""
import safe  # noqa: F401  (single-threaded BLAS + low process priority)
import json
import os
import time
import zlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from maxent_levy import scan_orders, scan_orders_fe, select

_DATA_ROOT = os.environ.get("LEVY_DATA_DIR", r"Z:/Datasets/Levy-walks")
F = os.path.join(_DATA_ROOT, "tcells_jerison_quake2020", "master_trajectory_file_all_experiments.txt")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
GROUPS = [("fish_T", "control", 45.0), ("fish_T", "rockout", 45.0), ("Gerard_mouse_T", "control", None),
          ("Gerard_mouse_T", "Myo1g_KO", None), ("Gautreau_dicty", "WT", None)]
KMAX = 6


def load_group(exp, trt, dt_nom):
    df = pd.read_csv(F, sep="\t")
    df = df[(df.experiment == exp) & (df.treatment == trt)]
    seqs = []
    for s in df.iloc[:, 4]:
        a = np.array([[float(v) for v in p.split(",")] for p in s.strip().strip(";").split(";")])
        dt = np.diff(a[:, 0])
        d0 = dt_nom if dt_nom else np.median(dt)
        step = np.hypot(*np.diff(a[:, 1:], axis=0).T)
        ok = np.isclose(dt, d0, rtol=0.05)
        # split at temporal gaps
        cur = []
        for v, o in zip(step, ok):
            if o:
                cur.append(v)
            else:
                if len(cur) > 25:
                    seqs.append(np.array(cur))
                cur = []
        if len(cur) > 25:
            seqs.append(np.array(cur))
    return seqs


def analyze(g):
    exp, trt, dt_nom = g
    rng = np.random.default_rng(zlib.crc32(repr(g).encode()))
    seqs = load_group(exp, trt, dt_nom)
    allv = np.concatenate(seqs)
    lmin = float(np.percentile(allv, 5))     # noise floor: 5th percentile
    lmax = float(allv.max() * 1.001)
    L = float(np.log(lmax / lmin))
    xs = [np.clip(np.log(s / lmin), 0, L) for s in seqs]
    po = scan_orders(xs, L, KMAX)
    fe = scan_orders_fe(xs, L, KMAX)
    po_ln = scan_orders(xs, L, KMAX, family="lognormal")
    fe_ln = scan_orders_fe(xs, L, KMAX, family="lognormal")
    # pooled surrogates: scan up to KMAX; fixed-effects surrogates stop at K = 3
    sh = [select(scan_orders([rng.permutation(x) for x in xs], L, KMAX), "bic")[0] for _ in range(10)]
    sh_fe = [select(scan_orders_fe([rng.permutation(x) for x in xs], L, 3), "bic")[0] for _ in range(10)]
    return dict(group=f"{exp}:{trt}", n_tracks=len(xs), n=int(sum(len(x) for x in xs)), lmin=lmin, lmax=lmax, L=L,
                pooled_K=select(po, "bic")[0], fe_K=select(fe, "bic")[0],
                pooled_J=po[KMAX].J.tolist(), fe_J=fe[KMAX].extra["J"].tolist(), fe_seJ=fe[KMAX].extra["se_J"].tolist(),
                pooled_mu=float(1 - po[0].h), mu_i=(1 - fe[0].extra["h_i"]).tolist(),
                dBIC_pooled_pl_ln=float(po[0].bic - po_ln[0].bic), dBIC_fe_pl_ln=float(fe[0].bic - fe_ln[0].bic),
                shuffled_pooled_K=sh, shuffled_fe_K=sh_fe,
                pooled_bic=[f.bic for f in po], fe_bic=[f.bic for f in fe])


def shuffle_pooled(g):
    """Pooled permutation surrogates only (within-track permutation), scanned up to KMAX."""
    exp, trt, dt_nom = g
    rng = np.random.default_rng(zlib.crc32(repr(g).encode()))
    seqs = load_group(exp, trt, dt_nom)
    allv = np.concatenate(seqs)
    lmin = float(np.percentile(allv, 5))
    lmax = float(allv.max() * 1.001)
    L = float(np.log(lmax / lmin))
    xs = [np.clip(np.log(s / lmin), 0, L) for s in seqs]
    sh = [select(scan_orders([rng.permutation(x) for x in xs], L, KMAX), "bic")[0] for _ in range(10)]
    return dict(group=f"{exp}:{trt}", Kmax=KMAX, shuffled_pooled_K=sh)


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "shuffle":
    t0 = time.time()
    res = [shuffle_pooled(g) for g in GROUPS]          # sequenziale: fit leggeri
    with open(os.path.join(RES, "tcells_shuffle_pooled.json"), "w") as f:
        json.dump(res, f)
    for r in res:
        print(f"{r['group']:28s} shuffled pooled K* (K<={r['Kmax']}): {r['shuffled_pooled_K']}")
    print(f"done in {time.time() - t0:.0f}s")
elif __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(5) as ex:
        res = list(ex.map(analyze, GROUPS))
    with open(os.path.join(RES, "tcells.json"), "w") as f:
        json.dump(res, f, default=float)
    for r in res:
        print(f"{r['group']:28s} n_tr={r['n_tracks']:4d} N={r['n']:6d} L={r['L']:.2f} | pooled K*={r['pooled_K']} "
              f"μ={r['pooled_mu']:.2f} | FE K*={r['fe_K']} J={np.round(r['fe_J'][:3], 3)} | "
              f"shuffled K* pooled {np.bincount(r['shuffled_pooled_K']).tolist()} FE {np.bincount(r['shuffled_fe_K']).tolist()} | "
              f"ΔBIC(pl-ln) pooled {r['dBIC_pooled_pl_ln']:.0f} FE {r['dBIC_fe_pl_ln']:.0f}")
    print(f"done in {time.time() - t0:.0f}s")
