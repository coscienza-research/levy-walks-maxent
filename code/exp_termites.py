"""Termite workers Cornitermes cumulans (Castiblanco et al., Zenodo 3628061, CC-BY 4.0).

75 focal workers tracked in mm at 2 Hz for 6 h in circular Petri dishes (~49 mm) across varying
nestmate densities and soldier:worker ratios. Long sequences (~3700 flights median) enable
per-track inference, local-field drift controls, and wall-proximity (thigmotaxis) controls.
"""
import glob
import json
import os
import re
import time
import zlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from maxent_levy import scan_orders, scan_orders_fe, scan_orders_cov, select
from exp_drosophila import track_flights, chunk_shuffle, to_x_list
from baselines import fit_hmm

_DATA_ROOT = os.environ.get("LEVY_DATA_DIR", r"Z:/Datasets/Levy-walks")
D = os.path.join(_DATA_ROOT, "termites_castiblanco_zenodo3628061")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
EPS, LMIN, KMAX = 0.5, 1.0, 10


def parse(name):
    m = re.match(r"S(\d+)_d(\d+)_(\d+)s_(\d+)w", name)
    return dict(id=int(m.group(1)), density=int(m.group(2)) / 100, soldiers=int(m.group(3)), workers=int(m.group(4)))


def one_track(f):
    rng = np.random.default_rng(zlib.crc32(f.encode()))
    xy = pd.read_csv(f)[["X", "Y"]].to_numpy()
    lm, sm = track_flights(xy, EPS)
    lmax = lm.max() * 1.001
    xs, L, clip = to_x_list([lm], LMIN, lmax)
    x = xs[0]
    fits = scan_orders([x], L, KMAX)
    kb, ka = select(fits, "bic")[0], select(fits, "aic")[0]
    fits_ln = scan_orders([x], L, KMAX, family="lognormal")
    kb_ln = select(fits_ln, "bic")[0]
    perm = [select(scan_orders([rng.permutation(x)], L, 4), "bic")[0] for _ in range(10)]
    # pipeline null (1 replicate): shuffled flight chunks with real sub-sampled microstructure
    xy2 = chunk_shuffle(xy, sm, rng)
    lp, _ = track_flights(xy2, EPS)
    xp, Lp, _ = to_x_list([lp], LMIN, max(lmax, lp.max() * 1.001))
    pipe = select(scan_orders(xp, Lp, 4), "bic")[0]
    hm2 = fit_hmm([x], L, M=2, fam="pl", Kcond=KMAX, n_starts=4, rng=rng)
    hm3 = fit_hmm([x], L, M=3, fam="pl", Kcond=KMAX, n_starts=5, rng=rng)
    out = dict(file=os.path.basename(f), **parse(os.path.basename(f)), n_frames=len(xy), n_flights=len(x),
               lmax=float(lmax), L=L, clip=clip, K_bic=kb, K_aic=ka, K_bic_ln=kb_ln,
               J=fits[KMAX].J.tolist(), seJ=fits[KMAX].se_sandwich[1:].tolist(),
               J_kstar=fits[kb].J.tolist(), mu_iid=float(1 - fits[0].h),
               bic=[g.bic for g in fits], aic=[g.aic for g in fits],
               bic_ln=[g.bic for g in fits_ln],
               dBIC_pl_ln=float(fits[kb].bic - fits_ln[kb_ln].bic),
               perm_K=perm, pipe_K=pipe,
               bic_hmm2=hm2["bic"], bic_hmm3=hm3["bic"], aic_hmm2=hm2["aic"], aic_hmm3=hm3["aic"],
               bic_me=float(min(fits[kb].bic, fits_ln[kb_ln].bic)), aic_me=float(min(fits[ka].aic, fits_ln[select(fits_ln, 'aic')[0]].aic)),
               x=x.tolist())
    return out


def block_track(args):
    """Local fields per block of B flights (B=None: single field) + within-block permutation null."""
    rec, B, fam = args
    rng = np.random.default_rng(zlib.crc32(repr((rec["file"], B, fam)).encode()))
    x = np.asarray(rec["x"]); L = rec["L"]
    fits = scan_orders_fe([x], L, 4, family=fam, block=B)
    J1 = float(fits[1].extra["J"][0])
    nullJ = []
    for _ in range(10):
        xs = x.copy()
        if B:
            for s in range(4, len(xs), B):
                xs[s:s + B] = rng.permutation(xs[s:s + B])
        else:
            xs = rng.permutation(xs)
        nullJ.append(float(scan_orders_fe([xs], L, 4, family=fam, block=B)[1].extra["J"][0]))
    return dict(file=rec["file"], B=B if B else 0, fam=fam, K_bic=select(fits, "bic")[0], J1=J1,
                null_mean=float(np.mean(nullJ)), null_sd=float(np.std(nullJ)),
                z=float((J1 - np.mean(nullJ)) / (np.std(nullJ) + 1e-12)))


def wall_track(arg):
    f, d_wall = arg if isinstance(arg, tuple) else (arg, 3.0)
    """Memory with and without external wall field (thigmotaxis in circular arena)."""
    xy = pd.read_csv(f)[["X", "Y"]].to_numpy()
    lm, sm = track_flights(xy, EPS)
    ctr = (xy.max(axis=0) + xy.min(axis=0)) / 2
    r = np.hypot(*(xy - ctr).T)
    R = np.percentile(r, 99.5)
    ends = list(sm[1:]) + [len(xy) - 1]
    wall = np.array([np.mean(r[s:e + 1] > R - d_wall) > 0.5 for s, e in zip(sm, ends)], dtype=float)
    lmax = lm.max() * 1.001
    xs, L, clip = to_x_list([lm], LMIN, lmax)
    out = dict(file=os.path.basename(f), d_wall=d_wall, **parse(os.path.basename(f)), R=float(R), frac_wall_flights=float(wall.mean()),
               frac_time_wall=float(np.mean(r > R - d_wall)))
    for fam in ["powerlaw", "lognormal"]:
        f0 = scan_orders(xs, L, 6, family=fam)
        f1 = scan_orders_cov(xs, [wall], L, 6, family=fam)
        out[fam] = dict(K_nowall=select(f0, "bic")[0], K_wall=select(f1, "bic")[0],
                        J1_nowall=float(f0[1].J[0]), J1_wall=float(f1[1].extra["J"][0]),
                        seJ1_wall=float(f1[1].extra["se_J"][0]), g_wall=float(f1[0].extra["g"][0]),
                        dBIC_wall=float(f1[select(f1, "bic")[0]].bic - f0[select(f0, "bic")[0]].bic),
                        J_wall_K6=f1[6].extra["J"].tolist(), J_nowall_K6=f0[6].J.tolist())
    # memory within interior segments only (consecutive runs far from the wall, ≥ 30 flights)
    runs, cur = [], []
    for xv, wv in zip(xs[0], wall):
        if wv == 0:
            cur.append(xv)
        else:
            if len(cur) >= 30:
                runs.append(np.array(cur))
            cur = []
    if len(cur) >= 30:
        runs.append(np.array(cur))
    if runs and sum(len(u) for u in runs) > 200:
        fi = scan_orders_fe(runs, L, 3)
        out["interior"] = dict(n_runs=len(runs), n=int(sum(len(u) for u in runs)), K=select(fi, "bic")[0],
                               J1=float(fi[1].extra["J"][0]), seJ1=float(fi[1].extra["se_J"][0]))
    return out


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "wall":
    t0 = time.time()
    fs = sorted(glob.glob(D + "/S*.csv"))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(wall_track, [(f, d) for d in [3.0, 6.0, 9.0] for f in fs]))
    with open(os.path.join(RES, "termites_wall.json"), "w") as f:
        json.dump(res, f, default=float)
    rows = []
    for r in res:
        d = dict(id=r["file"][:3], d_wall=r["d_wall"], wall_time=r["frac_time_wall"], wall_fl=r["frac_wall_flights"])
        for fam in ["powerlaw", "lognormal"]:
            d.update({f"{fam[:2]}_{k}": v for k, v in r[fam].items() if not isinstance(v, list)})
        if "interior" in r:
            d.update({f"int_{k}": v for k, v in r["interior"].items()})
        rows.append(d)
    df = pd.DataFrame(rows)
    print(df.groupby("d_wall").median(numeric_only=True).round(3).T.to_string())
    print(f"wall done in {time.time() - t0:.0f}s")
elif __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "block":
    t0 = time.time()
    recs = json.load(open(os.path.join(RES, "termites.json")))
    tasks = [(r, B, "lognormal") for r in recs for B in [None, 50, 25]]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(block_track, tasks))
    with open(os.path.join(RES, "termites_block.json"), "w") as f:
        json.dump(res, f, default=float)
    df = pd.DataFrame(res)
    print(df.groupby("B")[["K_bic", "J1", "z"]].describe().round(2).to_string())
    print("frac z>3 by B:", df.groupby("B").z.apply(lambda v: np.mean(v > 3)).to_dict())
    print(f"block done in {time.time() - t0:.0f}s")
elif __name__ == "__main__":
    t0 = time.time()
    fs = sorted(glob.glob(D + "/S*.csv"))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(one_track, fs))
    with open(os.path.join(RES, "termites.json"), "w") as f:
        json.dump(res, f, default=float)
    df = pd.DataFrame(res)
    print(f"done in {time.time() - t0:.0f}s")
    print(df[["id", "density", "soldiers", "workers", "n_flights", "clip", "K_bic", "K_aic", "mu_iid", "pipe_K"]].to_string())
    print("perm K>0 frac:", np.mean([k > 0 for p in df.perm_K for k in p]))
