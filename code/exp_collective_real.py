"""Real-data pilot for the self + social collective model (fixed clock).

R1  zebrafish T cells in vivo (Jerison & Quake 2020): each fish = one co-recorded group,
    units = cells, frames of 45 s, variable = displacement per frame.
R2  Drosophila larvae (Sims et al. 2019): each trial = one co-recorded group (~10 larvae in a
    24 cm arena), steps of 1 s (2 frames). Expected negative control (sparse, contact-only).
Null: time-shift surrogate (each unit's track shifted by a random circular offset within its
group): destroys cross-unit synchrony (interaction and common drives), keeps self memory.
"""
import safe  # noqa: F401  (must come first: thread limits + low priority)
import glob
import json
import os
import re
import sys
import time
import zlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

import collective as co
from collective import Group, build_design, pre_pipeline

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
_DATA_ROOT = os.environ.get("LEVY_DATA_DIR", r"Z:/Datasets/Levy-walks")
TC = os.path.join(_DATA_ROOT, "tcells_jerison_quake2020", "master_trajectory_file_all_experiments.txt")
DROS = os.path.join(_DATA_ROOT, "drosophila_larvae_sims2019")


def to_groups(raw, dt, min_units=5):
    """raw: list of (group_name, [ (t_seconds array, xy array) per unit ]).
    Builds Groups on a common integer clock t = round(time/dt); X = log displacement per frame."""
    groups, ells = [], []
    for name, units in raw:
        if len(units) < min_units:
            continue
        tmax = max(int(round(t[-1] / dt)) for t, _ in units) + 1
        U = len(units)
        P = np.full((U, tmax, 2), np.nan)
        for i, (t, xy) in enumerate(units):
            q = t / dt
            keep = np.isclose(q, np.round(q))                  # samples exactly on the clock
            k = np.round(q[keep]).astype(int)
            P[i, k] = xy[keep]
        d = np.diff(P, axis=1)
        ell = np.hypot(d[..., 0], d[..., 1])          # NaN if either frame missing
        groups.append((name, P[:, :-1], ell))
        ells.append(ell[~np.isnan(ell)])
    allv = np.concatenate(ells)
    return groups, allv


def finalize(groups, allv, q_min=5, lmin=None, q_max=99.95):
    """Cutoffs: lmin = noise floor (q_min percentile of positive steps, or given); steps above the
    q_max percentile are tracking glitches (identity swaps) and are treated as missing."""
    if lmin is None:
        lmin = float(np.percentile(allv[allv > 0], q_min))   # zero steps (immobile frames) are clipped to x = 0
    lmax = float(np.percentile(allv, q_max))
    L = float(np.log(lmax / lmin))
    out = []
    for name, P, ell in groups:
        ell = np.where(ell > lmax, np.nan, ell)
        X = np.where(np.isnan(ell), np.nan, np.clip(np.log(np.maximum(ell, 1e-12) / lmin), 0, L))
        out.append(Group(X=X, P=P, box=None, name=name))
    return out, L, lmin, lmax


def load_tcells(treatment="control"):
    df = pd.read_csv(TC, sep="\t")
    df = df[(df.experiment == "fish_T") & (df.treatment == treatment)]
    raw = []
    for fish, sub in df.groupby("sample"):
        units = []
        for s in sub.iloc[:, 4]:
            a = np.array([[float(v) for v in p.split(",")] for p in s.strip().strip(";").split(";")])
            units.append((a[:, 0], a[:, 1:3]))
        raw.append((fish, units))
    return to_groups(raw, 45.0)


def load_larvae():
    files = sorted(glob.glob(os.path.join(DROS, "*.csv")))
    trials = {}
    for f in files:
        trial = re.sub(r"_-_larva\(\d+\)\.csv$", "", os.path.basename(f))
        d = pd.read_csv(f)
        ts = pd.to_datetime(d["Date"], format="%d/%m/%Y %H:%M:%S")
        sec = (ts - pd.Timestamp("2014-01-01")).dt.total_seconds().to_numpy()   # frame index (2 fps)
        trials.setdefault(trial, []).append((sec, d[["X", "Y"]].to_numpy()))
    raw = []
    for trial, units in trials.items():
        if "15Hz" in trial:
            continue
        # frames are labelled 1 per 'second' = 1 frame at 2 fps; use steps of 2 frames (1 s)
        raw.append((trial, [(t, xy) for t, xy in units]))
    return to_groups(raw, 2.0)


def time_shift(groups, rng):
    """Circularly shift each unit's (X, P) by a random offset within its group."""
    out = []
    for g in groups:
        U, T = g.X.shape
        X = np.empty_like(g.X)
        P = np.empty_like(g.P)
        for i in range(U):
            s = rng.integers(T)
            X[i] = np.roll(g.X[i], s)
            P[i] = np.roll(g.P[i], s, axis=0)
        out.append(Group(X=X, P=P, box=g.box, name=g.name))
    return out


def analyze(groups, L, Ks_max=4, Kc_max=6, k_nn=3, n_null=20, rng=None, label="", time_fields=True):
    rng = np.random.default_rng(0) if rng is None else rng
    res = dict(label=label, n_groups=len(groups), n_units=int(sum(g.X.shape[0] for g in groups)), L=L)
    variants = [("global", "global", False), ("local", "local", False)]
    if time_fields:
        variants.append(("local_tfe", "local", True))
    for name, mode, tfe in variants:
        D = build_design(groups, Ks_max, Kc_max, mode=mode, k_nn=k_nn)
        fits = co.scan(D, range(0, Ks_max + 1), range(0, Kc_max + 1), L=L, unit_fe=True, time_fe=tfe)
        kb = co.best(fits, "bic")[0]
        full = fits[(Ks_max, Kc_max)]
        res[name] = dict(N=int(D.y.size), sel=list(kb), a=full.a.tolist(), se_a=full.se_a.tolist(),
                         J=full.J.tolist(), se_J=full.se_J.tolist(),
                         bic={f"{a},{b}": fits[(a, b)].bic for (a, b) in fits})
    out, tsel, delay = pre_pipeline(groups, L, 8)
    res["pre"] = dict(tauM=tsel, delay=delay, J8=out[8]["J"].tolist(), se8=out[8]["se"].tolist(),
                      aic={k: v["aic"] for k, v in out.items()})
    # time-shift null: social couplings and selected Kc under destroyed synchrony
    nulls = {"global": [], "local": [], "pre_tauM": []}
    for _ in range(n_null):
        gs = time_shift(groups, rng)
        for name in ["global", "local"]:
            D = build_design(gs, Ks_max, Kc_max, mode=name, k_nn=k_nn)
            f = co.fit_collective(D, Ks_max, Kc_max, L=L, unit_fe=True)
            fits = co.scan(D, [res[name]["sel"][0]], range(0, Kc_max + 1), L=L, unit_fe=True)
            nulls[name].append(dict(J=f.J.tolist(), Kc=co.best(fits, "bic")[0][1]))
        nulls["pre_tauM"].append(pre_pipeline(gs, L, 8)[1])
    res["null"] = nulls
    return res


def run_one(which):
    if which == "tcells":
        g, allv = load_tcells("control")
    elif which == "tcells_rockout":
        g, allv = load_tcells("rockout")
    else:
        g, allv = load_larvae()
    groups, L, lmin, lmax = finalize(g, allv, lmin=0.05 if which == "larvae" else None)
    rng = np.random.default_rng(zlib.crc32(which.encode()))
    k_nn = 3 if which.startswith("tcells") else 1
    # larvae: ~115k frames -> time fields would mean ~1e5 parameters; not needed for a negative control
    res = analyze(groups, L, k_nn=k_nn, n_null=20, rng=rng, label=which, time_fields=which.startswith("tcells"))
    res.update(lmin=lmin, lmax=lmax,
               units_per_frame=float(np.mean([np.mean((~np.isnan(gg.X)).sum(axis=0)[(~np.isnan(gg.X)).sum(axis=0) > 0])
                                              for gg in groups])))
    with open(os.path.join(RES, f"collective_real_{which}.json"), "w") as f:
        json.dump(res, f, default=float)
    return which, res


if __name__ == "__main__":
    which = sys.argv[1:] or ["tcells", "tcells_rockout", "larvae"]
    t0 = time.time()
    with ProcessPoolExecutor(1) as ex:
        for w, r in ex.map(run_one, which):
            print(f"== {w}: groups={r['n_groups']} units={r['n_units']} L={r['L']:.2f} units/frame={r['units_per_frame']:.1f}")
            for name in [n for n in ["global", "local", "local_tfe"] if n in r]:
                q = r[name]
                print(f"   {name:9s} N={q['N']} sel(Ks,Kc)={q['sel']} a={np.round(q['a'],3)} J={np.round(q['J'],3)} se={np.round(q['se_J'],3)}")
            print(f"   PRE: tau_M={r['pre']['tauM']} delay={r['pre']['delay']} J8={np.round(r['pre']['J8'],3)}")
            for name in ["global", "local"]:
                Jn = np.array([z["J"] for z in r["null"][name]])
                print(f"   null {name}: J mean {np.round(Jn.mean(0),3)} sd {np.round(Jn.std(0),3)}; Kc* {np.bincount([z['Kc'] for z in r['null'][name]], minlength=7).tolist()}")
            print(f"   null PRE tau_M: {np.bincount(r['null']['pre_tauM'], minlength=9).tolist()}")
    print(f"done in {time.time() - t0:.0f}s")
