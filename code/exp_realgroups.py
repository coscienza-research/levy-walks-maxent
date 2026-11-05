"""Self + social analysis of two open co-recorded groups (Sec. VI of the paper):

  baboons10  wild baboon troop (Strandburg-Peshkin et al. 2015), 14 days, clock 10 s (primary)
  baboons30  same, clock 30 s (robustness)
  pigeons    homing-pigeon flocks (Mehlhausen-Franks & Portugal 2025), 41 flights, clock 1 s

For each: group-average analysis (depth and delay, mean-field reading); self + social model with unit
fields, with and without time fields, global and local coupling; BIC depths along the axes; single-
neighbour families of increasing rank (decay with distance); neighbours ahead vs behind along the group
heading (leader-follower asymmetry); the 'now' reading (directions).  Null: every individual shifted in
time independently within its group (keeps own memory, destroys synchrony; with positions shifted too,
the 'neighbours' of a shifted individual are whoever passed nearby at other times, so the null retains
fields fixed in space).

    python exp_realgroups.py baboons10 pigeons [--null 20] [--workers 1] [--out ../results] [--quick]
"""
import safe  # noqa: F401
import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import realgroups as rg

warnings.simplefilter("ignore", RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

SPEC = {
    "baboons10": dict(data="baboons", dt=10, min_step=1.0, Ks=3, Kc=6, kc_list=[0, 1, 2, 3, 4, 6], main=(0, 3),
                      ranks=[(0, 1), (1, 2), (2, 3), (5, 6), (11, 12)], fb=(0, 6), tmax=10, variables=("dir", "mod")),
    "baboons30": dict(data="baboons", dt=30, min_step=1.0, Ks=3, Kc=6, kc_list=[0, 1, 2, 3, 4, 6], main=(0, 3),
                      ranks=None, fb=None, tmax=10, variables=("dir", "mod")),
    "pigeons": dict(data="pigeons", dt=1, min_step=3.0, Ks=3, Kc=6, kc_list=[0, 1, 2, 3, 4, 6], main=(0, 1),
                    ranks=[(0, 1), (1, 2), (2, 3)], fb=(0, 3), tmax=6, variables=("dir", "mod")),
}
KF = 3          # lags per family in the rank and ahead/behind fits


def load(spec, quick=False):
    if spec["data"] == "baboons":
        z = np.load(os.path.join(RES, "baboons_1hz.npz"))
        days = z["days"][:2] if quick else z["days"]
        return [rg.Grp(str(int(d)), z[f"P_{int(d)}"].astype(float)[:, ::spec["dt"]]) for d in days]
    z = np.load(os.path.join(RES, "pigeons_1hz.npz"))
    names = z["names"][:8] if quick else z["names"]
    return [rg.Grp(str(k), z[f"P_{k}"].astype(float)) for k in names]


def _J1(f, n_fam):
    J = np.asarray(f["J"]).reshape(n_fam, -1); se = np.asarray(f["se_J"]).reshape(n_fam, -1)
    return dict(J=J.tolist(), se=se.tolist())


def analyze(groups, spec, null=False):
    """everything that is compared between data and null; null=False adds the data-only readings"""
    out = {}
    cut = rg.cutoffs(groups)
    out["cut"] = list(cut)
    Ks, Kc, ms = spec["Ks"], spec["Kc"], spec["min_step"]
    for var in spec["variables"]:
        kw = dict(min_step=ms) if var == "dir" else dict(cut=cut[:2])
        L = cut[2] if var == "mod" else None
        r = {}
        D = rg.rows(groups, var, Ks, Kc, social=[("then", spec["main"])], **kw)
        r["n_rows"] = int(len(D["y"]) if var == "dir" else D.y.size)
        r["local_T"] = rg.fit(D, var, Ks, Kc, L=L)
        # is a social coupling declared?  BIC with time fields, at the selected own depth
        ks_b = {ks: rg.fit(D, var, ks, 0, L=L)["bic"] for ks in range(Ks + 1)}
        ks = min(ks_b, key=ks_b.get)
        kcl = spec["kc_list"] if not null else [0, 1, 2]
        kc_b = {kc: rg.fit(D, var, ks, kc, L=L)["bic"] for kc in kcl}
        r["Ks_sel"], r["Kc_sel"] = int(ks), int(min(kc_b, key=kc_b.get))
        r["ks_bic"] = {str(k): v for k, v in ks_b.items()}; r["kc_bic"] = {str(k): v for k, v in kc_b.items()}
        r["group_average"] = rg.group_average(groups, var, spec["tmax"], min_step=ms, cut=cut[:2])
        if spec["ranks"]:
            fam = [("then", rk) for rk in spec["ranks"]]
            Dr = rg.rows(groups, var, Ks, KF, social=fam, **kw)
            r["ranks"] = _J1(rg.fit(Dr, var, Ks, KF * len(fam), L=L), len(fam))
        if spec["fb"] and var == "dir":
            Df = rg.rows(groups, var, Ks, KF, social=[("ahead", spec["fb"]), ("behind", spec["fb"])], **kw)
            r["ahead_behind"] = _J1(rg.fit(Df, var, Ks, 2 * KF, L=L), 2)
        if not null:
            r["local_noT"] = rg.fit(D, var, Ks, Kc, L=L, time_fe=False)
            Dg = rg.rows(groups, var, Ks, Kc, social=[("then", (0, None))], **kw)
            r["global_noT"] = rg.fit(Dg, var, Ks, Kc, L=L, time_fe=False)
            if var == "dir":
                Dn = rg.rows(groups, var, Ks, Kc, social=[("now", spec["main"])], **kw)
                r["now_T"] = rg.fit(Dn, var, Ks, Kc, L=L)
        out[var] = r
    return out


def _null_job(args):
    name, seed, quick = args
    spec = SPEC[name]
    rng = np.random.default_rng(seed)
    gs = [g.shifted(rng) for g in load(spec, quick)]
    t0 = time.time()
    res = analyze(gs, spec, null=True)
    res["seed"] = seed; res["seconds"] = time.time() - t0
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--null", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=RES)
    ap.add_argument("--quick", action="store_true")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    for name in A.names:
        spec = SPEC[name]
        t0 = time.time()
        res = dict(name=name, spec={k: v for k, v in spec.items()}, data=analyze(load(spec, A.quick), spec))
        print(f"{name}: data done in {time.time() - t0:.0f} s", flush=True)
        jobs = [(name, 1000 + k, A.quick) for k in range(A.null)]
        with ProcessPoolExecutor(max(1, A.workers)) as ex:
            res["null"] = list(ex.map(_null_job, jobs))
        res["seconds"] = time.time() - t0
        fn = os.path.join(A.out, f"realgroups_{name}{'_quick' if A.quick else ''}.json")
        with open(fn, "w") as fh:
            json.dump(res, fh)
        print(f"{name}: {A.null} nulls, total {time.time() - t0:.0f} s -> {fn}", flush=True)
