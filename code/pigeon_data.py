"""Homing-pigeon flocks (Mehlhausen-Franks & Portugal 2025, Dryad doi:10.5061/dryad.4tmpg4fp6, CC0):
three lofts (6, 10 and 8 tracked birds) released repeatedly (41 flights).  Each flight is one co-recorded
group on the common 1 s GPS clock (UTC seconds); for the bird-flights logged at 5 Hz the first sample of
each second is kept.  Positions are projected to metres and cached as P_<loft>_<iteration> (U, T, 2)."""
import os
import numpy as np
import pandas as pd

_DATA_ROOT = os.environ.get("LEVY_DATA_DIR", r"Z:/Datasets/Levy-walks")
CSV = os.path.join(_DATA_ROOT, "pigeons_mehlhausen", "raw", "Pigeon.Flights.Habitat.csv")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "pigeons_1hz.npz")


def build():
    d = pd.read_csv(CSV)
    d = d.rename(columns={d.columns[0]: "idx"})
    lat0, lon0 = d.Latitude.median(), d.Longitude.median()
    d["x"] = (d.Longitude - lon0) * 111320.0 * np.cos(np.deg2rad(lat0))
    d["y"] = (d.Latitude - lat0) * 110574.0
    d = d.sort_values(["Loft", "Iteration", "Bird", "Time", "Counter"]).drop_duplicates(["Loft", "Iteration", "Bird", "Time"])
    out, names = {}, []
    for (loft, it), f in d.groupby(["Loft", "Iteration"]):
        birds = np.sort(f.Bird.unique())
        t0, t1 = int(f.Time.min()), int(f.Time.max())
        P = np.full((len(birds), t1 - t0 + 1, 2), np.nan)
        P[np.searchsorted(birds, f.Bird.to_numpy()), f.Time.to_numpy() - t0] = f[["x", "y"]].to_numpy()
        key = f"{loft}_{int(it):02d}"
        out[f"P_{key}"] = P; out[f"birds_{key}"] = birds; names.append(key)
    np.savez_compressed(CACHE, names=np.array(names), **out)
    return names


if __name__ == "__main__":
    names = build()
    z = np.load(CACHE)
    for k in names:
        P = z[f"P_{k}"]; ok = ~np.isnan(P[..., 0])
        print(k, P.shape, "birds/frame %.1f" % ok.sum(0).mean(), "coverage %.2f" % ok.mean())
