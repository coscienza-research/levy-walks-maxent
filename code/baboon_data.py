"""Baboon troop GPS (Strandburg-Peshkin et al. 2015, Movebank CC0): load, project to metres,
align on the 1 Hz clock and cache one array per day.  X[u, t] positions in metres (NaN = no fix)."""
import safe  # noqa: F401
import os, sys, numpy as np, pandas as pd

_DATA_ROOT = os.environ.get("LEVY_DATA_DIR", r"Z:/Datasets/Levy-walks")
BAB = os.path.join(_DATA_ROOT, "baboons_strandburg_peshkin2015", "raw")
CSV = os.path.join(BAB, "Collective movement in wild baboons (data from Strandburg-Peshkin et al. 2015).csv")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "baboons_1hz.npz")


def build():
    d = pd.read_csv(CSV, usecols=["timestamp", "location-long", "location-lat", "individual-local-identifier"],
                    dtype={"location-long": "float64", "location-lat": "float64", "individual-local-identifier": "int16"})
    d.columns = ["ts", "lon", "lat", "id"]
    d = d.dropna()
    t = pd.to_datetime(d["ts"]).dt.as_unit("s").astype("int64").to_numpy()    # seconds, UTC
    lat0, lon0 = d.lat.median(), d.lon.median()
    y = (d.lat.to_numpy() - lat0) * 110574.0
    x = (d.lon.to_numpy() - lon0) * 111320.0 * np.cos(np.deg2rad(lat0))
    ids = np.sort(d.id.unique())
    day = (t + 3 * 3600) // 86400                                           # East-Africa time (UTC+3)
    out = {}
    for dd in np.unique(day):
        m = day == dd
        t0 = t[m].min(); T = int(t[m].max() - t0) + 1
        P = np.full((len(ids), T, 2), np.nan, dtype=np.float32)
        ui = np.searchsorted(ids, d.id.to_numpy()[m])
        P[ui, (t[m] - t0).astype(int), 0] = x[m]; P[ui, (t[m] - t0).astype(int), 1] = y[m]
        out[f"P_{int(dd)}"] = P; out[f"t0_{int(dd)}"] = t0
    np.savez_compressed(CACHE, ids=ids, days=np.unique(day), **out)
    return ids


if __name__ == "__main__":
    ids = build()
    z = np.load(CACHE)
    print("ids", ids.tolist())
    for dd in z["days"]:
        P = z[f"P_{int(dd)}"]; ok = ~np.isnan(P[..., 0])
        print(int(dd), P.shape, "fixes", int(ok.sum()), "units>=1:", int((ok.sum(1) > 0).sum()),
              "mean units/frame", round(float(ok.sum(0).mean()), 1))
