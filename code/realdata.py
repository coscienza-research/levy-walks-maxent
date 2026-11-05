"""Single-walker analysis pipeline: flight sequences → causal MaxEnt order scan.

Functions:
  analyze(seqs_ell, lmin, lmax, Kmax)  — order scan (power-law and lognormal families),
      AIC/BIC/TIC/CV, permutation surrogates, 2- and 3-state HMM comparison,
      and condensation control parameter Φ = J·L/(μ_bare - 1).
"""
import numpy as np
from maxent_levy import LogSpace, scan_orders, select, blocked_cv
from baselines import fit_hmm


def to_x(seqs_ell, lmin, lmax):
    ls = LogSpace(lmin, lmax)
    xs, fr = [], []
    for s in seqs_ell:
        x, f = ls.to_x(np.asarray(s, dtype=float))
        xs.append(x)
        fr.append(f)
    return xs, ls.L, float(np.average(fr, weights=[len(s) for s in seqs_ell]))


def summarize_fits(fits):
    out = {}
    for crit in ["aic", "bic", "tic"]:
        k, vals = select(fits, crit)
        out[crit] = dict(K=k, delta=(vals - vals.min()).tolist())
    return out


def analyze(seqs_ell, lmin, lmax, Kmax=10, n_surr=20, hmm=True, cv=True, rng=None, label=""):
    rng = np.random.default_rng(0) if rng is None else rng
    xs, L, frac_clip = to_x(seqs_ell, lmin, lmax)
    xs = [x for x in xs if len(x) > Kmax + 5]
    N = sum(len(x) - Kmax for x in xs)
    res = dict(label=label, lmin=lmin, lmax=lmax, L=L, n_seq=len(xs), N=N, frac_clip=frac_clip)

    fits = scan_orders(xs, L, Kmax)
    res["pl"] = summarize_fits(fits)
    Kb = res["pl"]["bic"]["K"]
    res["pl"]["params_bic"] = fits[Kb].params.tolist()
    res["pl"]["se_bic"] = fits[Kb].se_sandwich.tolist()
    res["pl"]["params_Kmax"] = fits[Kmax].params.tolist()
    res["pl"]["se_Kmax"] = fits[Kmax].se_sandwich.tolist()
    res["pl"]["mu_iid"] = float(1 - fits[0].h)
    res["pl"]["c"] = fits[0].c
    # phase control parameter: sum of couplings × range / (μ_bare - 1)
    f = fits[Kb]
    mu_bare = 1 - f.h_uncentered
    res["pl"]["mu_bare"] = float(mu_bare)
    res["pl"]["phase"] = float(np.sum(f.J) * L / max(mu_bare - 1, 1e-9)) if Kb > 0 else 0.0
    res["pl"]["aic_by_K"] = [g.aic for g in fits]
    res["pl"]["bic_by_K"] = [g.bic for g in fits]

    fits_ln = scan_orders(xs, L, Kmax, family="lognormal")
    res["ln"] = summarize_fits(fits_ln)
    Kb_ln = res["ln"]["bic"]["K"]
    res["ln"]["params_bic"] = fits_ln[Kb_ln].params.tolist()
    res["ln"]["aic_by_K"] = [g.aic for g in fits_ln]
    res["ln"]["bic_by_K"] = [g.bic for g in fits_ln]
    res["dAIC_pl_minus_ln_K0"] = fits[0].aic - fits_ln[0].aic
    res["best_maxent"] = dict(
        pl=dict(K=Kb, aic=fits[Kb].aic, bic=fits[Kb].bic),
        ln=dict(K=Kb_ln, aic=fits_ln[Kb_ln].aic, bic=fits_ln[Kb_ln].bic))

    if cv:
        cvv = blocked_cv(xs, L, Kmax, n_blocks=5)
        res["cv"] = dict(K=int(np.argmin(cvv)), delta=(cvv - cvv.min()).tolist())

    # within-sequence permutation surrogates (destroys memory, preserves marginal -> K* = 0)
    ks = []
    for _ in range(n_surr):
        xs_s = [rng.permutation(x) for x in xs]
        ks.append(select(scan_orders(xs_s, L, Kmax), "bic")[0])
    res["surrogate_bic_K"] = ks

    if hmm:
        res["hmm"] = {}
        for M in [2, 3]:
            for fam in ["pl", "exp"]:
                try:
                    r = fit_hmm(xs, L, M=M, fam=fam, Kcond=Kmax, n_starts=6 if M == 2 else 8, rng=rng)
                    res["hmm"][f"{fam}{M}"] = dict(aic=r["aic"], bic=r["bic"], nll=r["nll"], k=r["k"],
                                                   em=r["em"].tolist(), A=r["A"].tolist())
                except Exception as e:  # pragma: no cover
                    res["hmm"][f"{fam}{M}"] = dict(error=str(e))
    return res


def report(res):
    pl, ln = res["pl"], res["ln"]
    s = [f"== {res['label']}: {res['n_seq']} sequences, N={res['N']}, L={res['L']:.2f} "
         f"(ℓmax/ℓmin={np.exp(res['L']):.0f}), clip={res['frac_clip']:.3f}",
         f"   power-law: K* AIC/BIC/TIC = {pl['aic']['K']}/{pl['bic']['K']}/{pl['tic']['K']}"
         + (f", CV = {res['cv']['K']}" if 'cv' in res else "")
         + f";  μ_iid={pl['mu_iid']:.3f}, μ_bare={pl['mu_bare']:.3f}, phase J·L/(μ-1)={pl['phase']:.2f}",
         f"   J (K*={pl['bic']['K']}): {np.round(pl['params_bic'][1:], 3).tolist()}",
         f"   lognormal: K* AIC/BIC/TIC = {ln['aic']['K']}/{ln['bic']['K']}/{ln['tic']['K']};"
         f"  ΔAIC(pl-ln, K=0) = {res['dAIC_pl_minus_ln_K0']:.1f}",
         f"   surrogates (permuted) K*_BIC: {np.bincount(res['surrogate_bic_K']).tolist()}"]
    if "hmm" in res:
        best = res["best_maxent"]
        ref = min(best["pl"]["aic"], best["ln"]["aic"])
        refb = min(best["pl"]["bic"], best["ln"]["bic"])
        hs = ", ".join(f"{k}: ΔAIC={v['aic'] - ref:+.1f} ΔBIC={v['bic'] - refb:+.1f}"
                       for k, v in res["hmm"].items() if "aic" in v)
        s.append(f"   HMM vs best MaxEnt (negative = HMM better): {hs}")
    return "\n".join(s)
