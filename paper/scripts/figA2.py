"""Fig. A2 -- validation of the inverse problem on synthetic walkers: planted vs inferred
couplings, calibration of the Fisher errors, depth selection, and false positives under the null
(AIC at the Spitzer-Woodroofe level, BIC vanishing)."""
from common import S, np, pd, mpl, dat, res
from scipy.stats import chi2, norm


def make():
    V = dat("validation.json")
    R = V["rows"]; K = V["K"]
    f = S.fig(S.FULL, 54)
    gs = f.add_gridspec(1, 4, width_ratios=[1.05, 1, 1, 1.1], left=0.058, right=0.985, top=0.86, bottom=0.2,
                        wspace=0.5)
    tcols = [S.INDIGO, S.mix(S.INDIGO, 0.62), S.mix(S.INDIGO, 0.36)]
    tmark = ["o", "s", "^"]

    # ------------------------------------------------ (a) planted vs inferred
    ax = f.add_subplot(gs[0, 0])
    ax.plot([-0.25, 0.25], [-0.25, 0.25], color=S.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for k in range(K - 1, -1, -1):
        Jp = np.array([r["J"][k] for r in R]); Ji = np.array([r["J_hat"][k] for r in R])
        ax.scatter(Jp, Ji, s=5, marker=tmark[k], facecolor=tcols[k], edgecolor="none", alpha=0.85, zorder=2 + k,
                   label=rf"$J_{k + 1}$")
    ax.set_xlim(-0.23, 0.23); ax.set_ylim(-0.25, 0.25)
    ax.set_xticks([-0.2, 0, 0.2]); ax.set_yticks([-0.2, 0, 0.2])
    ax.set_xlabel("planted"); ax.set_ylabel("inferred")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper left", markerscale=1.6, handletextpad=0.1, fontsize=6.0,
              borderaxespad=0.1)
    ins = ax.inset_axes([0.64, 0.08, 0.34, 0.34], label="inset")
    hp = np.array([r["hp"] for r in R]); hi = np.array([r["hp_hat"] for r in R])
    ins.plot([-1.6, -0.2], [-1.6, -0.2], color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
    ins.scatter(hp, hi, s=2.2, color=S.TEAL, lw=0)
    ins.set_xlim(-1.6, -0.2); ins.set_ylim(-1.6, -0.2)
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color(S.AXIS); sp.set_linewidth(0.5)
    S.text(ins, 0.06, 0.94, r"$h'$", c=S.TEAL, fontsize=6.0, transform=ins.transAxes, va="top")
    S.despine(ax)

    # ------------------------------------------------ (b) calibration of the Fisher errors
    ax = f.add_subplot(gs[0, 1])
    zJ = np.concatenate([(np.array(r["J_hat"]) - np.array(r["J"])) / np.array(r["se_J"]) for r in R])
    zh = np.array([(r["hp_hat"] - r["hp"]) / r["se_hp"] for r in R])
    z = np.concatenate([zJ, zh])
    bins = np.linspace(-4, 4, 25)
    ax.hist(z, bins=bins, density=True, color=S.mix(S.INDIGO, 0.3), edgecolor="white", lw=0.4)
    zz = np.linspace(-4, 4, 200)
    ax.plot(zz, norm.pdf(zz), color=S.INK, lw=1.0)
    ax.set_xlim(-4, 4); ax.set_ylim(0, 0.47); ax.set_yticks([0, 0.2, 0.4])
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel(r"(inferred $-$ planted) / s.e.")
    ax.set_ylabel("density")
    S.text(ax, 3.9, 0.46, f"mean {z.mean():+.2f}".replace("-", "\N{MINUS SIGN}") + f"\ns.d. {z.std():.2f}", c=S.INK2, fontsize=5.8, ha="right", va="top",
           linespacing=1.1)
    S.text(ax, -3.9, 0.46, "standard\nnormal", c=S.INK, fontsize=5.8, ha="left", va="top", linespacing=1.1)
    S.despine(ax)

    # ------------------------------------------------ (c) depth selection
    ax = f.add_subplot(gs[0, 2])
    kb = np.array([r["K_bic"] for r in R]) - K; ka = np.array([r["K_aic"] for r in R]) - K
    d = np.arange(-3, 6)
    pb = np.array([np.mean(kb == v) for v in d]); pa = np.array([np.mean(ka == v) for v in d])
    ax.bar(d - 0.2, pb, width=0.38, color=S.INDIGO, edgecolor="none")
    ax.bar(d + 0.2, pa, width=0.38, color=S.CORAL, edgecolor="none")
    ax.set_xticks(d)
    ax.set_xlabel(r"$K^*-K_{\rm planted}$")
    ax.set_ylabel("fraction of walkers")
    ax.set_ylim(0, 0.9); ax.set_xlim(-3.6, 5.6)
    S.text(ax, 0.75, 0.52, "too deep:", c=S.INK2, fontsize=5.9, ha="left", va="bottom")
    S.text(ax, 0.75, 0.43, f"BIC {100 * np.mean(kb > 0):.0f}%", c=S.INDIGO, fontsize=5.9, ha="left", va="bottom")
    S.text(ax, 0.75, 0.34, f"AIC {100 * np.mean(ka > 0):.0f}%", c=S.CORAL, fontsize=5.9, ha="left", va="bottom")
    S.despine(ax)

    # ------------------------------------------------ (d) false positives on memoryless walkers
    ax = f.add_subplot(gs[0, 3])
    df = pd.DataFrame(res("synthetic_S2.json"))
    df["mu"] = df.par.apply(lambda p: p["mu"]); df["L"] = df.par.apply(lambda p: p.get("L", np.log(1e3)))
    df["has_cv"] = df.par.apply(lambda p: p["cv"])
    base = df[(df.mu == 2.0) & np.isclose(df.L, np.log(1e3))]
    th = 1 - np.exp(-sum(chi2.sf(2 * k, k) / k for k in range(1, 7)))
    Tg = np.logspace(np.log10(200), np.log10(2e4), 50)
    ax.axhline(th, color=S.CORAL, lw=0.7, ls=(0, (1.2, 1.2)), zorder=1)
    ax.plot(Tg, chi2.sf(np.log(Tg), 1), color=S.INDIGO, lw=0.7, ls=(0, (1.2, 1.2)), zorder=1)
    for col, c, mk, mf, ls in [("cv", S.mix(S.CORAL, 0.55), "D", "white", (0, (3, 1.5))),
                               ("aic", S.CORAL, "o", S.CORAL, "-"), ("bic", S.INDIGO, "s", S.INDIGO, "-")]:
        sub = base[base.has_cv] if col == "cv" else base
        fp = sub.groupby("T")[col].apply(lambda v: np.mean(v > 0))
        n = sub.groupby("T")[col].size()
        err = np.sqrt(fp * (1 - fp) / n)
        ax.errorbar(fp.index, fp.values, yerr=err.values, color=c, marker=mk, ms=3.2, mfc=mf, mew=0.8, lw=1.1, ls=ls,
                    elinewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlim(180, 2.4e4); ax.set_ylim(-0.015, 0.42)
    ax.set_xticks([250, 1000, 4000, 16000]); ax.set_xticklabels(["250", "1000", "4000", "16000"])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(r"flights per walker $T$")
    ax.set_ylabel("memory declared (no memory)")
    S.text(ax, 2.2e4, 0.375, "cross-validation", c=S.mix(S.CORAL, 0.8), fontsize=5.8, ha="right", va="bottom")
    S.text(ax, 2.2e4, 0.215, "AIC", c=S.CORAL, fontsize=6.2, fontweight="bold", ha="right", va="top")
    S.text(ax, 2.2e4, 0.17, f"Spitzer–Woodroofe {100 * th:.1f}%", c=S.CORAL, fontsize=5.6, ha="right", va="top")
    S.text(ax, 2.2e4, 0.035, "BIC", c=S.INDIGO, fontsize=6.2, fontweight="bold", ha="right", va="bottom")
    S.despine(ax)
    axs = f.axes
    main = [a for a in axs if a.get_label() != "inset"][:4]
    S.heads(f, [(main[0], "(a)", "planted vs inferred"), (main[1], "(b)", "Fisher errors are calibrated"),
                (main[2], "(c)", "selected depth"), (main[3], "(d)", "false memory, renewal walkers")])
    S.save(f, "figA2_validation")


if __name__ == "__main__":
    make()
