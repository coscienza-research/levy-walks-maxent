"""Fig. A3 -- how memory is missed or invented, on synthetic walkers: the linear statistic is blind,
quenched heterogeneity and segmentation invent memory, and different mechanisms leave different
signatures in the selected depth K*(T)."""
from common import S, np, pd, mpl, res


def make():
    f = S.fig(S.FULL, 54)
    gs = f.add_gridspec(1, 4, width_ratios=[1.12, 1.02, 0.97, 1.12], left=0.068, right=0.885, top=0.86, bottom=0.2,
                        wspace=0.52)
    # ------------------------------------------------ (a) the right variable
    ax = f.add_subplot(gs[0, 0])
    df = pd.DataFrame(res("synthetic_S3.json"))
    df["rho"] = df.par.apply(lambda p: p["rho"])
    for T, c, ls, lab in [(4000, S.INDIGO, "-", None), (1000, S.mix(S.INDIGO, 0.5), "-", None)]:
        s = df[df["T"] == T]
        p = s.groupby("rho")["bic"].apply(lambda v: np.mean(v > 0))
        ax.plot(p.index, p.values, color=c, marker="o", ms=3.0, lw=1.2, ls=ls)
    s = df[df["T"] == 4000]
    p = s.groupby("rho")["lin_bic"].apply(lambda v: np.mean(v > 0))
    ax.plot(p.index, p.values, color=S.CORAL, marker="o", ms=3.0, mfc="white", mew=0.8, lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.4]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.4"])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(-0.04, 1.06); ax.set_xlim(0.042, 0.47)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(r"planted memory $\rho$")
    ax.set_ylabel("memory detected (BIC)")
    S.text(ax, 0.46, 0.66, r"log, $T$=4000", c=S.INDIGO, fontsize=5.8, ha="right", va="center")
    S.text(ax, 0.46, 0.555, r"log, $T$=1000", c=S.mix(S.INDIGO, 0.7), fontsize=5.8, ha="right", va="center")
    S.text(ax, 0.46, 0.45, r"linear, $T$=4000", c=S.CORAL, fontsize=5.8, ha="right", va="center")
    S.despine(ax)

    # ------------------------------------------------ (b) quenched heterogeneity
    ax = f.add_subplot(gs[0, 1])
    d6 = pd.DataFrame(res("synthetic_S6.json"))
    d6["J1"] = d6.par.apply(lambda p: p["J1"])
    s = d6[d6["T"] == 1000]
    rng = np.random.default_rng(1)
    for i, J1 in enumerate(sorted(s.J1.unique())):
        q = s[s.J1 == J1]
        ax.plot([i - 0.42, i + 0.42], [J1, J1], color=S.INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.scatter(i - 0.17 + rng.uniform(-0.09, 0.09, len(q)), q.pooled_J1, s=3.5, color=S.CORAL, lw=0, alpha=0.85)
        ax.scatter(i + 0.17 + rng.uniform(-0.09, 0.09, len(q)), q.fe_J1, s=3.5, color=S.INDIGO, lw=0, alpha=0.85)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["0", "0.05", "0.1"])
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.02, 0.2)
    ax.set_yticks([0, 0.05, 0.1, 0.15])
    ax.set_xlabel(r"planted $J_1$")
    ax.set_ylabel(r"inferred $J_1$")
    q0 = s[s.J1 == 0]
    S.text(ax, -0.5, 0.19, f"one field for all:\n$K^*=6$ in {100 * np.mean(q0.pooled_bic == 6):.0f}%",
           c=S.CORAL, fontsize=5.8, ha="left", va="top", linespacing=1.1)
    S.text(ax, 2.55, 0.012, f"one field per walker:\n$K^*>0$ in {100 * np.mean(q0.fe_bic > 0):.0f}%",
           c=S.INDIGO, fontsize=5.8, ha="right", va="bottom", linespacing=1.1)
    S.despine(ax)

    # ------------------------------------------------ (c) segmentation
    ax = f.add_subplot(gs[0, 2])
    d5 = pd.DataFrame(res("synthetic_S5.json"))
    for k in ["rho", "dt"]:
        d5[k] = d5.par.apply(lambda p: p[k])
    s = d5[(d5.rho == 0.0) & (d5.dt == 0.25)]
    Ks = np.arange(6)
    for j, (col, c) in enumerate([("true_bic", S.MUTED), ("raw_bic", S.CORAL), ("merged_bic", S.INDIGO)]):
        p = [np.mean(s[col] == k) for k in Ks]
        ax.bar(Ks + (j - 1) * 0.27, p, width=0.25, color=c, edgecolor="none")
    ax.set_xticks(Ks)
    ax.set_xlabel(r"selected depth $K^*$")
    ax.set_ylabel("fraction of tracks")
    ax.set_ylim(0, 1.08); ax.set_xlim(-0.6, 5.5)
    S.text(ax, 5.45, 0.86, "true flights", c=S.MUTED, fontsize=5.9, ha="right", va="top")
    S.text(ax, 5.45, 0.76, "segmented", c=S.CORAL, fontsize=5.9, ha="right", va="top")
    S.text(ax, 5.45, 0.66, "segmented, merged", c=S.INDIGO, fontsize=5.9, ha="right", va="top")
    S.despine(ax)

    # ------------------------------------------------ (d) signatures of the mechanisms
    ax = f.add_subplot(gs[0, 3])
    d4 = pd.DataFrame(res("synthetic_S4.json"))
    d4["gen"] = d4.par.apply(lambda p: p["gen"]); d4["Kmax"] = d4.par.apply(lambda p: p["Kmax"])
    s = d4[d4.Kmax == 12]
    spec = [("fgn", "#1B2550", "long-range (fGn)"), ("maxent3", S.INDIGO, r"MaxEnt, $K=3$"),
            ("ccrw", S.TEAL, "hidden states"), ("copAR2", S.mix(S.INDIGO, 0.5), "copula AR(2)"),
            ("iid", S.MUTED, "renewal")]
    ypos = {"fgn": 8.6, "maxent3": 3.25, "ccrw": 2.2, "copAR2": 1.55, "iid": 0.25}
    for g, c, lab in spec:
        q = s[s.gen == g]
        m = q.groupby("T")["bic"].mean(); e = q.groupby("T")["bic"].std() / np.sqrt(q.groupby("T")["bic"].size())
        ax.errorbar(m.index, m.values, yerr=e.values, color=c, marker="o", ms=2.8, lw=1.2, elinewidth=0.6)
        S.text(ax, 1.75e4 * 1.12, ypos[g], lab, c=c, fontsize=5.8, ha="left", va="center")
    ax.axhline(3, color=S.INDIGO, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([250, 1000, 4000, 16000]); ax.set_xticklabels(["250", "1000", "4000", "16000"])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(180, 1.9e4); ax.set_ylim(-0.3, 9.4)
    ax.set_xlabel(r"flights per walker $T$")
    ax.set_ylabel(r"mean selected depth $K^*$")
    ax.spines["right"].set_visible(False)
    S.despine(ax)
    axs = f.axes
    S.heads(f, [(axs[0], "(a)", "the right variable"), (axs[1], "(b)", "heterogeneity invents memory"),
                (axs[2], "(c)", "segmentation invents memory"), (axs[3], "(d)", "signatures of mechanisms")])
    S.save(f, "figA3_pitfalls")


if __name__ == "__main__":
    make()
