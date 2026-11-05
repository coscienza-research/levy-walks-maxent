"""
panels.py -- one function per panel, shared by the figure modules.

Main text:  fig1 (model and phase diagram), fig2 (many walkers), fig3 (single walkers in the data),
            fig4 (real groups).
Appendix:   figA1 (theory details), figA2 (validation), figA3 (synthetic controls),
            figA4 (controls on real walkers), figA5 (collective: drives and fluctuations).
Each function draws on the axes it is given and returns nothing unless stated.
"""
from common import S, np, pd, mpl, os, DATA, GROUP, GMARK, SHORT, FancyArrowPatch
from common import order_parameter, apparent_mu, dat, res
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch

LS = [np.log(10.0 ** d) for d in (2, 3, 6, 12)]
LS_LAB = [r"$10^2$", r"$10^3$", r"$10^6$", r"$10^{12}$"]
LS_COL = [S.mix(S.INDIGO, w) for w in (0.35, 0.58, 0.8)] + ["#1B2550"]
PHIS = np.linspace(0, 3.2, 161)


# ============================================================================ theory of one walker
def langevin_response(ax):
    from maxent_levy import langevin, langevin_prime
    y = np.linspace(-24, 24, 801)
    ax.axhline(0.5, color=S.GRID, lw=0.6, zorder=0); ax.axvline(0, color=S.GRID, lw=0.6, zorder=0)
    ax.plot(y, 0.5 * (1 + langevin(y / 2)), color=S.INDIGO, lw=1.3)
    ax.plot(y, 3 * langevin_prime(y / 2), color=S.MUTED, lw=1.0, ls=(0, (3, 1.5)))
    ax.set_xlim(-24, 24); ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_xlabel(r"$\theta L$")
    S.text(ax, 23, 0.86, r"$\langle x\rangle/L$", c=S.INDIGO, fontsize=6.4, ha="right", va="top")
    S.text(ax, -23.5, 0.76, r"$12\,\mathrm{Var}(x)/L^2$", c=S.MUTED, fontsize=6.0, va="center", ha="left")
    S.text(ax, 23, 0.22, r"$\theta=1-\mu$", c=S.INK2, fontsize=6.0, ha="right")
    S.despine(ax)


def crossing(ax):
    cache = os.path.join(DATA, "fig2_pi.npy")
    if os.path.exists(cache):
        P = np.load(cache)
    else:
        P = np.array([[order_parameter(2.0, p, L, n=3001) for p in PHIS] for L in LS])
        np.save(cache, P)
    ax.axvline(1, color=S.INK2, lw=0.7, ls=(0, (1.3, 1.3)), zorder=0)
    ax.axvline(2, color=S.INK2, lw=0.7, ls=(0, (4, 2)), zorder=0)
    for row, c in zip(P, LS_COL):
        ax.plot(PHIS, row, color=c, lw=1.2)
    ax.plot(2, 0.5, "o", ms=3.6, color=S.INK, zorder=5)
    ax.set_xlim(0, 3.2); ax.set_ylim(-0.02, 1.03)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel(r"$\Phi=J_1L/(\mu_{\rm bare}-1)$")
    ax.set_ylabel(r"$\Pi=P(\ell>\sqrt{\ell_{\min}\ell_{\max}})$")
    S.text(ax, 0.95, 0.985, "spinodal", c=S.INK2, fontsize=5.8, ha="right", va="top")
    S.text(ax, 1.94, 0.985, "self-dual", c=S.INK2, fontsize=5.8, ha="right", va="top")
    for k, (c, lab) in enumerate(zip(LS_COL, LS_LAB)):
        ax.plot([2.38, 2.62], [0.40 - 0.085 * k] * 2, color=c, lw=1.2)
        S.text(ax, 2.68, 0.40 - 0.085 * k, lab, c=S.INK2, fontsize=5.8, va="center")
    S.text(ax, 2.38, 0.49, r"$\ell_{\max}/\ell_{\min}$", c=S.INK2, fontsize=5.8, va="center")
    S.despine(ax)


def dressed_exponent(ax):
    cache = os.path.join(DATA, "fig2_mu.npy")
    if os.path.exists(cache):
        M = np.load(cache)
    else:
        M = np.array([[apparent_mu(2.0, p, L, n=3001) for p in PHIS] for L in LS])
        np.save(cache, M)
    ax.axhline(2, color=S.GRID, lw=0.6, zorder=0)
    ax.axhline(1, color=S.GRID, lw=0.6, zorder=0)
    ax.axvline(1, color=S.INK2, lw=0.7, ls=(0, (1.3, 1.3)), zorder=0)
    ax.axvline(2, color=S.INK2, lw=0.7, ls=(0, (4, 2)), zorder=0)
    for row, c in zip(M, LS_COL):
        ax.plot(PHIS, row, color=c, lw=1.2)
    ax.plot(2, 1, "o", ms=3.6, color=S.INK, zorder=5)
    ax.set_xlim(0, 3.2); ax.set_ylim(-1.0, 2.25)
    ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([-1, 0, 1, 2])
    ax.set_xlabel(r"$\Phi=J_1L/(\mu_{\rm bare}-1)$")
    ax.set_ylabel(r"histogram exponent $\hat\mu$")
    S.text(ax, 3.15, 2.07, r"$\mu_{\rm bare}=2$", c=S.INK2, fontsize=5.8, ha="right", va="bottom")
    S.despine(ax)


# ============================================================================ single walkers in the data
def larval_rows():
    """conditions in the order of Table I, with gaps between controls, neural block and ablation"""
    main = res("drosophila_main.json")
    conds = [r["cond"] for r in main]
    ys = np.arange(len(conds))[::-1].astype(float)
    ys[6:] -= 0.45
    ys[9:] -= 0.45
    return main, conds, ys, dict(zip(conds, ys))


def larvae_coupling(ax):
    main, conds, ys, yof = larval_rows()
    ax.axvline(0, color=S.MUTED, lw=0.6, zorder=0)
    for r in main:
        y = yof[r["cond"]]
        sh = np.array(r["shuffle_J1"])
        ax.add_patch(Rectangle((sh.mean() - 2 * sh.std(), y - 0.32), 4 * sh.std(), 0.64, color=S.BAND, lw=0,
                               zorder=1))
        J1 = r["fe_J"]["1"][0]; se = r["fe_seJ"]["1"][0]
        mem = r["fe_K_bic"] > 0
        g = GROUP[r["cond"]]
        ax.errorbar(J1, y, xerr=2 * se, fmt="none", ecolor=S.INDIGO if mem else S.MUTED, elinewidth=0.9, zorder=3)
        ax.plot(J1, y, marker=GMARK[g], ms=4.0, color=S.INDIGO if mem else "white",
                mec=S.INDIGO if mem else S.MUTED, mew=0.8, zorder=4)
        ax.text(0.47, y, f"{r['fe_K_bic']}", color=S.INDIGO if mem else S.MUTED, fontsize=6.2, ha="center",
                va="center")
    ax.text(0.47, ys[0] + 0.95, r"$K^*$", color=S.INK2, fontsize=6.2, ha="center", va="center")
    ax.set_yticks(ys); ax.set_yticklabels([SHORT[c] for c in conds], fontsize=6.0)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(-0.33, 0.52); ax.set_ylim(ys[-1] - 0.7, ys[0] + 1.3)
    ax.set_xticks([-0.2, 0, 0.2, 0.4])
    ax.set_xlabel(r"lag-1 coupling $J_1$")
    S.despine(ax, left=True)


def larvae_robustness(ax, labels=True):
    main, conds, ys, yof = larval_rows()
    rob = res("drosophila_robust.json")
    blk = res("drosophila_block.json")
    settings = [(0.25, 0.5), (0.5, 0.5), (0.5, 1.0), (0.5, 2.0), (1.0, 2.0)]
    Z = np.full((len(conds), 8), np.nan)
    for i, c in enumerate(conds):
        for j, (eps, lmin) in enumerate(settings):
            src = main if (eps, lmin) == (0.5, 1.0) else rob
            q = [r for r in src if r["cond"] == c and r["eps"] == eps and r["lmin"] == lmin][0]
            J1 = q["fe_J"]["1"][0]; sh = np.array(q["shuffle_J1"])
            Z[i, j] = (J1 - sh.mean()) / sh.std()
        for j, B in enumerate([50, 25]):
            q = [b for b in blk if b["cond"] == c and b["B"] == B and b["fam"] == "powerlaw"][0]
            Z[i, 6 + j] = q["z"]
    cmap = LinearSegmentedColormap.from_list("div", [S.CORAL, S.mix(S.CORAL, 0.35), "#FFFFFF",
                                                     S.mix(S.INDIGO, 0.35), S.INDIGO, "#1B2550"])
    nrm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=16)
    xs = np.array([0, 1, 2, 3, 4, 5.4, 6.4])
    for i, c in enumerate(conds):
        for j, jj in enumerate([0, 1, 2, 3, 4, 6, 7]):
            v = Z[i, jj]
            ax.add_patch(Rectangle((xs[j] - 0.46, yof[c] - 0.42), 0.92, 0.84,
                                   color=cmap(nrm(np.clip(v, -8, 16))), lw=0))
            lab = "0" if abs(v) < 0.5 else f"{v:.0f}".replace("-", "\N{MINUS SIGN}")
            ax.text(xs[j], yof[c], lab, ha="center", va="center", fontsize=5.6, color="white" if v > 9 else S.INK)
    ax.set_xlim(-0.6, 6.95); ax.set_ylim(ys[-1] - 0.7, ys[0] + 1.3)
    if labels:
        ax.set_yticks(ys); ax.set_yticklabels([SHORT[c] for c in conds], fontsize=6.0)
        ax.tick_params(axis="y", length=0, pad=2)
    else:
        ax.set_yticks([])
    ax.set_xticks(xs)
    ax.set_xticklabels(["0.25\n0.5", "0.5\n0.5", "0.5\n1", "0.5\n2", "1\n2", "50", "25"], fontsize=5.7)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.text(2.0, ys[0] + 0.95, r"segmentation $\varepsilon$, $\ell_{\min}$ (mm)", ha="center", va="center",
            fontsize=6.0, color=S.INK2)
    ax.text(5.9, ys[0] + 0.95, "field per block", ha="center", va="center", fontsize=6.0, color=S.INK2)
    ax.set_xlabel(r"$z$-score of $J_1$ against permutations")
    ax.add_patch(Rectangle((2 - 0.5, ys[-1] - 0.5), 1.0, ys[0] - ys[-1] + 1.0, fill=False, ec=S.INK2, lw=0.6,
                           zorder=5))
    for sp in ax.spines.values():
        sp.set_visible(False)


def termite_depth(ax):
    R = res("termites.json")
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "x"} for r in R])
    k = np.bincount(df.K_bic, minlength=11)[:11] / len(df)
    kp = np.bincount([v for p in df.perm_K for v in p], minlength=11)[:11] / sum(len(p) for p in df.perm_K)
    kq = np.bincount(df.pipe_K, minlength=11)[:11] / len(df)
    xk = np.arange(11)
    ax.bar(xk[:1] - 0.21, kp[:1], width=0.38, color=S.MUTED, edgecolor="none")
    ax.bar(xk[:1] + 0.21, kq[:1], width=0.38, color=S.mix(S.MUTED, 0.5), edgecolor="none")
    ax.bar(xk[1:], k[1:], width=0.64, color=S.OCHRE, edgecolor="none")
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_ylim(0, 1.05); ax.set_xlim(-0.7, 10.5)
    ax.set_xlabel(r"selected depth $K^*$")
    ax.set_ylabel("fraction of termites")
    S.text(ax, 0.75, 0.97, "nulls", c=S.INK2, fontsize=5.8, va="top")
    S.text(ax, 5.3, 0.36, "data", c=S.OCHRE, fontsize=6.0, ha="left")
    S.despine(ax)


def termite_kernel(ax):
    R = res("termites.json")
    J = np.array([r["J"] for r in R])                   # K = 10 fits
    tau = np.arange(1, J.shape[1] + 1)
    lo, med, hi = np.percentile(J, [25, 50, 75], axis=0)
    ax.axhline(0, color=S.MUTED, lw=0.6)
    for j in J:
        ax.plot(tau, j, color=S.mix(S.OCHRE, 0.25), lw=0.4, zorder=1)
    ax.fill_between(tau, lo, hi, color=S.mix(S.OCHRE, 0.45), lw=0, zorder=2)
    ax.plot(tau, med, color=S.OCHRE, lw=1.4, marker="o", ms=2.6, zorder=3)
    ax.set_xticks([1, 4, 7, 10]); ax.set_xlim(0.6, 10.4)
    ax.set_ylim(-0.12, 0.62)
    ax.set_xlabel(r"lag $\tau$ (flights)")
    ax.set_ylabel(r"coupling $J_\tau$")
    S.despine(ax)


def dressing_scatter(ax):
    """exponent of the histogram against that of the one-step rule: memory pushes points below the diagonal"""
    main = {r["cond"]: r for r in res("drosophila_main.json")}
    T = dat("termites_K1.json")
    tb, ti = np.array([t["mu_bare"] for t in T]), np.array([t["mu_iid"] for t in T])
    lim = (0.9, 3.1)
    ax.plot(lim, lim, color=S.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.scatter(tb, ti, s=6, color=S.OCHRE, edgecolor="white", lw=0.3, zorder=2)
    for q in dat("larvae_K1.json"):
        if main[q["cond"]]["fe_K_bic"] == 0:
            continue
        g = GROUP[q["cond"]]
        ax.plot(np.median(q["mu_bare"]), np.median(q["mu_iid"]), marker=GMARK[g], ms=4.0, color=S.INDIGO,
                mec="white", mew=0.5, zorder=4)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xticks([1, 1.5, 2, 2.5, 3]); ax.set_yticks([1, 1.5, 2, 2.5, 3])
    ax.set_aspect("equal")
    ax.set_xlabel(r"one-step rule $\mu_{\rm bare}$")
    ax.set_ylabel(r"histogram $\hat\mu$")
    S.text(ax, 2.25, 2.31, "no memory", c=S.MUTED, fontsize=5.6, ha="right", va="bottom", rotation=45,
           rotation_mode="anchor")
    S.text(ax, 2.95, 1.12, "termites", c=S.OCHRE, fontsize=6.0, ha="right")
    S.text(ax, 1.05, 2.95, "larvae", c=S.INDIGO, fontsize=6.0, ha="left", va="top")
    S.despine(ax)


def termite_wall(ax):
    w = pd.DataFrame(res("termites_wall.json"))
    s = w[w.d_wall == 6.0]
    a = np.array([p["J1_nowall"] for p in s.powerlaw]); b = np.array([p["J1_wall"] for p in s.powerlaw])
    ax.plot([0.1, 0.8], [0.1, 0.8], color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.scatter(a, b, s=6, color=S.OCHRE, edgecolor="white", lw=0.3)
    ax.set_xlim(0.12, 0.78); ax.set_ylim(0.12, 0.78)
    ax.set_xticks([0.2, 0.4, 0.6]); ax.set_yticks([0.2, 0.4, 0.6])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$J_1$ without wall field")
    ax.set_ylabel(r"$J_1$ with wall field")
    S.despine(ax)


TC_LAB = {"fish_T:control": "zebrafish T", "fish_T:rockout": "zebrafish T, ROCK-inh.",
          "Gerard_mouse_T:control": "mouse T", "Gerard_mouse_T:Myo1g_KO": "mouse T, Myo1g KO",
          "Gautreau_dicty:WT": "Dictyostelium"}


def tcell_shuffle(ax):
    T = res("tcells.json")
    shp = {r["group"]: r["shuffled_pooled_K"] for r in res("tcells_shuffle_pooled.json")}
    yy = np.arange(len(T))[::-1] - 0.6
    for y_, r in zip(yy, T):
        ps = np.array(shp[r["group"]]); fs = np.array(r["shuffled_fe_K"])
        ax.plot([np.median(ps)], [y_ + 0.18], "x", ms=4.0, mew=1.1, color=S.CORAL)
        ax.plot([np.median(fs)], [y_ + 0.18], "o", ms=3.4, mfc="white", mec=S.MUTED, mew=0.8)
        ax.plot([r["fe_K"]], [y_ - 0.18], "o", ms=3.6, color=S.SLATE, mec="white", mew=0.4)
    ax.set_yticks(yy); ax.set_yticklabels([TC_LAB[r["group"]] for r in T], fontsize=5.9)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlim(-0.5, 6.6); ax.set_xticks(range(7)); ax.set_ylim(-1.05, len(T) + 0.9)
    ax.set_xlabel(r"selected depth $K^*$")
    ax.plot(0.0, len(T) + 0.55, "x", ms=3.6, mew=1.0, color=S.CORAL)
    ax.text(0.35, len(T) + 0.55, "shuffled, one field", color=S.CORAL, fontsize=5.6, va="center")
    ax.plot(0.0, len(T) + 0.05, "o", ms=3.0, mfc="white", mec=S.MUTED, mew=0.8)
    ax.text(0.35, len(T) + 0.05, "shuffled, field per cell", color=S.MUTED, fontsize=5.6, va="center")
    ax.plot(0.0, len(T) - 0.45, "o", ms=3.2, color=S.SLATE, mec="white", mew=0.4)
    ax.text(0.35, len(T) - 0.45, "data, field per cell", color=S.SLATE, fontsize=5.6, va="center")
    S.despine(ax, left=True)


# ============================================================================ many walkers
def _node(ax, x, y, r, txt, ec, fc, fs=6.0, lw=0.8):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec, lw=lw, zorder=3))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, color=S.INK, zorder=4)


def _arrow(ax, p, q, c, lw=0.8, rad=0.0, ms=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=2))


def self_social_schematic(ax):
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 7.2); ax.set_aspect("equal"); ax.axis("off")
    n = 6; s0 = 0.62; x0, y0 = 0.35, 1.25
    for i in range(n):
        for j in range(n):
            c = S.INDIGO if i == j else S.mix(S.PLUM, 0.35)
            ax.add_patch(Rectangle((x0 + j * s0, y0 + (n - 1 - i) * s0), s0 * 0.92, s0 * 0.92, color=c, lw=0))
    ax.text(x0 + n * s0 / 2, y0 + n * s0 + 0.35, r"$\langle x_i^t\,x_j^{t-\tau}\rangle$", ha="center",
            va="bottom", fontsize=6.4, color=S.INK)
    ax.text(x0 - 0.15, y0 + n * s0 / 2, r"$i$", ha="right", va="center", fontsize=6.2, color=S.INK2)
    ax.text(x0 + n * s0 / 2, y0 - 0.15, r"$j$", ha="center", va="top", fontsize=6.2, color=S.INK2)
    ax.text(x0 + n * s0 + 0.18, y0 + 0.31, "own " + r"$A(\tau)$", color=S.INDIGO, fontsize=5.8, ha="left",
            va="center")
    ax.text(x0 + n * s0 / 2, 0.05, r"others' $S(\tau)$", color=S.PLUM, fontsize=5.8, ha="center", va="center")
    cx, cy = 9.0, 3.6
    _node(ax, cx, cy, 0.62, r"$x_i^t$", S.INK, "#FFFFFF", fs=6.4)
    for k, (dx, lab) in enumerate([(-2.4, r"$x_i^{t-1}$"), (-4.0, r"$x_i^{t-2}$")]):
        _node(ax, cx + dx, cy, 0.55, lab, S.INDIGO, S.mix(S.INDIGO, 0.12), fs=5.6)
        _arrow(ax, (cx + dx + 0.55, cy + (0.0 if k == 0 else 0.25)), (cx - 0.62, cy + (0.0 if k == 0 else 0.15)),
               S.INDIGO, rad=0.0 if k == 0 else -0.35)
    ax.text(cx - 1.55, cy - 0.32, r"$a_1$", color=S.INDIGO, fontsize=6.0, ha="center", va="top")
    ax.text(cx - 2.6, cy + 1.15, r"$a_2$", color=S.INDIGO, fontsize=6.0, ha="center", va="bottom")
    for dx in [-1.3, 0.0, 1.3]:
        _node(ax, cx + dx, cy + 2.55, 0.42, "", S.PLUM, S.mix(S.PLUM, 0.15))
        _arrow(ax, (cx + dx * 0.9, cy + 2.13), (cx + dx * 0.25, cy + 0.6), S.PLUM)
    ax.text(cx + 1.85, cy + 2.55, r"$x_j^{t-\tau}$", color=S.PLUM, fontsize=5.8, ha="left", va="center")
    ax.text(cx + 0.95, cy + 1.25, r"$J_\tau$", color=S.PLUM, fontsize=6.0, ha="left", va="center")
    for dx, lab in [(-0.75, r"$h_i$"), (0.75, r"$g_t$")]:
        ax.add_patch(FancyBboxPatch((cx + dx - 0.42, cy - 2.55), 0.84, 0.62, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=S.mix(S.TEAL, 0.15), edgecolor=S.TEAL, lw=0.7, zorder=3))
        ax.text(cx + dx, cy - 2.24, lab, color=S.TEAL, fontsize=6.0, ha="center", va="center", zorder=4)
        _arrow(ax, (cx + dx * 0.9, cy - 1.93), (cx + dx * 0.3, cy - 0.6), S.TEAL)
    ax.text(cx, cy - 3.25, "unit / frame fields", color=S.TEAL, fontsize=5.6, ha="center", va="center")


def aggregation(ax, inset=True):
    R2 = pd.DataFrame(res("collective_C2.json")); R2["N"] = R2.par.apply(lambda p: p["N"])
    s = R2[R2.N == 30]
    Jpre = np.array([p["J8"] for p in s.pre]); pred = np.array([p["pred8"] for p in s.pre])
    Jpa = np.array([p["J8"] for p in s.pa]); Jo = np.array([p["J6"] for p in s.ours])
    tau = np.arange(1, 9)
    ax.axhline(0, color=S.MUTED, lw=0.6)
    ax.plot(tau, pred.mean(0), color=S.INK2, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.errorbar(tau - 0.13, Jpre.mean(0), yerr=Jpre.std(0), fmt="s", color=S.CORAL, mfc="white", ms=3.2, mew=0.8,
                elinewidth=0.8, zorder=3)
    ax.errorbar(tau + 0.0, Jpa.mean(0), yerr=Jpa.std(0), fmt="D", color=S.mix(S.CORAL, 0.6), ms=2.8,
                elinewidth=0.8, zorder=3)
    ax.errorbar(np.arange(1, 7) + 0.13, Jo.mean(0), yerr=Jo.std(0), fmt="o", color=S.PLUM, ms=3.2, elinewidth=0.8,
                zorder=4)
    ax.set_xticks(tau); ax.set_ylim(-0.17, 0.6); ax.set_xlim(0.5, 8.5); ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel(r"delay $\tau$ (frames)")
    ax.set_ylabel(r"social coupling $J_\tau$")
    S.text(ax, 8.45, 0.59, "group average:\noriginal (open), exact (filled)", c=S.CORAL, fontsize=5.6, va="top",
           ha="right", linespacing=1.05)
    S.text(ax, 3.3, 0.235, "predicted", c=S.INK2, fontsize=5.8, va="center")
    S.text(ax, 0.65, -0.145, "self + social", c=S.PLUM, fontsize=5.8, ha="left", va="center")
    S.despine(ax)
    if not inset:
        return
    ins = ax.inset_axes([0.68, 0.40, 0.30, 0.27])
    rows = []
    for N, q in R2.groupby("N"):
        rows.append((N, np.mean([p["tauM"] > 0 for p in q.pre]), np.mean([p["sel"][1] > 0 for p in q.ours])))
    rows = np.array(rows)
    xi = np.arange(len(rows))
    ins.bar(xi - 0.2, rows[:, 1], width=0.36, color=S.CORAL)
    ins.bar(xi + 0.2, np.maximum(rows[:, 2], 0.012), width=0.36, color=S.PLUM)
    ins.set_xticks(xi); ins.set_xticklabels([f"{int(v)}" for v in rows[:, 0]], fontsize=5.0)
    ins.set_yticks([0, 1]); ins.set_yticklabels(["0", "1"], fontsize=5.0)
    ins.tick_params(length=1.2, pad=0.8)
    ins.set_ylim(0, 1.1)
    ins.set_title(r"declared vs $N$", fontsize=5.2, color=S.INK2, pad=1, loc="center")
    S.despine(ins)


def common_drives(ax_rate, ax_J):
    R3 = pd.DataFrame(res("collective_C3.json")); R3["inter"] = R3.par.apply(lambda p: len(p["Jloc"]) > 0)
    no = R3[~R3.inter]; yes = R3[R3.inter]
    rates = [np.mean([p["tauM"] > 0 for p in no.pre]), np.mean([p["sel"][1] > 0 for p in no["global"]]),
             np.mean([p["sel"][1] > 0 for p in no["local"]]), np.mean([p["sel"][1] > 0 for p in no["local_tfe"]])]
    ax = ax_rate
    ax.bar(range(4), rates, color=[S.CORAL, S.CORAL, S.CORAL, S.PLUM], width=0.62)
    ax.text(3, 0.03, "0", ha="center", va="bottom", fontsize=6.2, color=S.PLUM)
    ax.set_xticks(range(4)); ax.set_xticklabels(["avg.", "glob.", "local", "local\n+ $g_t$"], fontsize=5.4)
    ax.get_xticklabels()[3].set_color(S.PLUM)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_ylabel("social coupling declared")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.22); ax.set_yticks([0, 0.5, 1.0])
    S.text(ax, 1.5, 1.2, "no interaction", c=S.INK2, fontsize=5.6, ha="center", va="top")
    S.despine(ax)
    ax = ax_J
    J_loc = np.array([p["J4"][1] for p in yes["local"]]); J_tfe = np.array([p["J4"][1] for p in yes["local_tfe"]])
    ax.axhline(0.3, color=S.INK, lw=0.7, ls=(0, (3, 2)))
    for x, v, c in [(0, J_loc, S.CORAL), (1, J_tfe, S.PLUM)]:
        ax.errorbar(x, v.mean(), yerr=v.std(), fmt="o", color=c, ms=3.2, elinewidth=0.8)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["local", "local\n+ $g_t$"], fontsize=5.8)
    ax.get_xticklabels()[1].set_color(S.PLUM)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_ylabel(r"inferred $J_2$")
    S.text(ax, 1.58, 0.31, "planted", c=S.INK2, fontsize=5.6, ha="right", va="bottom")
    S.despine(ax)


def mf_phase(ax):
    from meanfield import Lam, dLam, critical_point
    L = float(np.log(100.0))
    Jc, _, _ = critical_point(L)
    th = np.linspace(-6, 6, 4001)
    lp = dLam(th, L)
    Jp = 1 / lp; tt_p = th - Jp * (Lam(th, L) - L / 2)
    Jn = -1 / lp; tt_n = th + (1 / lp) * (Lam(th, L) - L / 2)
    xmax = 3.0
    sel = Jp / Jc <= xmax
    ax.fill_betweenx(tt_p[sel], Jp[sel] / Jc, xmax, color=S.mix(S.PLUM, 0.22), lw=0)
    ax.plot(Jp[sel] / Jc, tt_p[sel], color=S.PLUM, lw=1.1)
    sel = -Jn / Jc <= xmax
    ax.fill_betweenx(tt_n[sel], Jn[sel] / Jc, -xmax, color=S.mix(S.PLUM, 0.10), lw=0)
    ax.plot(Jn[sel] / Jc, tt_n[sel], color=S.PLUM, lw=1.1, ls=(0, (3, 1.5)))
    ax.plot([1, -1], [0, 0], "o", color=S.INK, ms=3.0, zorder=5)
    ax.axhline(0, color=S.GRID, lw=0.5, zorder=0)
    ax.set_xlim(-xmax, xmax); ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(r"delayed coupling $J/J_c$")
    ax.set_ylabel(r"bias $\tilde\theta=H+JL/2$")
    ax.annotate("bistable\n(stop-and-go)", xy=(2.55, 0.25), xytext=(2.05, 2.35), color=S.PLUM, fontsize=5.8,
                ha="center", va="center", linespacing=1.05,
                arrowprops=dict(arrowstyle="-", color=S.PLUM, lw=0.6, shrinkA=1, shrinkB=0))
    ax.annotate("oscillating", xy=(-2.3, -0.6), xytext=(-1.25, -2.6), color=S.PLUM, fontsize=5.8, ha="center",
                va="center", arrowprops=dict(arrowstyle="-", color=S.PLUM, lw=0.6, shrinkA=1, shrinkB=0))
    S.text(ax, 0.0, 2.9, "one steady\nstate", c=S.INK2, fontsize=5.8, ha="center", va="center", linespacing=1.0)
    S.despine(ax)


def mf_series(f, cell):
    """three short time series of the collective log-speed; returns the three axes"""
    from meanfield import critical_point, simulate_mf
    L = float(np.log(100.0))
    Jc, _, _ = critical_point(L)
    gse = cell.subgridspec(3, 1, hspace=0.42)
    rng = np.random.default_rng(5)
    cases = [("steady, $J=0.5J_c$", [0.5 * Jc], S.MUTED, 200, 150),
             ("stop-and-go, $J=1.15J_c$", [1.15 * Jc], S.PLUM, 40, 1500),
             ("oscillation, $J_3=-1.3J_c$", [0, 0, -1.3 * Jc], S.PLUM, 200, 75)]
    axes = []
    for k, (lab, J, col, N, T) in enumerate(cases):
        a_ = f.add_subplot(gse[k, 0])
        H = -sum(J) * L / 2
        m = simulate_mf(N, max(T, 400), H, J, L, rng)[:T]
        a_.plot(np.arange(len(m)), m / L, color=col, lw=0.6 if k == 1 else 0.9)
        a_.set_ylim(0.0, 1.32); a_.set_yticks([0.5]); a_.set_yticklabels(["½"], fontsize=5.6)
        a_.set_xlim(0, T)
        a_.set_xticks([0, T]); a_.set_xticklabels(["", f"{T}"], fontsize=5.4)
        a_.tick_params(length=1.5, pad=1)
        a_.axhline(0.5, color=S.GRID, lw=0.5, zorder=0)
        S.text(a_, 0.01, 0.99, lab, c=col if col != S.MUTED else S.INK2, fontsize=5.6, transform=a_.transAxes,
               va="top")
        S.despine(a_)
        axes.append(a_)
    axes[-1].set_xlabel("frame $t$", labelpad=0.5)
    axes[1].set_ylabel(r"$m_t/L$")
    return axes


def mf_fluctuations(ax):
    from meanfield import critical_point, unit_response, nvar_linear_response
    L = float(np.log(100.0))
    Jc, _, _ = critical_point(L)
    fr = np.linspace(-0.97, 0.97, 300)
    lp0 = L ** 2 / 12
    ax.plot(fr, lp0 / (1 - fr ** 2), color=S.INK2, lw=1.0)
    a = 0.2
    R_, Cs, _ = unit_response(-a * L / 2, a, L)
    fc = 1.0 / R_.sum() / Jc
    fe = np.linspace(-0.97, 0.985 * fc, 300)
    ax.plot(fe, [nvar_linear_response(v * Jc, R_, Cs) for v in fe], color=S.INDIGO, lw=1.0)
    ax.axvline(fc, color=S.INDIGO, lw=0.6, ls=(0, (1.2, 1.2)))
    df = pd.DataFrame(res("collective_C6.json"))
    df["frac"] = df.par.apply(lambda p: p["frac"]); df["a"] = df.par.apply(lambda p: p["a"])
    for a_, col, mk in [(0.0, S.INK2, "o"), (0.2, S.INDIGO, "s")]:
        q = df[df.a == a_].groupby("frac").NVar.agg(["mean", "std"]).reset_index()
        bist = (a_ > 0) & (q.frac > fc)
        ax.errorbar(q.frac[~bist], q["mean"][~bist], yerr=q["std"][~bist], fmt=mk, color=col, ms=2.8,
                    mfc="white" if a_ == 0 else col, mew=0.7, elinewidth=0.7)
    ax.set_yscale("log"); ax.set_ylim(1.0, 120); ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r"$J/J_c$ on $\tilde\theta=0$")
    ax.set_ylabel(r"$N\,\mathrm{Var}(m_t)$")
    S.text(ax, fc - 0.08, 100, r"$J_c(a)=0.71J_c$", c=S.INDIGO, fontsize=5.7, ha="right", va="center")
    S.text(ax, -0.55, 3.4, "no own memory", c=S.INK2, fontsize=5.7, ha="left", va="center")
    S.text(ax, -0.97, 1.2, r"own memory $a=0.2$", c=S.INDIGO, fontsize=5.7, ha="left", va="center")
    S.despine(ax)


# ============================================================================ real groups
def _null(R, v, key):
    return np.array([n[v][key]["J"] for n in R["null"]])


def lags_panel(ax, R, dt, unit="s", ylab=True, show_global=True):
    """delayed couplings to the nearest neighbors (directions), data vs time-shift null; squares, the global
    coupling without time fields"""
    q = R["data"]["dir"]
    J, se = np.array(q["local_T"]["J"]), np.array(q["local_T"]["se_J"])
    Jn = _null(R, "dir", "local_T")
    tau = dt * np.arange(1, len(J) + 1)
    m, s = Jn.mean(0), Jn.std(0)
    ax.axhline(0, color=S.MUTED, lw=0.6)
    ax.fill_between(tau, m - 2 * s, m + 2 * s, color=S.BAND2, lw=0, zorder=1)
    ax.plot(tau, m, color=S.MUTED, lw=0.9, zorder=2)
    Jg = np.array(q["global_noT"]["J"])
    if show_global:
        ax.plot(tau, Jg, "s", mfc="white", mec=S.CORAL, mew=0.8, ms=3.2, zorder=3)
    ax.errorbar(tau, J, yerr=2 * se, fmt="o", color=S.PLUM, ms=3.4, lw=0.8, zorder=4)
    ax.plot(tau, J, color=S.PLUM, lw=0.8, zorder=4)
    ax.set_xlabel(rf"delay $\tau$ ({unit})")
    if ylab:
        ax.set_ylabel(r"coupling $J_\tau$")
    ax.set_xticks(tau)
    S.despine(ax)
    return tau, J, Jg, m


def rank_panel(ax, R, ranks, ylab=True):
    q = R["data"]["dir"]
    J1 = np.array(q["ranks"]["J"])[:, 0]; se = np.array(q["ranks"]["se"])[:, 0]
    Jn = _null(R, "dir", "ranks")[:, :, 0]
    x = np.array([r[1] for r in ranks])
    ax.axhline(0, color=S.MUTED, lw=0.6)
    ax.fill_between(x, Jn.mean(0) - 2 * Jn.std(0), Jn.mean(0) + 2 * Jn.std(0), color=S.BAND2, lw=0)
    ax.plot(x, Jn.mean(0), color=S.MUTED, lw=0.9)
    ax.errorbar(x, J1, yerr=2 * se, fmt="o-", color=S.PLUM, ms=3.4, lw=0.8)
    ax.set_xlabel("neighbor rank")
    if ylab:
        ax.set_ylabel(r"$J_1$ of one neighbor")
    ax.set_xticks(x)
    S.despine(ax)


def ahead_panel(ax, R, dt, unit="s", ylab=True):
    q = R["data"]["dir"]
    A, As = np.array(q["ahead_behind"]["J"]), np.array(q["ahead_behind"]["se"])
    An = _null(R, "dir", "ahead_behind")
    tau = dt * np.arange(1, A.shape[1] + 1)
    ax.axhline(0, color=S.MUTED, lw=0.6)
    for k, (mfc, lab) in enumerate(((S.PLUM, "ahead"), ("white", "behind"))):
        ax.plot(tau, An[:, k].mean(0), color=S.MUTED, lw=0.9, ls="-" if k == 0 else "--")
        ax.errorbar(tau + (k - 0.5) * dt * 0.12, A[k], yerr=2 * As[k], fmt="o", mfc=mfc, mec=S.PLUM, color=S.PLUM,
                    ms=3.4, lw=0.8, mew=0.8)
        S.text(ax, tau[-1] + dt * 0.25, A[k, -1], lab, c=S.PLUM, fontsize=5.8, va="center")
    ax.set_xlabel(rf"delay $\tau$ ({unit})")
    if ylab:
        ax.set_ylabel(r"coupling $J_\tau$")
    ax.set_xticks(tau); ax.set_xlim(tau[0] - dt * 0.5, tau[-1] + dt * 1.4)
    S.despine(ax)


def rank_norm_panel(ax, R, ranks):
    """one-neighbor couplings at the first delay, relative to the nearest neighbor: headings and speeds"""
    x = np.array([r[1] for r in ranks])
    ax.axhline(0, color=S.MUTED, lw=0.6)
    for v, mk, lab in (("dir", "o", "headings"), ("mod", "D", "speeds")):
        q = R["data"][v]
        J1 = np.array(q["ranks"]["J"])[:, 0]; se = np.array(q["ranks"]["se"])[:, 0]
        Jn = _null(R, v, "ranks")[:, :, 0].mean(0)
        ax.plot(x, Jn / J1[0], color=S.MUTED, lw=0.9, ls="-" if v == "dir" else "--")
        ax.errorbar(x, J1 / J1[0], yerr=2 * se / J1[0], fmt=mk + "-", color=S.PLUM,
                    mfc=S.PLUM if v == "dir" else "white", ms=3.2, lw=0.8, mew=0.8)
        S.text(ax, x[-1] + 0.15, J1[-1] / J1[0], lab, c=S.PLUM, fontsize=5.8, va="center")
    S.text(ax, 1.05, 0.21, "time-shifted, speeds", c=S.INK2, fontsize=5.6, ha="left")
    S.text(ax, 1.05, 0.05, "time-shifted, headings", c=S.INK2, fontsize=5.6, ha="left")
    ax.set_xlabel("neighbor rank"); ax.set_ylabel(r"$J_1$ relative to nearest")
    ax.set_xticks(x); ax.set_xlim(0.7, x[-1] + 1.1); ax.set_ylim(-0.1, 1.2); ax.set_yticks([0, 0.5, 1.0])
    S.despine(ax)


def group_average_depth(ax):
    """memory depth declared by the group-average analysis, on the data and on time-shifted surrogates
    (individual tracks displaced in time, which removes every interaction): it does not tell them apart"""
    B = res("realgroups_baboons10.json"); P = res("realgroups_pigeons.json")
    T = res("collective_real_tcells.json")
    rows = [("T cells", [T["pre"]["tauM"]], T["null"]["pre_tauM"], 8, "frames"),
            ("baboons", [g["tauM"] for g in B["data"]["dir"]["group_average"]],
             [g["tauM"] for n in B["null"] for g in n["dir"]["group_average"]], B["spec"]["tmax"], "days"),
            ("pigeons", [g["tauM"] for g in P["data"]["dir"]["group_average"]],
             [g["tauM"] for n in P["null"] for g in n["dir"]["group_average"]], P["spec"]["tmax"], "releases")]
    ys = np.arange(len(rows))[::-1].astype(float)
    ax.axvline(1, color=S.GRID, lw=0.6, zorder=0)
    for y, (lab, d, n, tmax, _) in zip(ys, rows):
        for vals, dy, kw in [(np.array(d) / tmax, 0.14, dict(color=S.CORAL, mec=S.CORAL)),
                             (np.array(n) / tmax, -0.14, dict(color="white", mec=S.MUTED))]:
            q1, q2, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([q1, q3], [y + dy] * 2, color=kw["mec"], lw=1.0, zorder=2)
            ax.plot(q2, y + dy, "o", ms=3.8, mew=0.9, zorder=3, **{"mfc": kw["color"], "mec": kw["mec"]})
        S.text(ax, 1.04, y, f"max {tmax}", c=S.INK2, fontsize=5.6, va="center", ha="left")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(-0.03, 1.3); ax.set_xticks([0, 0.5, 1]); ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.85)
    ax.set_xlabel(r"selected depth $\tau_M$ / largest tried")
    ax.plot(0.02, ys[0] + 0.55, "o", ms=3.4, color=S.CORAL, mec=S.CORAL)
    ax.text(0.08, ys[0] + 0.55, "data", color=S.CORAL, fontsize=5.8, va="center")
    ax.plot(0.36, ys[0] + 0.55, "o", ms=3.4, mfc="white", mec=S.MUTED, mew=0.9)
    ax.text(0.42, ys[0] + 0.55, "time-shifted", color=S.INK2, fontsize=5.8, va="center")
    S.despine(ax, left=True)
