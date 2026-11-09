"""Fig. 1 -- the whole paper in one figure: a larva cut into flights, flights as spins in time,
the phase diagram of the memory chain with the real walkers on it, and what the phases look like."""
from common import *  # noqa: F401,F403
from common import S, np, pd, mpl, os, FancyArrowPatch, PHASE, GROUP, GMARK, DATA
from common import stationary, order_parameter, ccdf_of, draw_track, draw_spin, larva_track, dat


def make():
    f = S.fig(S.FULL, 92)
    gs = f.add_gridspec(2, 3, width_ratios=[1.02, 1.16, 0.80], height_ratios=[0.86, 1.0],
                        left=0.006, right=0.988, top=0.93, bottom=0.105, wspace=0.40, hspace=0.42)
    # ------------------------------------------------ (a) a larval track cut into flights
    xy, lens, st = larva_track()
    xy = np.column_stack([xy[:, 1], -xy[:, 0]])          # rotated: long axis horizontal
    axa = f.add_subplot(gs[0, 0])
    ends = list(st) + [len(xy) - 1]
    axa.plot(xy[:, 0], xy[:, 1], color="#E2E1DC", lw=0.45, zorder=1)
    cmap_a = plt.cm.viridis
    norm_a = LogNorm(vmin=1.0, vmax=20.0)
    for i in np.argsort(lens):
        a_, b_ = ends[i], ends[i + 1]
        c_ = cmap_a(norm_a(np.clip(lens[i], 1.0, 20.0)))
        axa.plot([xy[a_, 0], xy[b_, 0]], [xy[a_, 1], xy[b_, 1]], color=c_, lw=0.85, zorder=2,
                 solid_capstyle="round")
    tp = xy[ends]
    axa.scatter(tp[:, 0], tp[:, 1], s=1.8, color=S.INK2, lw=0, zorder=3, alpha=0.55)
    axa.set_aspect("equal"); axa.axis("off")
    x0, x1 = axa.get_xlim(); y0, y1 = axa.get_ylim()
    axa.set_ylim(y0 - 0.32 * (y1 - y0), y1)
    yb = y0 - 0.20 * (y1 - y0)
    axa.plot([x0 + 1, x0 + 21], [yb, yb], color=S.INK, lw=1.1, solid_capstyle="butt")
    S.text(axa, x0 + 11, yb + 2.0, "20 mm", ha="center", va="bottom", fontsize=5.8)
    # colorbar for flight lengths
    cax_a = axa.inset_axes([0.42, 0.08, 0.52, 0.05])
    cb_a = f.colorbar(mpl.cm.ScalarMappable(norm=norm_a, cmap=cmap_a), cax=cax_a, orientation="horizontal")
    cb_a.outline.set_visible(False)
    cb_a.set_ticks([1, 2, 5, 10, 20])
    cb_a.set_ticklabels(["1", "2", "5", "10", "20"])
    cax_a.tick_params(labelsize=5.4, length=1.4, pad=1.0, color=S.AXIS)
    cax_a.set_title(r"flight length $\ell$ (mm)", fontsize=5.8, color=S.INK2, pad=2.5, loc="left")

    # ------------------------------------------------ (b) flights as spins in time
    axb = f.add_subplot(gs[1, 0])
    axb.set_xlim(0, 10); axb.set_ylim(0, 6.4); axb.set_aspect("equal"); axb.axis("off")
    Lb = float(np.log(lens.max() * 1.001))
    i0 = 122                                            # short, short, long, long
    seq = np.clip(np.log(lens[i0:i0 + 4]), 0, Lb)
    u = 2 * seq / Lb - 1
    xs_c = np.array([1.15, 3.55, 5.95, 8.55]); yc = 3.05; r = 0.98
    for i, (cx, uu) in enumerate(zip(xs_c, u)):
        last = i == len(u) - 1
        draw_spin(axb, cx, yc, r, uu, col=S.INDIGO, lw=1.1 if last else 0.9,
                  fill="#DFE5F3" if last else "#F3F4F8")
    S.text(axb, xs_c[-1] + r + 0.12, yc + r - 0.05, r"$\ell_{\max}$", c=S.INK2, fontsize=5.6, va="center")
    S.text(axb, xs_c[-1] + r + 0.12, yc - r + 0.05, r"$\ell_{\min}$", c=S.INK2, fontsize=5.6, va="center")
    labs = [r"$x_{t-3}$", r"$x_{t-2}$", r"$x_{t-1}$", r"$x_t$"]
    for cx, s in zip(xs_c, labs):
        S.text(axb, cx, yc - r - 0.22, s, c=S.INK, ha="center", va="top", fontsize=6.6)
    # one arc per lag, each landing on its own point of the last sphere, so that they never cross
    y_ = yc + r + 0.08
    for src, x_end, lab in [(2, xs_c[3] - 0.55, r"$J_1$"), (1, xs_c[3] - 0.12, r"$J_2$"), (0, xs_c[3] + 0.32, r"$J_3$")]:
        span = x_end - xs_c[src]
        h_ = 0.42 + 0.17 * span                       # apex height grows with the lag
        rad = -2 * h_ / span
        axb.add_patch(FancyArrowPatch((xs_c[src], y_), (x_end, y_ - 0.02),
                                      connectionstyle=f"arc3,rad={rad:.3f}", arrowstyle="-|>", mutation_scale=5.5,
                                      color=S.INDIGO, lw=0.8, shrinkA=0.5, shrinkB=0.5, zorder=5))
        S.text(axb, xs_c[src] - 0.12, y_ + 0.30, lab, c=S.INDIGO, ha="right", va="center", fontsize=6.5)
    axb.add_patch(FancyArrowPatch((xs_c[3], 0.15), (xs_c[3], yc - r - 0.95), arrowstyle="-|>",
                                  mutation_scale=5.5, color=S.TEAL, lw=1.1, shrinkA=0, shrinkB=0))
    S.text(axb, xs_c[3] + 0.22, 0.55, r"$h_i$", c=S.TEAL, ha="left", va="center", fontsize=6.8)
    S.text(axb, 0.05, 0.85, r"$p(x_t\,|\,\mathrm{past})\propto e^{\theta_t x_t}$", c=S.INK, ha="left",
           va="center", fontsize=6.3)
    S.text(axb, 0.05, -0.45, r"$\theta_t=h_i+\sum_{\tau}J_\tau\,x_{t-\tau}$", c=S.INK, ha="left",
           va="center", fontsize=6.3)

    # ------------------------------------------------ (c) phase diagram with data
    axc = f.add_subplot(gs[:, 1])
    Lbg = float(np.log(100.0))
    mus = np.linspace(1.0, 3.2, 111); bes = np.linspace(-0.6, 4.6, 131)
    cache = os.path.join(DATA, "phase_map_L100.npy")
    if os.path.exists(cache):
        Pi = np.load(cache)
    else:
        Pi = np.array([[order_parameter(m, b, Lbg, n=801) for m in mus] for b in bes])
        np.save(cache, Pi)
    im = axc.imshow(Pi, origin="lower", extent=[mus[0], mus[-1], bes[0], bes[-1]], aspect="auto",
                    cmap=PHASE, vmin=0, vmax=1, interpolation="bilinear", zorder=0, rasterized=True)
    mm = np.linspace(1.0, 3.2, 10)
    axc.plot(mm, (mm - 1), color=S.INK2, lw=0.8, ls=(0, (1.3, 1.3)), zorder=2)
    axc.plot(mm, 2 * (mm - 1), color=S.INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
    axc.axhline(0, color=S.INK2, lw=0.6, zorder=2)
    def along(x_, y_, s_, slope, **kw):
        p1 = axc.transData.transform((x_, y_)); p2 = axc.transData.transform((x_ + 0.1, y_ + 0.1 * slope))
        ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        S.text(axc, x_, y_, s_, rotation=ang, rotation_mode="anchor", **kw)
    axc.set_xlim(1.0, 3.2); axc.set_ylim(-0.6, 3.6)
    along(2.93, 1.80, r"spinodal  $J_{\rm up}$", 1.0, c=S.INK2, ha="right", va="top", fontsize=5.9)
    along(2.36, 2.79, r"condensation  $J_{\rm cond}=2J_{\rm up}$", 2.0, c="white", ha="right", va="bottom",
          fontsize=5.9)
    S.text(axc, 1.06, 3.50, r"condensed at $\ell_{\max}$", c="white", ha="left", va="top", fontsize=6.4,
           fontweight="bold")
    S.text(axc, 3.16, 0.30, "Pareto phase", c=S.INK2, ha="right", va="bottom", fontsize=6.2)
    S.text(axc, 3.16, -0.06, r"renewal ($J=0$)", c=S.INK2, ha="right", va="top", fontsize=5.9)
    T = dat("termites_K1.json")
    tm = np.array([t["mu_bare"] for t in T]); tb = np.array([t["J1"] * t["L"] for t in T])
    axc.scatter(tm, tb, s=8, color=S.OCHRE, edgecolor="white", lw=0.35, zorder=4)
    S.text(axc, 3.17, 2.45, "termites", c=S.OCHRE, ha="right", va="bottom", fontsize=6.4, fontweight="bold",
           zorder=9)
    for q in dat("larvae_K1.json"):
        mb = np.array(q["mu_bare"]); m0 = np.median(mb); lo, hi = np.percentile(mb, [25, 75])
        b0 = q["J1"] * q["L"]; db = 2 * q["seJ1"] * q["L"]
        g = GROUP[q["cond"]]
        axc.plot([lo, hi], [b0, b0], color=S.INDIGO, lw=0.65, alpha=0.7, zorder=5)
        axc.plot([m0, m0], [b0 - db, b0 + db], color=S.INDIGO, lw=0.65, alpha=0.7, zorder=5)
        axc.plot(m0, b0, marker=GMARK[g], ms=4.3 if g != "ablation" else 4.8, color=S.INDIGO, mec="white",
                 mew=0.6, zorder=6)
    S.text(axc, 1.30, -0.47, "larvae", c=S.INDIGO, ha="center", va="bottom", fontsize=6.4, fontweight="bold",
           zorder=9)
    S.text(axc, 2.86, 1.16, "ablated", c=S.INDIGO, ha="left", va="top", fontsize=5.8, zorder=9)
    for s_, b_ in zip("ABC", [0.5, 1.5, 2.5]):
        axc.plot(2.0, b_, marker="o", ms=7.0, mfc="white", mec=S.INK, mew=0.7, zorder=7)
        S.text(axc, 2.0, b_ - 0.01, s_, c=S.INK, ha="center", va="center", fontsize=5.5, fontweight="bold",
               zorder=8)
    axc.set_xlabel(r"bare exponent  $\mu_{\rm bare}$")
    axc.set_ylabel(r"memory coupling  $J_1 L$")
    axc.set_xticks([1, 1.5, 2, 2.5, 3])
    cax = axc.inset_axes([1.025, 0.62, 0.028, 0.36])
    cb = f.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([0, 0.5, 1])
    cax.tick_params(labelsize=5.4, length=1.4, pad=1.0, color=S.AXIS)
    cax.set_title(r"$\Pi$", fontsize=6.4, color=S.INK2, pad=2.0, loc="left")

    # ------------------------------------------------ (d) what a walker looks like in each phase
    axd = f.add_subplot(gs[0, 2])
    Ld = float(np.log(1e3))
    greys = ["#9C9A94", "#5E5D58", S.INK]
    curves = []
    for b_, c_ in zip([0.5, 1.5, 2.5], greys):
        x, p = stationary(2.0, b_, Ld, n=6001)
        cc = ccdf_of(x, p)
        axd.plot(np.exp(x), cc, color=c_, lw=1.2)
        curves.append((np.exp(x), cc))
    x, p = stationary(2.0, 0.0, Ld, n=6001)
    axd.plot(np.exp(x), ccdf_of(x, p), color=S.MUTED, lw=0.7, ls=(0, (2.5, 1.5)))
    axd.set_xscale("log"); axd.set_yscale("log")
    axd.set_ylim(2e-4, 1.6); axd.set_xlim(1, 1.6e3)
    axd.set_xticks([1, 10, 100, 1000]); axd.set_xticklabels(["1", "10", r"$10^2$", r"$10^3$"])
    axd.set_yticks([1e-3, 1e-2, 1e-1, 1]); axd.set_yticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"])
    axd.minorticks_off()
    axd.set_xlabel(r"$\ell/\ell_{\min}$", labelpad=1.5)
    axd.set_ylabel(r"$P(\ell_t>\ell)$")
    for (xx, yy), s_, c_ in zip(curves, "ABC", greys):
        k = np.searchsorted(xx, 120)
        S.text(axd, 120 * 1.12, yy[k] * (1.5 if s_ != "C" else 1.35), s_, c=c_, fontsize=6.2, fontweight="bold",
               ha="left", va="bottom")
    S.text(axd, 1.25, 3.2e-4, r"dashed: renewal, $J=0$", c=S.MUTED, fontsize=5.6)
    S.despine(axd)

    # ------------------------------------------------ (e) free-energy picture of condensation
    axe = f.add_subplot(gs[1, 2])
    uu = np.linspace(0, 1, 401)
    axe.set_xlim(0, 1); axe.set_ylim(-0.62, 1.05)
    for s_, b_, c_, ls_ in [("A", 0.5, greys[0], "-"), ("B", 1.5, greys[1], "-"), ("", 2.0, S.INK2, (0, (4, 2))),
                            ("C", 2.5, greys[2], "-")]:
        F = uu - np.maximum(b_ * uu - 1.0, 0.0)          # F(u) = -f(u) with mu_bare - 1 = 1
        axe.plot(uu, F, color=c_, lw=1.2 if s_ else 0.9, ls=ls_)
    S.text(axe, 0.30, 0.42, "A", c=greys[0], fontsize=6.2, fontweight="bold")
    S.text(axe, 0.86, 0.66, "B", c=greys[1], fontsize=6.2, fontweight="bold")
    S.text(axe, 0.86, -0.20, "C", c=greys[2], fontsize=6.2, fontweight="bold")
    p1 = axe.transData.transform((0.6, 0.4)); p2 = axe.transData.transform((0.9, 0.1))
    ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    S.text(axe, 0.80, 0.235, "coexistence", c=S.INK2, fontsize=5.6, ha="center", va="bottom", rotation=ang,
           rotation_mode="anchor")
    axe.set_xticks([0, 0.5, 1]); axe.set_xticklabels([r"$\ell_{\min}$", r"$\sqrt{\ell_{\min}\ell_{\max}}$",
                                                      r"$\ell_{\max}$"])
    axe.set_yticks([-0.5, 0, 0.5, 1])
    axe.axhline(0, color=S.GRID, lw=0.6, zorder=0)
    axe.set_xlabel(r"$\ell$  (log scale)", labelpad=1.5)
    axe.set_ylabel(r"$F=-\lim\ L^{-1}\ln\pi$")
    S.despine(axe)
    S.despine(axc)
    S.heads(f, [(axa, "(a)", r"a $\mathit{Drosophila}$ larva, cut into flights"),
                (axb, "(b)", r"flights as spins:  $x=\ln(\ell/\ell_{\min})\in[0,L]$"),
                (axc, "(c)", "phase diagram of the memory chain"),
                (axd, "(d)", r"stationary law at A, B, C"),
                (axe, "(e)", r"free energy, $L\to\infty$")])
    S.save(f, "fig1_overview")


if __name__ == "__main__":
    make()
