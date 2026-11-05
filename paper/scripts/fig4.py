"""Fig. 4 -- real groups read with the self + social model: delayed alignment of baboons with their nearest
neighbors, its decay with rank and its asymmetry; the flock-wide coupling of homing pigeons; and the depth that
a group-average analysis declares, equal on the data and on time-shifted surrogates."""
from common import S, res
import panels as P


def make():
    B = res("realgroups_baboons10.json")
    Pg = res("realgroups_pigeons.json")
    f = S.fig(S.FULL, 96)
    gs = f.add_gridspec(2, 3, left=0.07, right=0.975, top=0.93, bottom=0.1, wspace=0.42, hspace=0.55)
    axs = [f.add_subplot(gs[r, k]) for r in range(2) for k in range(3)]
    tau, J, Jg, m = P.lags_panel(axs[0], B, 10)
    S.text(axs[0], 33, 0.565, "global, no time fields", c=S.CORAL, fontsize=5.6, ha="left")
    S.text(axs[0], 35, 0.27, "3 nearest, time fields", c=S.PLUM, fontsize=5.6, ha="left")
    S.text(axs[0], 60, 0.035, "time-shifted", c=S.INK2, fontsize=5.6, ha="right")
    P.rank_panel(axs[1], B, B["spec"]["ranks"])
    axs[1].set_xscale("log"); axs[1].set_xticks([1, 2, 3, 6, 12]); axs[1].set_xticklabels(["1", "2", "3", "6", "12"])
    axs[1].minorticks_off()
    S.text(axs[1], 1.55, -0.024, "time-shifted", c=S.INK2, fontsize=5.6, ha="left")
    P.ahead_panel(axs[2], B, 10)
    S.text(axs[2], 10, 0.075, "time-shifted", c=S.INK2, fontsize=5.6, ha="left")
    tau, J, Jg, m = P.lags_panel(axs[3], Pg, 1, show_global=False)
    S.text(axs[3], 1.25, J[0] + 1, "nearest, time fields", c=S.PLUM, fontsize=5.6, ha="left")
    S.text(axs[3], 6, m[-1] - 6, "time-shifted", c=S.INK2, fontsize=5.6, ha="right", va="top")
    P.rank_norm_panel(axs[4], Pg, Pg["spec"]["ranks"])
    P.group_average_depth(axs[5])
    S.heads(f, [(a, f"({chr(97 + k)})", t) for k, (a, t) in enumerate(zip(axs, [
        "baboons: delayed alignment", "baboons: decay with rank", "baboons: ahead vs behind",
        "pigeons: delayed alignment", "pigeons: flock-wide", "group average: depth"]))])
    S.save(f, "fig4_groups")


if __name__ == "__main__":
    make()
