"""Fig. A5 -- collective controls on synthetic groups: drives common to all units read as interactions unless time
fields are fitted, and critical fluctuations of the collective speed, with and without own memory."""
from common import S
import panels as P


def make():
    f = S.fig(S.FULL, 52)
    gs = f.add_gridspec(1, 3, width_ratios=[0.95, 0.80, 1.25], left=0.07, right=0.985, top=0.86, bottom=0.22,
                        wspace=0.44)
    axs = [f.add_subplot(gs[0, k]) for k in range(3)]
    P.common_drives(axs[0], axs[1]); P.mf_fluctuations(axs[2])
    S.heads(f, [(axs[0], "(a)", "drive without interaction"),
                (axs[1], "(b)", "planted coupling"),
                (axs[2], "(c)", "critical fluctuations")], one_row=True)
    S.save(f, "figA5_collective")


if __name__ == "__main__":
    make()
