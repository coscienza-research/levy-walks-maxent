"""Fig. A5 -- collective controls on synthetic groups: drives common to all units read as interactions unless time
fields are fitted, and critical fluctuations of the collective speed, with and without own memory."""
from common import S
import panels as P


def make():
    f = S.fig(S.FULL, 50)
    gs = f.add_gridspec(1, 3, width_ratios=[0.85, 0.5, 1.15], left=0.07, right=0.985, top=0.86, bottom=0.2,
                        wspace=0.6)
    axs = [f.add_subplot(gs[0, k]) for k in range(3)]
    P.common_drives(axs[0], axs[1]); P.mf_fluctuations(axs[2])
    S.heads(f, [(axs[0], "(a)", "common drives"), (axs[2], "(b)", "critical fluctuations")], one_row=True)
    S.save(f, "figA5_collective")


if __name__ == "__main__":
    make()
