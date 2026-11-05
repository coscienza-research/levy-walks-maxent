"""Fig. A4 -- controls on real walkers: larval couplings under other segmentations and with a field per block,
termite couplings with and without a wall field, and the depth that heterogeneity invents in T cells."""
from common import S
import panels as P


def make():
    f = S.fig(S.FULL, 60)
    gs = f.add_gridspec(1, 3, width_ratios=[1.45, 0.82, 1.05], left=0.085, right=0.985, top=0.84, bottom=0.2,
                        wspace=0.55)
    axs = [f.add_subplot(gs[0, k]) for k in range(3)]
    P.larvae_robustness(axs[0]); P.termite_wall(axs[1]); P.tcell_shuffle(axs[2])
    S.heads(f, [(axs[0], "(a)", "larvae: robustness"), (axs[1], "(b)", "termites: wall field"),
                (axs[2], "(c)", "T cells: heterogeneity")], one_row=True)
    S.save(f, "figA4_controls")


if __name__ == "__main__":
    make()
