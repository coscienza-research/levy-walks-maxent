"""Fig. 3 -- single walkers in the data: larval couplings per condition, depth and kernel of termite memory, and
the dressing of the histogram exponent by memory in both species."""
from common import S
import panels as P


def make():
    f = S.fig(S.FULL, 54)
    gs = f.add_gridspec(1, 4, width_ratios=[1.12, 0.9, 0.9, 0.95], left=0.105, right=0.99, top=0.86, bottom=0.2,
                        wspace=0.55)
    axs = [f.add_subplot(gs[0, k]) for k in range(4)]
    P.larvae_coupling(axs[0]); P.termite_depth(axs[1]); P.termite_kernel(axs[2]); P.dressing_scatter(axs[3])
    S.heads(f, [(axs[0], "(a)", "larvae"), (axs[1], "(b)", "termites: depth"), (axs[2], "(c)", "termites: kernel"),
                (axs[3], "(d)", "dressed exponent")], one_row=True)
    S.save(f, "fig3_walkers")


if __name__ == "__main__":
    make()
