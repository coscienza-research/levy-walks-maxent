"""Fig. 2 -- many walkers: one lagged correlation contains two memories; the group average reads own memory as a
delayed interaction; the mean-field phase diagram of the collective speed and its three regimes in time."""
from common import S
import panels as P


def make():
    f = S.fig(S.FULL, 50)
    gs = f.add_gridspec(1, 4, width_ratios=[1.3, 1, 1, 0.95], left=0.01, right=0.985, top=0.86, bottom=0.2,
                        wspace=0.5)
    axA = f.add_subplot(gs[0, 0]); P.self_social_schematic(axA)
    axB = f.add_subplot(gs[0, 1]); P.aggregation(axB)
    axC = f.add_subplot(gs[0, 2]); P.mf_phase(axC)
    axD = P.mf_series(f, gs[0, 3])
    S.heads(f, [(axA, "(a)", "own and others' past"), (axB, "(b)", "group average, no interaction"),
                (axC, "(c)", "collective speed"), (axD[0], "(d)", "three regimes")], one_row=True)
    S.save(f, "fig2_many")


if __name__ == "__main__":
    make()
