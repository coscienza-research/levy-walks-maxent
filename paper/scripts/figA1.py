"""Fig. A1 -- the memory chain at depth one: single-flight Langevin response, crossing of the order parameter at
the self-dual point, and the exponent that a renewal fit reads off the stationary histogram."""
from common import S
import panels as P


def make():
    f = S.fig(S.FULL, 52)
    gs = f.add_gridspec(1, 3, left=0.06, right=0.985, top=0.86, bottom=0.21, wspace=0.42)
    axs = [f.add_subplot(gs[0, k]) for k in range(3)]
    P.langevin_response(axs[0]); P.crossing(axs[1]); P.dressed_exponent(axs[2])
    S.heads(f, [(axs[0], "(a)", "one flight"), (axs[1], "(b)", "condensation"),
                (axs[2], "(c)", "dressed exponent")], one_row=True)
    S.save(f, "figA1_theory")


if __name__ == "__main__":
    make()
