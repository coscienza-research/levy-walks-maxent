"""make_figs.py -- rebuild every figure of the paper.

    python make_figs.py               # all
    python make_figs.py fig1 figA4    # some
"""
import importlib
import sys

ALL = ["fig1", "fig2", "fig3", "fig4", "figA1", "figA2", "figA3", "figA4", "figA5"]

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        print(name)
        importlib.import_module(name).make()
