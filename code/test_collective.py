"""Correctness tests for the collective fitter, simulator, and exact mean-field solver."""
import numpy as np
import collective as co
from collective import Group, build_design, fit_collective, pre_pipeline
from simulate_collective import speed_vicsek, group_from_positions
from maxent_levy import fit as fit_single, build_design as bd_single, GLQuad, stationary_marginal_K1
from meanfield import unit_response, critical_coupling_self, nvar_linear_response


def nll_at(D, f, bshift, L):
    """Recompute NLL with shared coefficients shifted, FE fixed (for a numeric gradient)."""
    cols = list(range(f.Ks)) + list(range(D.Ks_max, D.Ks_max + f.Kc))
    X = D.Xd[:, cols]
    b = f.b + bshift
    th = X @ b[:X.shape[1]] + f.fe_unit[D.unit] + (f.fe_time[D.time] if f.time_fe else 0)
    beta = b[-1] if f.family == "lognormal" else 0.0
    if f.family == "powerlaw":
        lz = co.pl_logZ(th, L)
    else:
        lz = GLQuad(L).stats(th, beta)[0]
    return float(np.sum(-th * D.y + beta * D.y**2 + lz))


def test_collective_derivatives_and_optimum():
    """Verify stationarity of the shared-block gradient at the converged collective MLE."""
    rng = np.random.default_rng(1)
    sim = speed_vicsek(12, 120, rng, a=[0.3], Jloc=[0.2], h=rng.uniform(-1.5, -0.5, 12))
    g, L = group_from_positions(sim)
    D = build_design([g], 2, 2, mode="local")
    for fam in ["powerlaw", "lognormal"]:
        for ufe, tfe in [(True, False), (True, True), (False, False)]:
            f = fit_collective(D, 2, 2, family=fam, unit_fe=ufe, time_fe=tfe, L=L, maxiter=3)
            print(f"{fam:9s} unit_fe={ufe!s:5s} time_fe={tfe!s:5s} converged(3 it)={f.converged} nll={f.nll:.4f}")

    for fam in ["powerlaw", "lognormal"]:
        f = fit_collective(D, 2, 2, family=fam, unit_fe=True, time_fe=True, L=L)
        eps = 1e-5
        gnum = [(nll_at(D, f, eps * e, L) - nll_at(D, f, -eps * e, L)) / (2 * eps) for e in np.eye(len(f.b))]
        dnll = abs(nll_at(D, f, 0 * f.b, L) - f.nll)
        print(f"{fam}: |numeric gradient at optimum| max = {np.max(np.abs(gnum)):.2e}; nll check {dnll:.2e}")
        assert f.converged
        assert np.max(np.abs(gnum)) < 1e-3
        assert dnll < 1e-8


def test_equivalence_with_single_unit_fitter():
    """Verify that the collective fitter with Kc=0 and no unit FE matches the single-unit fitter."""
    rng = np.random.default_rng(1)
    sim = speed_vicsek(12, 120, rng, a=[0.3], Jloc=[0.2], h=rng.uniform(-1.5, -0.5, 12))
    _, L = group_from_positions(sim)
    x = np.array([sim["X_true"][0]])
    gg = Group(X=x)
    Dg = build_design([gg], 3, 0, mode="global", c=2.0)
    fc = fit_collective(Dg, 3, 0, family="powerlaw", unit_fe=False, L=L)
    y, Z = bd_single([x[0]], 3, 2.0)
    fs = fit_single(y, Z, 3, L, 2.0)
    print("single vs collective: params", np.round(fs.params, 6), np.round(fc.b, 6), " dNLL", fs.nll - fc.nll)
    assert np.max(np.abs(fs.params - fc.b)) < 1e-6
    assert abs(fs.nll - fc.nll) < 1e-6


def test_collective_delay_recovery():
    """Verify recovery of self memory and delayed local social coupling (delay tau* = 3)."""
    rng = np.random.default_rng(7)
    sim = speed_vicsek(40, 800, rng, L=np.log(100), a=[0.25, 0.1], Jloc=[0, 0, 0.35], h=rng.uniform(-1.2, -0.4, 40))
    g, L = group_from_positions(sim)
    D = build_design([g], 3, 5, mode="local")
    fits = co.scan(D, range(0, 4), range(0, 6), family="powerlaw", unit_fe=True, L=L)
    kb, fb = co.best(fits, "bic")
    print("selected (Ks,Kc) by BIC:", kb, " a=", np.round(fb.a, 3), "±", np.round(fb.se_a, 3),
          " J=", np.round(fb.J, 3), "±", np.round(fb.se_J, 3))
    out, tsel, delay = pre_pipeline([g], L, 6)
    print("PRE pipeline: tau_M =", tsel, "delay(argmax|J|) =", delay, " J =", np.round(out[tsel]["J"], 3))
    assert kb == (2, 3)
    assert int(np.argmax(np.abs(fb.J))) + 1 == 3
    assert abs(fb.J[2] - 0.35) < 3.0 * fb.se_J[2]


def test_exact_mean_field_with_self_memory():
    """Verify exact mean-field susceptibility, critical coupling J_c(a), and NVAR linear response."""
    def k1_mean(h, a, L):
        x, p = stationary_marginal_K1(h, a, L, 0.0, n=40001)
        return np.trapezoid(x * p, x)

    Lm = np.log(100.0)
    assert abs(critical_coupling_self(0.0, Lm) - 12 / Lm**2) < 1e-6
    for a in [0.1, 0.2, 0.4]:
        h0 = -a * Lm / 2
        R, C, m = unit_response(h0, a, Lm)
        chi_fd = (k1_mean(h0 + 1e-4, a, Lm) - k1_mean(h0 - 1e-4, a, Lm)) / 2e-4
        assert abs(m - Lm / 2) < 1e-6 and np.all(R > -1e-12) and abs(R.sum() - chi_fd) < 1e-3 * chi_fd
        print(f"a={a}: m*=L/2, chi=sum R={R.sum():.4f} (dPsi/dh {chi_fd:.4f}), sum C={C[0] + 2 * C[1:].sum():.4f}, "
              f"J_c(a)/J_c(0)={critical_coupling_self(a, Lm) * Lm**2 / 12:.3f}")
    R0, C0, _ = unit_response(0.0, 0.0, Lm)
    lp = Lm**2 / 12
    assert abs(nvar_linear_response(0.5 / lp, R0, C0) - lp / (1 - 0.25)) < 1e-3 * lp
    print("exact mean field with self memory: OK")


if __name__ == "__main__":
    test_collective_derivatives_and_optimum()
    test_equivalence_with_single_unit_fitter()
    test_collective_delay_recovery()
    test_exact_mean_field_with_self_memory()
