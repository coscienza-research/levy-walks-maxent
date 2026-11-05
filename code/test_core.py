"""Consistency and numerical verification tests for the maxent_levy library."""
import numpy as np
from scipy.stats import truncnorm
from maxent_levy import (pl_logZ, pl_mean_var, pl_sample, GLQuad, build_design, fit, _terms,
                         simulate_model, stationary_marginal_K1, scan_orders, select, langevin)

L_DEFAULT = np.log(1e3)


def test_closed_forms_vs_quadrature():
    """Verify exact closed-form log-partition, mean, variance, and Heisenberg Langevin identity."""
    L = L_DEFAULT
    q = GLQuad(L, n=96)
    th = np.array([-30, -5, -1, -1e-6, 0.0, 1e-6, 0.3, 2, 8]) / 1.0
    lz, m1, _, vx, _, _ = q.stats(th, 0.0)
    lz_cf = pl_logZ(th, L)
    m_cf, v_cf = pl_mean_var(th, L)
    print("max |ΔlogZ|  :", np.max(np.abs(lz - lz_cf)))
    print("max |Δmean|  :", np.max(np.abs(m1 - m_cf)))
    print("max |Δvar|   :", np.max(np.abs(vx - v_cf)))
    assert np.max(np.abs(lz - lz_cf)) < 1e-11
    assert np.max(np.abs(m1 - m_cf)) < 1e-11
    assert np.max(np.abs(vx - v_cf)) < 1e-10

    hh = 2.7
    z = np.linspace(-1, 1, 200001)
    w = np.exp(hh * z)
    heis_num = np.trapezoid(z * w, z) / np.trapezoid(w, z)
    heis_cf = langevin(np.array([hh]))[0]
    print("Heisenberg <z> num vs L(h):", heis_num, heis_cf)
    assert abs(heis_num - heis_cf) < 1e-10


def test_exact_sampling():
    """Verify exact inverse-CDF sampling moments against analytical expressions."""
    rng = np.random.default_rng(0)
    L = L_DEFAULT
    for t0 in [-2.0, -0.2, 0.0, 0.5]:
        s = pl_sample(np.full(250000, t0), L, rng)
        m, v = pl_mean_var(np.array([t0]), L)
        print(f"θ={t0:+.1f}  mean sim {s.mean():.4f} vs {m[0]:.4f}   var sim {s.var():.4f} vs {v[0]:.4f}")
        assert abs(s.mean() - m[0]) < 0.015
        assert abs(s.var() - v[0]) < 0.03


def test_gradient_and_hessian_finite_differences():
    """Compare analytical gradient and Hessian against central finite differences."""
    rng = np.random.default_rng(2)
    L = L_DEFAULT
    q = GLQuad(L, n=96)
    X = simulate_model(-0.8, [0.2, 0.1], L, L / 3, 2000, rng)[0]
    y, Z = build_design([X], 2, X.mean())
    for fam, p in [("powerlaw", np.array([-0.7, 0.15, 0.05])), ("lognormal", np.array([-0.7, 0.15, 0.05, 0.02]))]:
        K = 2
        _, g, H, _ = _terms(p, y, Z, K, fam, L, q)
        eps = 1e-6
        gn = np.array([(_terms(p + eps * e, y, Z, K, fam, L, q)[0] - _terms(p - eps * e, y, Z, K, fam, L, q)[0]) / (2 * eps)
                       for e in np.eye(len(p))])
        Hn = np.array([(_terms(p + eps * e, y, Z, K, fam, L, q)[1] - _terms(p - eps * e, y, Z, K, fam, L, q)[1]) / (2 * eps)
                       for e in np.eye(len(p))])
        rel_g = np.max(np.abs(g - gn)) / np.max(np.abs(gn))
        rel_H = np.max(np.abs(H - Hn)) / np.max(np.abs(Hn))
        print(f"{fam:9s} rel.err grad {rel_g:.2e}  Hess {rel_H:.2e}")
        assert rel_g < 1e-7
        assert rel_H < 1e-7


def test_parameter_and_order_recovery():
    """Verify exact recovery of true order K*=3 and coupling parameters."""
    rng = np.random.default_rng(3)
    L = L_DEFAULT
    h_true, J_true = -0.9, np.array([0.25, 0.10, 0.06])
    c0 = 2.0
    X = simulate_model(h_true, J_true, L, c0, 20000, rng)[0]
    fits = scan_orders([X], L, 8, c=c0)
    for crit in ["aic", "bic", "tic"]:
        k_sel = select(fits, crit)[0]
        print(crit, "→ K* =", k_sel)
    assert select(fits, "bic")[0] == 3
    r = fits[3]
    print("estimated h, J:", np.round(r.params, 4), "±", np.round(r.se, 4), " true:", h_true, J_true)
    print("convergence:", all(f.converged for f in fits), [f.n_iter for f in fits])
    assert all(f.converged for f in fits)
    assert abs(r.h - h_true) < 3.5 * r.se[0]
    assert np.all(np.abs(r.J - J_true) < 3.5 * r.se[1:])


def test_stationary_marginal_K1():
    """Verify the closed-form K=1 stationary marginal density against direct simulation."""
    rng = np.random.default_rng(4)
    L = L_DEFAULT
    h1, J1, c1 = -0.6, 0.35, 0.0
    X = simulate_model(h1, [J1], L, c1, 250000, rng)[0]
    xg, pi = stationary_marginal_K1(h1, J1, L, c1)
    hist, edges = np.histogram(X, bins=40, range=(0, L), density=True)
    mid = (edges[1:] + edges[:-1]) / 2
    rel_err = np.max(np.abs(hist - np.interp(mid, xg, pi))) / pi.max()
    print("K=1 marginal: max |hist - π| / max π =", rel_err)
    assert rel_err < 0.03


def test_lognormal_family_recovery():
    """Verify recovery of quadratic stiffness β from i.i.d. truncated log-normal samples."""
    L = L_DEFAULT
    mu_ln, sd_ln = 2.5, 1.2
    a, b = (0 - mu_ln) / sd_ln, (L - mu_ln) / sd_ln
    xs = truncnorm.rvs(a, b, loc=mu_ln, scale=sd_ln, size=40000, random_state=1)
    fl = scan_orders([xs], L, 0, family="lognormal")[0]
    fp = scan_orders([xs], L, 0, family="powerlaw")[0]
    beta_true = 1 / (2 * sd_ln**2)
    print("estimated β", fl.beta, " true", beta_true, "| ΔAIC (powerlaw - lognormal):", fp.aic - fl.aic)
    assert abs(fl.beta - beta_true) < 0.03
    assert fp.aic - fl.aic > 100.0


if __name__ == "__main__":
    test_closed_forms_vs_quadrature()
    test_exact_sampling()
    test_gradient_and_hessian_finite_differences()
    test_parameter_and_order_recovery()
    test_stationary_marginal_K1()
    test_lognormal_family_recovery()
