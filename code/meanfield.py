"""Mean-field theory of the collective modulus model (global social coupling).

Kinetic model: theta_i^t = H + sum_tau a_tau x_i^{t-tau} + sum_tau J_tau m_{t-tau},
x_i^t ~ e^{theta x} on [0, L], m_t = (1/N) sum_i x_i^t.
For N -> infinity and a = 0:   m_t = Lambda(H + sum_tau J_tau m_{t-tau}),
Lambda(theta) = E_theta[x] = (L/2)[1 + Langevin(theta L / 2)],  Lambda' = Var_theta[x] <= L^2/12.
With lag-1 self memory a >= 0 (exact McKean-Vlasov limit): at a stationary state every unit is
an independent K=1 chain with field h_eff = H + J_tot m*, whose stationary law is
pi(x) ~ e^{h_eff x} Z(h_eff + a x); instability at J_tot chi = 1 with chi = sum_s R(s) = dE_pi[x]/dh.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from maxent_levy import pl_mean_var, pl_sample


def Lam(theta, L):
    return pl_mean_var(np.atleast_1d(np.asarray(theta, dtype=float)), L)[0]


def dLam(theta, L):
    return pl_mean_var(np.atleast_1d(np.asarray(theta, dtype=float)), L)[1]


def critical_point(L):
    """Cusp of the stationary equation m = Lambda(H + J m): J_c = 12/L^2, H_c = -6/L, m_c = L/2."""
    return 12.0 / L**2, -6.0 / L, L / 2


def fixed_points(H, Jtot, L, n=4001):
    m = np.linspace(1e-9, L - 1e-9, n)
    f = Lam(H + Jtot * m, L) - m
    roots = []
    for k in np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]:
        r = brentq(lambda z: Lam(H + Jtot * z, L)[0] - z, m[k], m[k + 1])
        if not roots or abs(r - roots[-1]) > 1e-6:
            roots.append(r)
    return np.array(roots)


def spinodal(L, thetas=None):
    """Spinodal lines in the (J, H) plane: J = 1/Lambda'(theta), H = theta - J Lambda(theta)."""
    if thetas is None:
        thetas = np.linspace(-12 / L, 12 / L, 2001)
    J = 1.0 / dLam(thetas, L)
    H = thetas - J * Lam(thetas, L)
    return J, H, thetas


def stable_fixed_point_theta(H, J, L):
    """theta* of the unique fixed point when J <= 0 (monotone decreasing map)."""
    m = fixed_points(H, J, L)
    return H + J * m[0]


def simulate_mf(N, T, H, J, L, rng, a=(), burn=500, x0=None):
    """Finite-N kinetic simulation with global coupling (full mean) and optional self memory.
    J, a: arrays indexed by lag (J[0] = lag 1). Returns m_t (length T)."""
    J = np.atleast_1d(np.asarray(J, dtype=float))
    a = np.atleast_1d(np.asarray(a, dtype=float)) if len(np.atleast_1d(a)) else np.zeros(0)
    K = max(len(J), len(a), 1)
    tot = T + burn
    X = np.empty((K + 1, N))                        # ring buffer of the last K frames
    hist_m = np.zeros(tot + K)
    xs = np.full((K, N), L / 2 if x0 is None else x0)
    ms = list(np.full(K, xs[0].mean()))
    out = np.empty(tot)
    for t in range(tot):
        theta = np.full(N, H)
        for tau in range(1, len(J) + 1):
            theta = theta + J[tau - 1] * ms[-tau]
        for tau in range(1, len(a) + 1):
            theta = theta + a[tau - 1] * xs[-tau]
        x = pl_sample(theta, L, rng)
        xs = np.vstack([xs[1:], x[None, :]]) if K > 0 else x[None, :]
        ms.append(x.mean())
        ms = ms[-K:]
        out[t] = x.mean()
    return out[burn:]


def unit_chain(h, a, L, n=400):
    """Single unit at fixed field h with lag-1 self coupling a, theta(x) = h + a x, on n
    Gauss-Legendre nodes. Returns nodes x, transition matrix P (row = previous step),
    stationary law pi and conditional mean lam(x) = E[x_t | x_{t-1} = x]."""
    u, w = leggauss(n)
    x = (u + 1) * L / 2
    logK = (h + a * x)[:, None] * x[None, :] + np.log(w * L / 2)[None, :]
    P = np.exp(logK - logK.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    ev, V = np.linalg.eig(P.T)
    pi = np.real(V[:, np.argmin(np.abs(ev - 1))])
    return x, P, pi / pi.sum(), P @ x


def unit_response(h, a, L, smax=400, n=400):
    """Response R(s) = Cov(x_s, eps_0) of one unit to a field pulse at time 0 (eps_0 = x_0 - E[x_0 | past],
    the innovation) and autocovariance C(s). sum_s R(s) = dE_pi[x]/dh (causal fluctuation-response
    relation); it differs from sum_s C(s) unless a = 0."""
    x, P, pi, lam = unit_chain(h, a, L, n)
    m = pi @ x
    w0 = (pi[:, None] * P * (x[None, :] - lam[:, None])).sum(axis=0)    # weight of x_0 times innovation
    R, C = np.zeros(smax), np.zeros(smax)
    f = x.copy()                                                          # (P^s x)(x_0)
    for s in range(smax):
        R[s] = w0 @ f
        C[s] = pi @ ((x - m) * f)
        f = P @ f
    return R, C, m


def critical_coupling_self(a, L):
    """Exact N -> infinity threshold for lag-1 social coupling with lag-1 self memory a >= 0, on the
    symmetry line H + (J + a) L / 2 = 0 (m* = L/2, h_eff = -a L / 2): J_c(a) = 1 / chi."""
    R, _, _ = unit_response(-a * L / 2, a, L)
    return 1.0 / R.sum()


def nvar_linear_response(J, R, C, nw=8192):
    """Leading order in 1/N for lag-1 social coupling J:
    N Var(m) = (1/2pi) int S_x(w) / |1 - J e^{-iw} R(w)|^2 dw, S_x the single-unit spectrum."""
    w = np.linspace(-np.pi, np.pi, nw, endpoint=False)
    E = np.exp(-1j * np.outer(w, np.arange(len(R))))
    Sx = C[0] + 2 * np.real(E[:, 1:] @ C[1:])
    return float(np.mean(Sx / np.abs(1 - J * np.exp(-1j * w) * (E @ R)) ** 2))


def iterate_map(H, J, L, T=400, m0=None):
    """Deterministic N=infinity delayed map."""
    J = np.atleast_1d(np.asarray(J, dtype=float))
    K = len(J)
    m = list(np.full(K, L / 2 if m0 is None else m0))
    for _ in range(T):
        th = H + sum(J[tau - 1] * m[-tau] for tau in range(1, K + 1))
        m.append(Lam(th, L)[0])
    return np.array(m[K:])
