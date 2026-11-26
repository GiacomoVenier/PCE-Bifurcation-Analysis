"""
PCE-Galerkin solver for interval reconstruction of bifurcation branches.
"""

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt


def set_plot_style(usetex=True):
    plt.rcParams.update({
        "font.size": 13,
        "text.usetex": usetex,
        "font.family": "serif",
        "legend.fontsize": "medium",
        "axes.labelsize": "x-large",
        "axes.titlesize": "xx-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
        "lines.linewidth": 3,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def legendre_basis(x, N):
    """Orthogonal Legendre polynomials P_0, ..., P_N evaluated at x."""
    return np.polynomial.legendre.legvander(x, N).T


def legendre_norms2(N):
    """Squared L^2 norms <P_k, P_k> = 1 / (2k + 1) on [-1, 1] with weight 1/2."""
    return 1.0 / (2.0 * np.arange(N + 1) + 1.0)


class PCEGalerkin:
    """PCE-Galerkin solver for steady states f(u, mu) = 0 with mu ~ U(a, b)."""

    def __init__(self, rhs, jac_u=None, mu_bounds=(-1.0, 1.0), state_dim=1,
                 quad_order=100):
        self.rhs = rhs
        self.jac_u = jac_u
        self.mu_bounds = mu_bounds
        self.n = state_dim

        a, b = mu_bounds
        self.x_quad, w = np.polynomial.legendre.leggauss(quad_order)
        self.weights = 0.5 * w
        self.mu_quad = a + 0.5 * (b - a) * (self.x_quad + 1.0)

    def residual(self, c_flat, phi):
        c = c_flat.reshape(self.n, -1)
        gamma = legendre_norms2(phi.shape[0] - 1)
        f_quad = self.rhs(c @ phi, self.mu_quad)
        return (((f_quad * self.weights) @ phi.T) / gamma).ravel()

    def jacobian(self, c_flat, phi):
        c = c_flat.reshape(self.n, -1)
        n_pc = phi.shape[0]
        gamma = legendre_norms2(n_pc - 1)
        df_du = self.jac_u(c @ phi, self.mu_quad)
        psi = (phi * self.weights) / gamma[:, None]
        J = np.zeros((self.n, n_pc, self.n, n_pc))
        for i in range(self.n):
            for j in range(self.n):
                J[i, :, j, :] = (psi * df_du[i, j]) @ phi.T
        return J.reshape(self.n * n_pc, self.n * n_pc)

    def solve(self, guess, phi, tol=1e-10):
        jac = self.jacobian if self.jac_u is not None else None
        sol = root(self.residual, guess, args=(phi,), jac=jac, method="lm", tol=tol)
        if np.linalg.norm(sol.fun, np.inf) > 1e-6:
            sol = root(self.residual, guess, args=(phi,), jac=jac, method="hybr", tol=tol)
        return sol

    def continuation(self, N_max, u0=None, tol=1e-10, return_history=False):
        """Degree continuation from N = 0 to N = N_max."""
        phi_all = legendre_basis(self.x_quad, N_max)
        if u0 is None:
            guess = np.zeros(self.n)
        else:
            guess = np.asarray(u0, dtype=float).ravel()

        history = {}
        c = None
        for N in range(N_max + 1):
            phi = phi_all[: N + 1]
            if N > 0:
                c_pad = np.zeros((self.n, N + 1))
                c_pad[:, :N] = c
                guess = c_pad.ravel()

            sol = self.solve(guess, phi, tol=tol)
            if np.linalg.norm(sol.fun, np.inf) > 1e-5:
                return None

            c = sol.x.reshape(self.n, -1)
            if return_history:
                history[N] = c.copy()

        if return_history:
            return history
        return c

    def find_branches(self, N_max, n_samples=50, seed=42, unique_tol=1e-3,
                      box=(-2.0, 2.0)):
        """Run degree continuation from random N = 0 guesses and collect distinct branches."""
        rng = np.random.default_rng(seed)
        branches = []
        for _ in range(n_samples):
            u0 = rng.uniform(box[0], box[1], size=self.n)
            c = self.continuation(N_max, u0=u0)
            if c is None:
                continue
            is_new = True
            for b in branches:
                if np.linalg.norm(c - b) <= unique_tol:
                    is_new = False
                    break
            if is_new:
                branches.append(c)
        return branches

    def random_roots(self, N, n_samples=50, seed=42, unique_tol=1e-3, scale=1.0):
        """Solve the Galerkin system directly at fixed degree N from random guesses."""
        phi = legendre_basis(self.x_quad, N)
        rng = np.random.default_rng(seed)
        roots = []
        for _ in range(n_samples):
            guess = rng.normal(0.0, scale, size=self.n * (N + 1))
            sol = self.solve(guess, phi)
            if np.linalg.norm(sol.fun, np.inf) <= 1e-5:
                c = sol.x.reshape(self.n, -1)
                is_new = True
                for r in roots:
                    if np.linalg.norm(c - r) <= unique_tol:
                        is_new = False
                        break
                if is_new:
                    roots.append(c)
        return roots

    def eval(self, coeffs, n_grid=500):
        """Evaluate PCE expansion on a uniform grid in [mu_min, mu_max]."""
        coeffs = np.atleast_2d(coeffs)
        x = np.linspace(-1.0, 1.0, n_grid)
        a, b = self.mu_bounds
        mu = a + 0.5 * (b - a) * (x + 1.0)
        u = coeffs @ legendre_basis(x, coeffs.shape[1] - 1)
        return mu, u

    def errors(self, coeffs, exact_fn, n_grid=2000):
        """Return (L^2 error, L^inf error, Galerkin residual of the L^2 projection)."""
        coeffs = np.atleast_2d(coeffs)
        N = coeffs.shape[1] - 1

        mu, u_approx = self.eval(coeffs, n_grid=n_grid)
        u_exact = np.atleast_2d(exact_fn(mu))
        diff = np.linalg.norm(u_exact - u_approx, axis=0)
        err_l2 = float(np.sqrt(np.mean(diff**2)))
        err_linf = float(np.max(diff))

        phi = legendre_basis(self.x_quad, N)
        gamma = legendre_norms2(N)
        c_ls = ((exact_fn(self.mu_quad) * self.weights) @ phi.T) / gamma
        res = self.residual(c_ls.ravel(), phi).reshape(self.n, -1)
        res_ls = float(np.sqrt(np.sum(res**2 * gamma)))

        return err_l2, err_linf, res_ls
