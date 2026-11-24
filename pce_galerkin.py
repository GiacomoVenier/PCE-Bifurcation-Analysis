"""
pce_galerkin.py
==================
Single-run interval bifurcation branches reconstruction for parameter-dependent
nonlinear ODEs via generalized Polynomial Chaos Expansion (PCE) and 
Galerkin projection.

Code developed for the paper:
"Stochastic bifurcation analysis via polynomial chaos: consistency and convergence
of branch-approximating solutions"
"""

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt


def set_plot_style(usetex=True):
    plt.rcParams.update({
        "font.size": 13,
        "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{lmodern}" if usetex else "",
        "font.family": "serif",
        "legend.fontsize": "x-large",
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

class PCEGalerkinSolver:
    """
    PCE-Galerkin solver for steady states f(u, mu) = 0 with uniformly 
    distributed parameter mu ~ U(mu_min, mu_max).

    Parameters
    ----------
    rhs : callable
        Vector field f(u, mu) accepting u of shape (n, Q) and mu of shape (Q,),
        returning an array of shape (n, Q).
    jac_u : callable, optional
        Jacobian D_u f(u, mu) accepting u of shape (n, Q) and mu of shape (Q,),
        returning an array of shape (n, n, Q). If None, uses finite differences.
    mu_bounds : tuple of float
        Parameter interval (mu_min, mu_max).
    state_dim : int
        Dimension n of the state vector u in R^n.
    quadrature_order : int
        Number of Gauss-Legendre quadrature points Q.
    """

    def __init__(self, rhs, jac_u=None, mu_bounds=(-1.0, 1.0), state_dim=1, 
                 quadrature_order=100):
        self.rhs = rhs
        self.jac_u = jac_u
        self.mu_bounds = mu_bounds
        self.n = state_dim

        a, b = mu_bounds
        self.mu_mean = 0.5 * (a + b)
        self.mu_std = (b - a) / np.sqrt(12.0)

        self.x_quad, w_leg = np.polynomial.legendre.leggauss(quadrature_order)
        self.weights = 0.5 * w_leg
        self.mu_quad = a + 0.5 * (b - a) * (self.x_quad + 1.0)

        self._phi_full = None
        self._phi_eval = None
        self._max_deg = -1

    def _ensure_basis(self, max_degree):
        """Compute orthonormal Legendre polynomials Phi_k(x) up to max_degree."""
        if self._phi_full is None or max_degree > self._max_deg:
            vander = np.polynomial.legendre.legvander(self.x_quad, max_degree).T
            norms = np.sqrt(2.0 * np.arange(max_degree + 1) + 1.0)[:, None]
            self._phi_full = vander * norms
            self._max_deg = max_degree

    def _unpack(self, c_flat):
        """Reshape flat coefficient vector (n*(N+1),) into (n, N+1)."""
        return c_flat.reshape(self.n, -1)

    def residual(self, c_flat):
        """
        Compute PCE-Galerkin residual F_N(c) of shape (n*(N+1),).
        Block (d, k) is <f(u_N, mu)_d, Phi_k>_{L^2}.
        """
        c = self._unpack(c_flat)
        u_quad = c @ self._phi_eval
        f_quad = self.rhs(u_quad, self.mu_quad)
        res = (f_quad * self.weights) @ self._phi_eval.T
        return res.ravel()

    def jacobian(self, c_flat):
        """
        Compute Jacobian JF_N(c) of shape (n*(N+1), n*(N+1)).
        """
        c = self._unpack(c_flat)
        n_pc = self._phi_eval.shape[0]
        u_quad = c @ self._phi_eval
        df_du = self.jac_u(u_quad, self.mu_quad)

        weighted_phi = self._phi_eval * self.weights[None, :]
        J = np.einsum("ijq,kq,lq->ikjl", df_du, weighted_phi, self._phi_eval, 
                      optimize=True)
        return J.reshape(self.n * n_pc, self.n * n_pc)

    def _solve_fixed_degree(self, guess_flat, tol=1e-10):
        """Solve Galerkin algebraic system at the currently active degree."""
        jac_fn = self.jacobian if self.jac_u is not None else None
        sol = root(self.residual, guess_flat, jac=jac_fn, method="lm", tol=tol)
        if np.linalg.norm(sol.fun, ord=np.inf) > 1e-6:
            sol = root(self.residual, guess_flat, jac=jac_fn, method="hybr", 
                       tol=tol)
        return sol

    def degree_continuation(self, max_degree, u_init=None, tol=1e-10, 
                            return_history=False):
        """
        Algorithm 1: Isolate a branch-approximating solution via degree 
        continuation from N = 0 up to N = max_degree.

        Parameters
        ----------
        max_degree : int
            Target degree N_max.
        u_init : array_like of shape (n,), optional
            Initial guess at N = 0. Defaults to zeros.
        tol : float
            Root-finding tolerance.
        return_history : bool
            If True, returns a dict {deg: coeffs_matrix} for all degrees.
        """
        self._ensure_basis(max_degree)
        if u_init is None:
            guess = np.zeros(self.n, dtype=float)
        else:
            guess = np.asarray(u_init, dtype=float).ravel()

        history = {}
        c_curr = None

        for deg in range(max_degree + 1):
            n_pc = deg + 1
            self._phi_eval = self._phi_full[:n_pc]

            if deg > 0:
                c_pad = np.zeros((self.n, n_pc))
                c_pad[:, :deg] = c_curr
                guess = c_pad.ravel()

            sol = self._solve_fixed_degree(guess, tol=tol)
            if np.linalg.norm(sol.fun, ord=np.inf) > 1e-5:
                return None

            c_curr = self._unpack(sol.x)
            if return_history:
                history[deg] = c_curr.copy()

        return history if return_history else c_curr

    def find_multiple_branches(self, max_degree, n_samples=50, seed=42,
                               unique_tol=1e-3, box=(-2.0, 2.0)):
        """
        Run Algorithm 1 from multiple random initializations at N = 0 and
        find distinct branch-approximating solutions.
        """
        rng = np.random.default_rng(seed)
        branches = []

        for _ in range(n_samples):
            u0 = rng.uniform(box[0], box[1], size=self.n)
            coeffs = self.degree_continuation(max_degree, u_init=u0)
            if coeffs is None:
                continue

            if all(np.linalg.norm(coeffs - b) > unique_tol for b in branches):
                branches.append(coeffs)

        return branches

    def direct_random_roots(self, degree, n_samples=50, seed=42, 
                            unique_tol=1e-3, scale=1.0):
        """
        Solve the Galerkin system directly at fixed degree N without continuation
        """
        self._ensure_basis(degree)
        self._phi_eval = self._phi_full[: degree + 1]
        rng = np.random.default_rng(seed)
        roots = []

        for _ in range(n_samples):
            guess = rng.normal(0.0, scale, size=self.n * (degree + 1))
            sol = self._solve_fixed_degree(guess)
            if np.linalg.norm(sol.fun, ord=np.inf) <= 1e-5:
                c = self._unpack(sol.x)
                if all(np.linalg.norm(c - r) > unique_tol for r in roots):
                    roots.append(c)
        return roots

    def evaluate(self, coeffs, n_grid=500):
        """
        Evaluate a PCE solution on a uniform grid in [mu_min, mu_max].
        Returns (mu_grid, u_grid) where u_grid has shape (n, n_grid).
        """
        coeffs = np.atleast_2d(coeffs)
        deg = coeffs.shape[1] - 1
        x_grid = np.linspace(-1.0, 1.0, n_grid)
        vander = np.polynomial.legendre.legvander(x_grid, deg).T
        norms = np.sqrt(2.0 * np.arange(deg + 1) + 1.0)[:, None]

        a, b = self.mu_bounds
        mu_grid = a + 0.5 * (b - a) * (x_grid + 1.0)
        u_grid = coeffs @ (vander * norms)
        return mu_grid, u_grid

    def least_squares_projection(self, exact_branch_fn, max_degree):
        """
        Compute best L^2 polynomial approximation coefficients u_{LS,N} of
        exact_branch_fn, which should be known .
        """
        self._ensure_basis(max_degree)
        u_exact_quad = exact_branch_fn(self.mu_quad)
        return (u_exact_quad * self.weights) @ self._phi_full.T

    def compute_errors(self, coeffs, exact_branch_fn, n_grid=2000):
        """
        Compute ||u_exact - u_N||_{L^2}, ||u_exact - u_N||_{L^inf}, and
        Galerkin residual ||F_N(u_{LS,N})||_2 for Theorems 3.1/3.3 verification.
        """
        coeffs = np.atleast_2d(coeffs)
        deg = coeffs.shape[1] - 1

        mu_grid, u_approx = self.evaluate(coeffs, n_grid=n_grid)
        u_exact = np.atleast_2d(exact_branch_fn(mu_grid))
        diff = np.linalg.norm(u_exact - u_approx, axis=0)
        err_l2 = float(np.sqrt(np.mean(diff**2)))
        err_linf = float(np.max(diff))

        c_ls = self.least_squares_projection(exact_branch_fn, deg)
        self._phi_eval = self._phi_full[: deg + 1]
        res_ls = float(np.linalg.norm(self.residual(c_ls.ravel())))

        return {"l2": err_l2, "linf": err_linf, "res_ls": res_ls}
    

def plot_branches_2d(solver, branches, exact_fn=None, state_idx=0, 
                     xlabel=r"$\mu$", ylabel=r"$u$", ax=None):
    """Plot reconstructed PCE branches and optional exact curve in 2D."""
    if ax is None:
        _, ax = plt.subplots()

    if exact_fn is not None:
        mu_ex = np.linspace(solver.mu_bounds[0], solver.mu_bounds[1], 600)
        u_ex = np.atleast_2d(exact_fn(mu_ex))
        ax.plot(mu_ex, u_ex[state_idx], "k-", lw=3.5, label=r"Exact $\bar{u}$",
                zorder=1)

    for idx, coeffs in enumerate(branches):
        mu_g, u_g = solver.evaluate(coeffs)
        deg = np.atleast_2d(coeffs).shape[1] - 1
        ax.plot(mu_g, u_g[state_idx], "--", lw=2.5, 
                label=rf"PCE ($N={deg}$)" if idx == 0 else None)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    return ax


def plot_convergence(degrees, errors_list, ref_rates=None, ax=None):
    """Plot L^2 error, L^inf error, and residual decay vs N."""
    if ax is None:
        _, ax = plt.subplots()

    degs = np.asarray(degrees)
    l2 = [e["l2"] for e in errors_list]
    linf = [e["linf"] for e in errors_list]
    res = [e["res_ls"] for e in errors_list]

    ax.semilogy(degs, l2, "-o", color="#065895",
                label=r"$\|\bar{u} - u_N\|_{L^2}$")
    ax.semilogy(degs, linf, "-s", color="#77ac30",
                label=r"$\|\bar{u} - u_N\|_{L^\infty}$")
    ax.semilogy(degs, res, "-^", color="#f79a25",
                label=r"$\|\mathcal{F}(u_{LS,N})\|_{L^2}$")

    if ref_rates:
        for label, curve in ref_rates.items():
            ax.semilogy(degs, curve, "-.", color="k", alpha=0.75, label=label)

    ax.set_xlabel(r"$N$")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.85)
    return ax