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

    def continuation(self, N_max, u0=None, tol=1e-10, return_history=False,
                     stop_on_fail=False):
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
                if stop_on_fail:
                    break
                return None

            c = sol.x.reshape(self.n, -1)
            if return_history:
                history[N] = c.copy()

        if return_history:
            return history
        return c

    def find_branches(self, N_max, n_samples=50, seed=42, unique_tol=1e-3,
                      box=(-2.0, 2.0), initialization=[]):
        """Run degree continuation from random N = 0 guesses and collect distinct branches."""
        rng = np.random.default_rng(seed)
        branches = []
        iterations = n_samples
        
        if len(initialization)>0:
            iterations+=len(initialization)
            
        for i in range(iterations):
            if i<n_samples:
                u0 = rng.uniform(box[0], box[1], size=self.n)
            else:
                u0 = initialization[i-n_samples]
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


def plot_roots_vs_branches(rhs, jac_u, mu_bounds, exact_curves, N=10,
                           n_samples=20, seed=2, scale=0.8, unique_tol=1e-6,
                           box=(-2.0, 2.0)):
    """Compare direct random root-finding at degree N with degree continuation."""
    solver = PCEGalerkin(rhs, jac_u, mu_bounds=mu_bounds, state_dim=1,
                         quad_order=2 * N + 1)
    all_roots = solver.random_roots(N, n_samples=n_samples, seed=seed, scale=scale)
    branches = solver.find_branches(N, n_samples=n_samples, seed=seed,
                                    unique_tol=unique_tol, box=box)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for c in all_roots:
        mu_g, u_g = solver.eval(c)
        axes[0].plot(mu_g, u_g[0], alpha=0.5)
    for mu_ex, u_ex in exact_curves:
        axes[0].plot(mu_ex, u_ex, "k:", lw=3.0, zorder=5)
    axes[0].set_title("Random initializations")
    axes[0].set_xlabel(r"$\mu$")
    axes[0].set_ylabel(r"$u$")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    for c in branches:
        mu_g, u_g = solver.eval(c)
        axes[1].plot(mu_g, u_g[0])
    for mu_ex, u_ex in exact_curves:
        axes[1].plot(mu_ex, u_ex, "k:", lw=3.0, zorder=5)
    axes[1].set_title("Degree continuation")
    axes[1].set_xlabel(r"$\mu$")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_regime(rhs, jac_u, exact_fn, name, bounds, ref_rates, u0,
                N_max=30, plot_degrees=None, exact_label=r"Exact $\bar{u}$"):
    """Plot branch approximations and convergence vs degree N for one interval."""
    a, b = bounds
    if plot_degrees is None:
        plot_degrees = [0, 1, N_max // 2, N_max]

    degrees = np.arange(1, N_max + 1)
    anchor_idx = int(0.6 * len(degrees))
    anchor_scale = 1.6

    solver = PCEGalerkin(rhs, jac_u, mu_bounds=bounds, state_dim=1,
                         quad_order=2 * N_max + 1)
    history = solver.continuation(N_max, u0=u0, return_history=True)

    fig, (ax_branch, ax_conv) = plt.subplots(1, 2, figsize=(11, 4.5))

    mu_ex = np.linspace(a, b, 600)
    ax_branch.plot(mu_ex, exact_fn(mu_ex)[0], "k-", lw=3.5, label=exact_label)
    for N in plot_degrees:
        mu_g, u_g = solver.eval(history[N])
        ax_branch.plot(mu_g, u_g[0], "--", lw=3.0, label=rf"$N={N}$")
    ax_branch.set_title(f"{name}")
    ax_branch.set_xlabel(r"$\mu$")
    ax_branch.set_ylabel(r"$u$")
    ax_branch.grid(True, linestyle="--", alpha=0.4)
    ax_branch.legend(loc="lower right", framealpha=0.85)

    err_l2, err_linf, res_ls = [], [], []
    for N in degrees:
        e2, einf, r = solver.errors(history[int(N)], exact_fn)
        err_l2.append(e2)
        err_linf.append(einf)
        res_ls.append(r)

    ax_conv.semilogy(degrees, err_l2, "-", color="#065895",
                     label=r"$\|\bar{u} - u_N\|_{L^2}$")
    ax_conv.semilogy(degrees, err_linf, "-", color="#77ac30",
                     label=r"$\|\bar{u} - u_N\|_{L^\infty}$")
    ax_conv.semilogy(degrees, res_ls, "-", color="#f79a25",
                     label=r"$\|\mathcal{F}(u_{LS,N})\|_{L^2}$")

    curves = {"l2": err_l2, "linf": err_linf, "res": res_ls}
    for label, style, rate_fn, anchor_key in ref_rates:
        target = np.asarray(curves[anchor_key])
        if rate_fn is None:
            mask = (degrees >= 5) & (target > 1e-13)
            slope, _ = np.polyfit(degrees[mask], np.log(target[mask]), 1)
            rho_est = np.exp(-slope)
            print(f"Estimated rho ({anchor_key}): {rho_est:.4f}")
            ref_curve = rho_est ** (-degrees.astype(float))
        else:
            ref_curve = rate_fn(degrees.astype(float))
        ref_curve *= anchor_scale * target[anchor_idx] / ref_curve[anchor_idx]
        ax_conv.semilogy(degrees, ref_curve, style, color="k", alpha=0.75, label=label)

    ax_conv.set_title("Convergence")
    ax_conv.set_xlabel(r"$N$")
    ax_conv.set_ylabel("Error")
    ax_conv.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_conv.legend(loc="lower left", framealpha=0.85)

    plt.tight_layout()
    plt.show()
