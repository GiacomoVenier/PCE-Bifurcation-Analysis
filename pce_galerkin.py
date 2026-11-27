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

    def continuation(self, N_max, u0=None, tol=1e-10, res_tol=1e-5,
                     return_history=False, stop_on_fail=False, start_degree=0):
        """Degree continuation from N = start_degree to N = N_max."""
        phi_all = legendre_basis(self.x_quad, N_max)
        if u0 is None:
            guess = np.zeros(self.n * (start_degree + 1))
        else:
            guess = np.asarray(u0, dtype=float).ravel()

        history = {}
        c = None
        for N in range(start_degree, N_max + 1):
            phi = phi_all[: N + 1]
            if N > start_degree:
                c_pad = np.zeros((self.n, N + 1))
                c_pad[:, :N] = c
                guess = c_pad.ravel()
            guess[np.abs(guess) < 1e-14] = 0.0

            sol = self.solve(guess, phi, tol=tol)
            if res_tol is not None and np.linalg.norm(sol.fun, np.inf) > res_tol:
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
                      box=(-2.0, 2.0), initialization=(), start_degree=0,
                      res_tol=1e-5):
        """Run degree continuation from random N = start_degree guesses and collect distinct branches."""
        rng = np.random.default_rng(seed)
        branches = []
        guesses = [
            rng.uniform(box[0], box[1], size=self.n * (start_degree + 1))
            for _ in range(n_samples)
        ] + [np.asarray(u0, dtype=float).ravel() for u0 in initialization]

        for u0 in guesses:
            c = self.continuation(N_max, u0=u0, res_tol=res_tol,
                                  start_degree=start_degree)
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


def plot_2d(solver, branches, exact_curves=(), state_idx=0, branch_style="--",
            branch_lw=2.5, branch_zorder=5, branch_alpha=1.0, branch_labels=None,
            colors=None, exact_style="k-", exact_lw=2.5, exact_zorder=1,
            exact_label=None, ylabel=r"$u$", title=None, xlim=None, ylim=None,
            ax=None):
    """Plot 2D (mu, u[state_idx]) exact curves and PCE branches on an axis."""
    if ax is None:
        _, ax = plt.subplots()

    for idx, (mu_ex, u_ex) in enumerate(exact_curves):
        u_arr = u_ex[state_idx] if np.ndim(u_ex) == 2 else u_ex
        ax.plot(mu_ex, u_arr, exact_style, lw=exact_lw, zorder=exact_zorder,
                label=exact_label if idx == 0 else None)

    for idx, c in enumerate(branches):
        mu_g, u_g = solver.eval(c)
        kw = {"lw": branch_lw, "zorder": branch_zorder, "alpha": branch_alpha}
        if colors is not None:
            kw["color"] = colors[idx % len(colors)]
        if branch_labels is not None:
            kw["label"] = branch_labels[idx]
        ax.plot(mu_g, u_g[state_idx], branch_style, **kw)

    ax.set_xlabel(r"$\mu$")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, linestyle="--", alpha=0.4)
    if exact_label is not None or branch_labels is not None:
        ax.legend(loc="best", framealpha=0.85)
    return ax


def plot_3d(solver, branches, exact_curves=(), idx_pair=(0, 1),
            labels=(r"$x$", r"$y$"), branch_style="--", branch_lw=2.5,
            branch_zorder=5, branch_alpha=1.0, branch_labels=None, colors=None,
            exact_style="k-", exact_lw=2.5, exact_zorder=1, exact_label=None,
            zlabel=r"$\mu$", elev=17, azim=-117, title=None, ax=None):
    """Plot 3D (u_i, u_j, mu) exact curves and PCE branches on an axis."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    i, j = idx_pair
    for idx, (mu_ex, u_ex) in enumerate(exact_curves):
        ax.plot(u_ex[i], u_ex[j], mu_ex, exact_style, lw=exact_lw,
                zorder=exact_zorder, label=exact_label if idx == 0 else None)

    for idx, c in enumerate(branches):
        mu_g, u_g = solver.eval(c)
        kw = {"lw": branch_lw, "zorder": branch_zorder, "alpha": branch_alpha}
        if colors is not None:
            kw["color"] = colors[idx % len(colors)]
        if branch_labels is not None:
            kw["label"] = branch_labels[idx]
        ax.plot(u_g[i], u_g[j], mu_g, branch_style, **kw)

    ax.set_xlabel(labels[0], labelpad=8)
    ax.set_ylabel(labels[1], labelpad=8)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zlabel, rotation=90, labelpad=10)
    if np.ptp(ax.get_xlim()) < 0.2:
        ax.set_xlim(-1.0, 1.0)
    if np.ptp(ax.get_ylim()) < 0.2:
        ax.set_ylim(-1.0, 1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(2))
        axis.set_tick_params(pad=2)
    ax.view_init(elev=elev, azim=azim)
    if title is not None:
        ax.set_title(title, y=0.96)
    if exact_label is not None or branch_labels is not None:
        ax.legend(loc="upper left", framealpha=0.85)
    return ax


def plot_convergence(solver, history, exact_fn, ref_rates=(), ax=None):
    """Plot L^2 error, L^inf error, and Galerkin residual vs degree N."""
    if ax is None:
        _, ax = plt.subplots()

    degrees = np.array(sorted(N for N in history if N >= 1))
    anchor_idx = int(0.6 * len(degrees))
    anchor_scale = 1.6

    err_l2, err_linf, res_ls = [], [], []
    for N in degrees:
        e2, einf, r = solver.errors(history[int(N)], exact_fn)
        err_l2.append(e2)
        err_linf.append(einf)
        res_ls.append(r)

    ax.semilogy(degrees, err_l2, "-", color="#065895",
                label=r"$\|\bar{u} - u_N\|_{L^2}$")
    ax.semilogy(degrees, err_linf, "-", color="#77ac30",
                label=r"$\|\bar{u} - u_N\|_{L^\infty}$")
    ax.semilogy(degrees, res_ls, "-", color="#f79a25",
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
        ax.semilogy(degrees, ref_curve, style, color="k", alpha=0.75, label=label)

    ax.set_title("Convergence")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("Error")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.85)
    return ax
