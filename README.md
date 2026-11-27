# PCE-Bifurcation-Analysis

Interval reconstruction of steady-state bifurcation branches $\mathbf{f}(\mathbf{u}, \mu) = \mathbf{0}$ for parameter-dependent ODEs via Polynomial Chaos Expansion (PCE) and Galerkin projection with degree continuation.

## Structure

- [`pce_galerkin.py`](pce_galerkin.py): `PCEGalerkin` solver (Gauss-Legendre quadrature, Galerkin projection, degree continuation, random root-finding, error metrics) and plotting helpers (`plot_2d`, `plot_3d`, `plot_convergence`).
- [`benchmarks/01_pitchfork.ipynb`](benchmarks/01_pitchfork.ipynb): Supercritical pitchfork, with oscillating roots vs. degree continuation and convergence across smooth, boundary-singularity, and interior-singularity regimes.
- [`benchmarks/02_s_shaped.ipynb`](benchmarks/02_s_shaped.ipynb): S-shaped bifurcation, with oscillating roots vs. degree continuation, convergence in a uniqueness region, and branches on partially supported intervals.
- [`benchmarks/03_genetic_toggle_switch.ipynb`](benchmarks/03_genetic_toggle_switch.ipynb): 2D genetic toggle switch, with double pitchfork reconstruction and uniqueness regime.
- [`benchmarks/04_lorenz_system.ipynb`](benchmarks/04_lorenz_system.ipynb): 3D Lorenz system, with pre-bifurcation uniqueness and pitchfork reconstruction.
- [`benchmarks/05_spherical_surface.ipynb`](benchmarks/05_spherical_surface.ipynb): System with a degenerate spherical steady-state manifold, with random initialization vs. degree continuation across intervals of increasing width.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

In each notebook in [`benchmarks/`](benchmarks/), one can freely adjust the parameters, such as polynomial degree (`N` / `N_max`), parametric interval (`mu_bounds`), number of initializations (`n_samples`), quadrature order (`quad_order`), random `seed` and any other configuration parameter to explore how the method behaves in different regimes.
