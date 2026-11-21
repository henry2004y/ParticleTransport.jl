# ParticleTransport.jl

Stochastic differential equation (SDE) solver for energetic particle transport and acceleration in magnetized plasmas, inspired by [`xiaocanli/stochastic-parker`](https://github.com/xiaocanli/stochastic-parker) (GPAT) and designed to complement [`TestParticle.jl`](https://github.com/JuliaSpacePhysics/TestParticle.jl) within the SciML (`StochasticDiffEq.jl`) ecosystem.

## Features

- **1D/2D/3D Cartesian Parker Transport**: Forward-in-time Itô SDEs in `(x, y, z, ln p)` with anisotropic spatial diffusion ($\kappa_\parallel, \kappa_\perp$), plasma convection ($\mathbf{u}$), adiabatic/compressional acceleration ($-\frac{1}{3}\nabla\cdot\mathbf{u}$), optional momentum diffusion ($D_{pp}$), and guiding-center magnetic drifts ($\mathbf{v}_d$).
- **Analytical & Numerical MHD Fields**: Define $\mathbf{u}(\mathbf{x}, t)$ and $\mathbf{B}(\mathbf{x}, t)$ via analytical functions or gridded arrays interpolated with `FastInterpolations.jl`, with automatic spatial derivatives via `ForwardDiff.jl` / interpolator gradients.
- **Native SciML Integration**: Build standard `SciMLBase.SDEProblem` and `EnsembleProblem` objects solved directly with `StochasticDiffEq.jl` (`EM`, `SRIW1`, `EnsembleThreads`, etc.).
- **Boundary Conditions & Particle Splitting**: Composable SciML callbacks for absorbing, periodic, reflecting, and momentum boundaries, plus a generational ensemble runner for particle splitting in high-energy power-law tails.
- **Diagnostics**: Weighted energy/momentum spectra $f(p, t)$ and spatial distributions $n(\mathbf{x}, E, t)$ both from `EnsembleSolution` and via $O(1)$-memory streaming `EnsembleProblem` reductions.

## Quick Example

```julia
using ParticleTransport, StochasticDiffEq, StaticArrays

# 1D diffusive shock profile u(x) = -tanh(x / L) + 2
u(x) = SA[1.5 - 0.5 * tanh(x[1] / 0.1), 0.0, 0.0]
B(x) = SA[1.0, 0.0, 0.0]

param = prepare(; u, B, kappa = ConstantDiffusion(0.1, 0.0), dims = 1)
u0 = SA[0.0, 0.0, 0.0, log(1.0)]
prob = TransportProblem(u0, (0.0, 10.0), param)
sol = solve(prob, EM(); dt = 1.0e-3)
```
