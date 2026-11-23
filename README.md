# ParticleTransport.jl

Stochastic differential equation (SDE) solver for energetic particle transport and acceleration in magnetized plasmas, inspired by [`xiaocanli/stochastic-parker`](https://github.com/xiaocanli/stochastic-parker) (GPAT) and designed to complement [`TestParticle.jl`](https://github.com/JuliaSpacePhysics/TestParticle.jl) within the SciML (`StochasticDiffEq.jl`) ecosystem.

## Features

- **Parker & Focused Transport (Cartesian & Spherical)**: Forward-in-time Itô SDEs in 1D/2D/3D Cartesian `(x, y, z, ln p[, μ])` or Spherical `(r, θ, φ, ln p[, μ])` coordinates, including anisotropic spatial diffusion ($\kappa_\parallel, \kappa_\perp$), pitch-angle diffusion ($D_{\mu\mu}$), magnetic focusing/mirroring, plasma convection ($\mathbf{u}$), adiabatic/compressional acceleration, momentum diffusion ($D_{pp}$), guiding-center magnetic drifts ($\mathbf{v}_d$), 2D out-of-plane tracking, and selective acceleration-region masking.
- **Physics-Based Diffusion Models**: Built-in `QLTDiffusion` (Giacalone & Jokipii 1999), `NLGCDiffusion` (Matthaeus et al. 2003), `AlfvenWaveMomentumDiffusion` (Skilling 1975), `FlowShearMomentumDiffusion` (Earl et al. 1988), and `QLTPitchAngleDiffusion`, plus power-law and custom callable models.
- **Analytical & Numerical MHD Fields**: Define $\mathbf{u}(\mathbf{x}, t)$ and $\mathbf{B}(\mathbf{x}, t)$ via analytical functions or gridded arrays interpolated with `FastInterpolations.jl`, with automatic spatial derivatives via `ForwardDiff.jl` / interpolator gradients.
- **Native SciML Integration**: Build standard `SciMLBase.SDEProblem` and `EnsembleProblem` objects solved directly with `StochasticDiffEq.jl` (`EM`, `SRIW1`, `EnsembleThreads`, etc.).
- **Injection, Boundary Conditions & Particle Splitting**: Initial/continuous injection samplers (delta, Maxwellian, power-law, box, and field-conditioned rejection sampling), composable callbacks (`TerminateOutside`, `PeriodicBoundary`, `ReflectingBoundary`, `MomentumBounds`, `PitchAngleBounds`), and generational particle splitting with configurable `pmin_split`.
- **Diagnostics**: Weighted energy/momentum spectra $f(p, t)$, 1D/2D/3D local phase-space distributions (`LocalDistribution`), per-face escaped-particle spectra (`escaped_spectra`), and $O(1)$-memory streaming `EnsembleProblem` reductions (including multi-snapshot `saveat`).

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
