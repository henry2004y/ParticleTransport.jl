# ParticleTransport.jl

`ParticleTransport.jl` solves energetic particle transport and acceleration equations in
magnetized plasmas using forward-in-time stochastic differential equations (SDEs) built on
`SciMLBase` and `StochasticDiffEq.jl`.

Inspired by [`xiaocanli/stochastic-parker`](https://github.com/xiaocanli/stochastic-parker)
(GPAT) and structured alongside
[`TestParticle.jl`](https://github.com/JuliaSpacePhysics/TestParticle.jl), it provides a
macroscopic diffusive transport solver that complements microscopic Lorentz and
guiding-center orbit tracing.

## Capabilities

- **1D/2D/3D Cartesian Parker Transport**: Forward-in-time Itô SDEs in ``(\mathbf{x}, \ln p)``
  with anisotropic spatial diffusion (``\kappa_\parallel, \kappa_\perp``), plasma convection
  (``\mathbf{u}``), adiabatic/compressional acceleration (``-\frac{1}{3}\nabla\cdot\mathbf{u}``),
  momentum diffusion (``D_{pp}``), and guiding-center magnetic drifts (``\mathbf{v}_d``).
- **Analytical & Gridded MHD Fields**: Define ``\mathbf{u}(\mathbf{x}, t)`` and
  ``\mathbf{B}(\mathbf{x}, t)`` as analytical functions (differentiated automatically via
  `ForwardDiff.jl`) or interpolated from gridded MHD arrays via `FastInterpolations.jl`.
- **Native SciML Integration**: Constructs standard `SciMLBase.SDEProblem` and
  `EnsembleProblem` objects solved with `StochasticDiffEq.jl` (`EM`, `SRIW1`,
  `EnsembleThreads`, etc.).
- **Boundary Conditions & Particle Splitting**: Composable SciML callbacks for absorbing,
  periodic, reflecting, and momentum boundaries, plus a weight-conserving generational
  particle splitter (`solve_with_splitting`) for steep high-energy power-law tails.
- **Diagnostics**: Weighted momentum spectra ``f(p, t)`` and spatial profiles
  ``n(\mathbf{x}, t)``, including ``O(1)``-memory streaming ensemble reductions.

## Quick Start

```jldoctest
julia> using ParticleTransport, StochasticDiffEq, StaticArrays

julia> u(x) = SA[1.5 - 0.5 * tanh(x[1] / 0.1), 0.0, 0.0];

julia> B(x) = SA[1.0, 0.0, 0.0];

julia> param = prepare(; u, B, kappa = ConstantDiffusion(0.1, 0.0), dims = 1);

julia> u0 = make_state(0.0, 1.0);

julia> prob = TransportProblem(u0, (0.0, 1.0), param);

julia> sol = solve(prob, EM(); dt = 1.0e-3);

julia> sol.retcode == StochasticDiffEq.ReturnCode.Success
true
```

## Documentation Guide

- **[Physics & SDE Formulation](@ref Physics-SDE)**: Parker's transport equation, its physical
  assumptions, and the complete derivation of the equivalent forward-in-time Itô SDE in
  ``(\mathbf{x}, \ln p)``.
- **[Numerical Methods](@ref Numerical-Methods)**: Choosing SDE integrators and step sizes, continuous vs.
  impulsive injection, particle splitting, boundary conditions, streaming diagnostics, and
  when to use `ParticleTransport.jl` vs. `TestParticle.jl`.
- **Analytical Benchmarks**: Executable examples validating the SDE solver against closed-form
  solutions for convection–diffusion, absorbing boundaries, adiabatic heating/cooling,
  diffusive shock acceleration (DSA), stochastic momentum diffusion, leaky-box spectra, and
  3D anisotropic/curved-field diffusion.
- **Comparison with TestParticle.jl**: Side-by-side executable comparisons between
  microscopic orbit tracing (`TestParticle.jl`) and macroscopic Parker transport
  (`ParticleTransport.jl`) for magnetic drifts and parallel MHD shocks.
- **[API Reference](@ref)**: Complete reference of exported and public types and functions.
