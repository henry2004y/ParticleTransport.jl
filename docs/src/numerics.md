# [Numerical Methods](@id Numerical-Methods)

This page explains how to set up, integrate, and analyze Parker transport simulations in
`ParticleTransport.jl`, including time-step selection, injection strategies, variance
reduction via particle splitting, and how the solver relates to `TestParticle.jl`.

## State Vector, Dimensions & Units

Every simulation uses a uniform 4-component state vector

```math
\mathbf{u} = \bigl(x,\; y,\; z,\; \ln p\bigr),
```

constructed with [`make_state`](@ref) and inspected with [`get_position`](@ref),
[`get_momentum`](@ref), and [`get_energy`](@ref):

```jldoctest
julia> using ParticleTransport, StaticArrays

julia> u0 = make_state(SA[1.0, 2.0, 0.0], 1.0);

julia> get_position(u0), get_momentum(u0)
([1.0, 2.0, 0.0], 1.0)
```

- **Spatial dimensionality (`dims = 1, 2, 3`)**: Passed to [`prepare`](@ref). Inactive
  spatial coordinates are masked to zero in both drift and diffusion, so 1D, 2D, and 3D
  problems share identical kernels and diagnostics.
- **Out-of-place vs. in-place**: Passing an `SVector{4}` to [`TransportProblem`](@ref)
  builds an allocation-free out-of-place `SDEProblem{false}` (recommended for CPU threads
  and GPU kernels). Passing a mutable `Vector` builds an in-place `SDEProblem{true}`.
- **SI vs. dimensionless units**: By default `prepare` uses SI constants for `species = Proton`
  (`q = Proton.q`, `m = Proton.m`, `c = C_SI`). For dimensionless benchmarks, pass `q`, `m`,
  and `c` explicitly (use `c = Inf` for non-relativistic kinematics ``v = p / m``).

## Choosing an SDE Integrator & Time Step

`TransportProblem` returns a standard `SciMLBase.SDEProblem`, so any solver from
`StochasticDiffEq.jl` that supports non-diagonal noise can be used:

| Algorithm | Strong / Weak Order | Best Used For |
| :--- | :--- | :--- |
| `EM()` | 0.5 / 1.0 (fixed `dt`) | Standard Monte Carlo ensembles, shocks, boundary callbacks, particle splitting |
| `LambaEM()` | 0.5 / 1.0 (adaptive) | Smooth fields with spatially varying diffusion timescales |
| `SRIW1()` / `SOSRI()` | 1.5 / 2.0 (adaptive) | Diagonal-noise problems or high-precision weak convergence checks |

For fixed-step Euler–Maruyama (`EM()`), choose ``\Delta t`` to resolve the shortest
physical scale in the problem:

1. **Shock or gradient scale ``L``**:
   ```math
   |\mathbf{u}|\,\Delta t \ll L, \qquad \sqrt{2\kappa\,\Delta t} \ll L.
   ```
   In diffusive shock acceleration with shock ramp width ``L_{\mathrm{sh}}``, requiring
   ``\sqrt{2\kappa\,\Delta t} \lesssim L_{\mathrm{sh}}`` ensures that particles sample the
   compression ``-\frac{1}{3}\nabla\cdot\mathbf{u}`` accurately across the ramp.
2. **Curved magnetic fields** (radius of curvature ``R_c``):
   ```math
   \frac{\kappa_\parallel \Delta t}{R_c^2} \ll 1,
   ```
   so the discrete tangent step stays close to the curved field line and the Itô drift
   ``\nabla\cdot\boldsymbol{\kappa}`` cancels the geometric curvature push accurately.
3. **Momentum diffusion**:
   ```math
   \frac{D_{pp}(p)}{p^2}\,\Delta t \ll 1.
   ```

## Analytical vs. Gridded MHD Fields

Both `u` and `B` passed to [`prepare`](@ref) may be:
- **Analytical functions** `f(x)` (static) or `f(x, t)` (time-dependent) returning an
  `SVector{3}`. Spatial Jacobians, divergences (``\nabla\cdot\mathbf{u}``), and curls
  (``\nabla\times(\mathbf{B}/B^2)``) are computed automatically via `ForwardDiff.jl`.
- **Gridded 1D/2D/3D arrays** wrapped with [`build_interpolator`](@ref) (backed by
  `FastInterpolations.jl`), which provides analytical interpolant gradients without finite
  differences.
- **Time-varying MHD snapshots** wrapped with [`LazyTimeInterpolator`](@ref), which
  interpolates linearly between consecutive snapshots on demand.
- **`nothing`** (default), representing a zero field with zero runtime overhead.

## Impulsive vs. Continuous Injection

In the forward-in-time SDE method, the source term ``Q(\mathbf{x}, p, t)`` is represented
by how initial conditions ``(\mathbf{x}_0, p_0, t_0)`` are sampled across the ensemble:

- **Impulsive source** ``Q \propto \delta(t)``: Every trajectory runs over ``(0, T)``.
- **Continuous steady source** ``Q(t) = Q_0`` on ``[0, T]``: Sample each trajectory's
  injection time uniformly ``t_{\mathrm{inj}} \sim \mathcal{U}(0, T)`` in `prob_func` and
  integrate over ``(t_{\mathrm{inj}}, T)``:
  ```julia
  t_injs = rand(rng, n_traj) .* T_max
  prob_func = (prob, ctx) -> remake(prob; tspan = (t_injs[ctx.sim_id], T_max))
  ```
  As ``T`` exceeds the relevant acceleration or escape timescale, the distribution at
  ``t = T`` converges to the exact steady-state solution of Parker's transport equation.

## Particle Splitting for Power-Law Tails

In acceleration problems such as DSA (``f(p) \propto p^{-4}``), the number of particles
above momentum ``p`` drops steeply (``N(>p) \propto p^{-1}`` or faster). Direct Monte Carlo
quickly suffers from shot noise in the high-energy tail.

[`solve_with_splitting`](@ref) implements generational momentum-based particle splitting
configured by [`SplitRule`](@ref):

1. Whenever a particle's momentum increases by a factor `ratio` above its current split
   threshold, its statistical weight ``w`` is divided by `nsplit` and `nsplit - 1` identical
   child particles are recorded at ``(\mathbf{u}(t), t)`` with weight ``w / \mathrm{nsplit}``.
2. After the current generation finishes, the queued children are integrated in the next
   parallel `EnsembleProblem` generation (up to `max_generations` or `max_particles`).
3. **Exact weight conservation**: At every split event,
   ``w_{\mathrm{parent}} + (n_{\mathrm{split}} - 1) w_{\mathrm{child}} = w_{\mathrm{old}}``,
   so total particle weight ``\sum_i w_i`` is strictly conserved to floating-point precision
   while boosting statistical counts in the high-energy tail by orders of magnitude.

## Boundary Conditions & Streaming Diagnostics

- **Spatial & momentum callbacks**: Pass [`TerminateOutside`](@ref) (absorbing / free-escape
  boundary), [`PeriodicBoundary`](@ref), [`ReflectingBoundary`](@ref), or
  [`MomentumBounds`](@ref) via the standard SciML `callback` keyword (combining multiple
  callbacks with `CallbackSet`).
- **Standard diagnostics**: [`momentum_spectrum`](@ref) and [`spatial_density`](@ref) accept
  an `EnsembleSolution`, a [`SplittingResult`](@ref), or a raw vector of states.
- **``O(1)``-memory streaming**: For very large ensembles (``N \gtrsim 10^5``), use
  [`streaming_spectrum_ensemble`](@ref) together with [`spectrum_from_counts`](@ref) to
  accumulate weighted histogram counts on the fly without storing individual trajectories.

## ParticleTransport.jl vs. TestParticle.jl

`ParticleTransport.jl` and [`TestParticle.jl`](https://github.com/JuliaSpacePhysics/TestParticle.jl)
share the same SciML ecosystem, `PlasmaPopulations.jl` species, and `FastInterpolations.jl`
field interpolation, but target complementary physical regimes:

| Aspect | `TestParticle.jl` | `ParticleTransport.jl` |
| :--- | :--- | :--- |
| **Governing equation** | Lorentz ODE ``\dot{\mathbf{p}} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B})`` or Guiding-Center ODE | Parker's transport PDE via forward-in-time Itô SDE in ``(\mathbf{x}, \ln p)`` |
| **Scale regime** | Kinetic / microscopic (``r_g \sim L`` or ``\lambda \gtrsim L``) | Macroscopic / diffusive (``r_g \ll \lambda \ll L``) |
| **Pitch-angle distribution** | Arbitrary, anisotropic, gyrophase-resolved | Near-isotropic in the local plasma frame |
| **Wave–particle scattering** | Explicit turbulent ``\delta\mathbf{B}(\mathbf{x}, t)`` fields | Parameterized via ``\kappa_\parallel, \kappa_\perp, D_{pp}`` |
| **Time step** | Must resolve gyroperiod ``\Omega_c^{-1}`` | Resolves macroscopic flow/diffusion scales ``\gg \Omega_c^{-1}`` |
| **Best for** | Magnetospheric access, shock microphysics, measuring ``\kappa_\parallel, \kappa_\perp`` from turbulence | Diffusive shock acceleration, solar modulation, stochastic acceleration, global MHD cosmic-ray transport |

### Mapping `TestParticle.jl` Demos to `ParticleTransport.jl`

Several `TestParticle.jl` documentation examples have direct counterparts or complementary
views in `ParticleTransport.jl`:

| `TestParticle.jl` Demo | Physical Correspondence in `ParticleTransport.jl` |
| :--- | :--- |
| `demo_magnetic_drift.jl` | Isotropic pitch-angle average of curvature + grad-``B`` drift equals `magnetic_drift` (``\mathbf{v}_d = \frac{pv}{3q}\nabla\times(\mathbf{B}/B^2)``) **exactly** — see the live comparison demo. |
| `demo_shock.jl` (parallel shock) | Collisionless orbit tracing sees ``\mathbf{E} = 0`` and no acceleration across a parallel shock, whereas `ParticleTransport.jl` captures diffusive shock acceleration via ``\kappa_\parallel`` — see the live comparison demo. |
| `demo_cosmicray.jl` | Measures microscopic ``\langle|\Delta\mathbf{x}|^2\rangle`` in synthetic 3D MHD turbulence to determine ``(\kappa_\parallel, \kappa_\perp)``, which directly parameterizes `ConstantDiffusion` / `PowerLawDiffusion` in `ParticleTransport.jl`. |
| `demo_ExB_drift.jl` | Uniform ``\mathbf{E}\times\mathbf{B}`` drift corresponds to pure plasma convection ``\mathbf{u} = (\mathbf{E}\times\mathbf{B})/B^2`` with `ZeroDiffusion()`. |
| `demo_adiabatic_acceleration.jl` | Microscopic betatron + Fermi curvature-drift energization in a compressing mirror corresponds macroscopically to isotropic adiabatic heating ``-\frac{1}{3}(\nabla\cdot\mathbf{u})\,p\,\partial_p f``. |
| `demo_fermi_foreshock.jl` | Microscopic 1D Fermi bouncing between converging magnetic walls (with/without ``\delta B`` scattering) mirrors 1D diffusive compression in `ParticleTransport.jl`. |
