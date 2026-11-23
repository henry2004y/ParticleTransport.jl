# # Spherical Coordinates: Solar Wind Cooling, Focusing & Escape Spectra
#
# Global coronal and heliospheric transport problems are naturally formulated in spherical
# coordinates ``(r, \theta, \phi)`` via `coords = SphericalCoordinates()`.
#
# This demo illustrates:
#
# 1. **1D Spherical Parker transport in a radial solar wind**: Adiabatic cooling
#    ``p(r) = p_0 (r_0 / r)^{2/3}`` combined with radial diffusion, using
#    [`InjectionSource`](@ref) and [`escaped_spectra`](@ref) at inner/outer absorbing
#    boundaries.
# 2. **1D Spherical Focused Transport in a radial interplanetary field**
#    ``B_r \propto 1/r^2``: Outward adiabatic magnetic focusing (``\mu \to +1``) conserving
#    ``(1 - \mu^2) r^2``.

import DisplayAs #hide
using ParticleTransport
using StochasticDiffEq: EM
using StaticArrays: SA
using Random: Random, MersenneTwister
using Statistics: mean
using Printf
using CairoMakie
CairoMakie.activate!(; type = "png") #hide

# ## 1. Radial Solar Wind Adiabatic Cooling & Escape Spectra
#
# In a constant radial wind ``\mathbf{u} = u_0 \hat{\mathbf{r}}``, the spherical divergence
# is ``\nabla \cdot \mathbf{u} = 2 u_0 / r``, yielding exact adiabatic cooling
# ``p(r) = p_0 (r_0 / r)^{2/3}`` in the zero-diffusion limit. Adding radial diffusion
# ``\kappa_{rr}`` broadens the momentum distribution of particles escaping the outer
# boundary ``r_{\max} = 3.0`` (recorded via [`escaped_spectra`](@ref)).

u_sw = 1.5
u_rad(x) = SA[u_sw, 0.0, 0.0]
B_rad(x) = SA[1.0 / x[1]^2, 0.0, 0.0]

r0 = 1.0
p0 = 4.0
r_bounds = ((0.5, 3.0),)

## Deterministic cooling reference (κ = 0)
param_det = prepare(;
    u = u_rad, B = B_rad, coords = SphericalCoordinates(), dims = 1,
    q = 1.0, m = 1.0, c = Inf,
)
prob_det = TransportProblem(make_state(SA[r0, pi / 2, 0.0], p0), (0.0, 1.4), param_det)
sol_det = solve(prob_det, EM(); dt = 2.0e-3)
rs_det = [s[1] for s in sol_det.u]
ps_det = [get_momentum(s) for s in sol_det.u]
ps_exact = [p0 * (r0 / r)^(2 / 3) for r in rs_det]

@printf "Adiabatic cooling at r = %.2f: p = %.3f (exact %.3f)\n" rs_det[end] ps_det[end] ps_exact[end]

## Diffusive radial transport with Maxwellian injection and absorbing boundaries
param_diff = prepare(;
    u = u_rad, B = B_rad, kappa = IsotropicDiffusion(0.15),
    coords = SphericalCoordinates(), dims = 1,
    q = 1.0, m = 1.0, c = Inf,
)
src = InjectionSource(;
    space = PointInjection(SA[r0, pi / 2, 0.0]),
    momentum = MaxwellianMomentum(; p_th = 2.0, pmin = 0.5, pmax = 12.0),
)
prob_diff = TransportProblem(make_state(SA[r0, pi / 2, 0.0], p0), (0.0, 4.0), param_diff)
ep_diff = EnsembleProblem(
    prob_diff; prob_func = injection_prob_func(src; rng = MersenneTwister(501))
)
Random.seed!(501)
sim_diff = solve(
    ep_diff, EM(), EnsembleThreads();
    dt = 5.0e-3, trajectories = 3_000,
    callback = TerminateOutside(r_bounds), save_everystep = false,
)
p_edges = 10.0 .^ range(log10(0.3), log10(12.0); length = 22)
esc = escaped_spectra(sim_diff, r_bounds, p_edges)

@printf "Escaped particles: inner (r_min) = %.0f,  outer (r_max) = %.0f\n" sum(esc[:xmin].counts) sum(esc[:xmax].counts)

# ## 2. Spherical Focused Transport: Radial Magnetic Focusing
#
# In ``B_r(r) \propto 1/r^2``, the focusing length is ``L_B = -(\hat{\mathbf{b}}\cdot\nabla\ln B)^{-1} = r / 2``,
# so an outward-moving particle is focused toward ``\mu \to +1`` while conserving
# ``(1 - \mu^2) r^2 = (1 - \mu_0^2) r_0^2``.

param_sfte = prepare(;
    B = B_rad, model = FocusedTransport(), coords = SphericalCoordinates(),
    dims = 1, q = 1.0, m = 1.0, c = Inf,
)
mu0_f = 0.25
prob_sfte = TransportProblem(make_state(SA[r0, pi / 2, 0.0], 1.5, mu0_f), (0.0, 2.5), param_sfte)
sol_sfte = solve(prob_sfte, EM(); dt = 2.0e-3)
rs_f = [s[1] for s in sol_sfte.u]
mus_f = [get_pitch_angle(s) for s in sol_sfte.u]
mus_f_exact = [sqrt(1.0 - (1.0 - mu0_f^2) * (r0 / r)^2) for r in rs_f]

@printf "Radial focusing at r = %.2f: μ = %.4f (exact %.4f)\n" rs_f[end] mus_f[end] mus_f_exact[end]

# ## Visualization

fig = Figure(; size = (1350, 430), fontsize = 14)

ax1 = Axis(
    fig[1, 1];
    title = "Solar Wind Adiabatic Cooling (κ = 0)",
    xlabel = "Radius r",
    ylabel = "Momentum p",
)
lines!(ax1, rs_det, ps_exact; color = :black, linestyle = :dash, linewidth = 2.2, label = "Exact p₀(r₀/r)^(2/3)")
lines!(ax1, rs_det, ps_det; color = :royalblue, linewidth = 2.0, label = "Spherical SDE")
axislegend(ax1; position = :rt, framevisible = false)

ax2 = Axis(
    fig[1, 2];
    title = "Escaped Spectra (Maxwellian Source at r₀ = 1)",
    xlabel = "Momentum p",
    ylabel = "dN / d(ln p)",
    xscale = log10,
)
scatterlines!(ax2, esc[:xmax].centers, esc[:xmax].dN_dlnp; color = :royalblue, linewidth = 2.0, label = "Outer r = 3.0 (cooled)")
scatterlines!(ax2, esc[:xmin].centers, esc[:xmin].dN_dlnp; color = :crimson, linewidth = 2.0, label = "Inner r = 0.5 (heated)")
axislegend(ax2; position = :rt, framevisible = false)

ax3 = Axis(
    fig[1, 3];
    title = "Spherical FTE Radial Focusing (B_r ∝ 1/r²)",
    xlabel = "Radius r",
    ylabel = "Pitch-angle cosine μ",
)
lines!(ax3, rs_f, mus_f_exact; color = :black, linestyle = :dash, linewidth = 2.2, label = "Exact √(1 - (1-μ₀²)(r₀/r)²)")
lines!(ax3, rs_f, mus_f; color = :seagreen, linewidth = 2.0, label = "Spherical FTE")
axislegend(ax3; position = :rb, framevisible = false)

fig = DisplayAs.PNG(fig) #hide
