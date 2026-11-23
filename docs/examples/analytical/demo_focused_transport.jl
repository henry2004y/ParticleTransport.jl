# # Focused Transport: Magnetic Mirroring & Pitch-Angle Scattering
#
# When the scattering mean free path is comparable to or larger than the macroscopic field
# scale, the pitch-angle distribution becomes anisotropic and Parker's isotropic equation no
# longer applies. `ParticleTransport.jl` solves the 5D **Focused Transport Equation (FTE)**
# in ``(x, y, z, \ln p, \mu)`` via `model = FocusedTransport()`.
#
# This demo illustrates two foundational FTE benchmarks:
#
# 1. **Magnetic mirroring in a converging bottle**: Deterministic parallel streaming and
#    focusing ``d\mu/dt = -\frac{1-\mu^2}{2} v (\hat{\mathbf{b}}\cdot\nabla\ln B)``
#    conserves the adiabatic invariant ``(1 - \mu^2) / B(x)``.
# 2. **Pitch-angle isotropization**: Under [`QLTPitchAngleDiffusion`](@ref) and
#    [`PitchAngleBounds`](@ref), a beamed initial population relaxes to an isotropic
#    distribution uniform in ``\mu \in [-1, 1]``.

import DisplayAs #hide
using ParticleTransport
using StochasticDiffEq: EM
using StaticArrays: SA
using LinearAlgebra: norm
using Random: Random
using Statistics: mean, std
using Printf
using CairoMakie
CairoMakie.activate!(; type = "png") #hide

# ## 1. Magnetic Mirroring in a Converging Bottle
#
# Consider a static magnetic bottle ``B_x(x) = B_0 (1 + (x / L)^2)`` with zero bulk flow
# and zero scattering. A particle launched from ``x = 0`` with ``\mu_0 = 0.6`` streams into
# the stronger-field region, mirrors at ``x_{\mathrm{m}} = L \sqrt{\mu_0^2 / (1 - \mu_0^2)}``
# where ``\mu = 0``, and oscillates back and forth while conserving ``(1 - \mu^2) / B(x)``.

B0 = 1.0
L_bottle = 2.0
B_bottle(x) = SA[B0 * (1.0 + (x[1] / L_bottle)^2), 0.0, 0.0]

param_mir = prepare(;
    B = B_bottle, model = FocusedTransport(), dims = 1,
    q = 1.0, m = 1.0, c = Inf,
)
mu0_mir = 0.6
p0_mir = 1.5
u0_mir = make_state(SA[0.0, 0.0, 0.0], p0_mir, mu0_mir)
prob_mir = TransportProblem(u0_mir, (0.0, 12.0), param_mir)
sol_mir = solve(prob_mir, EM(); dt = 2.0e-3)

ts_mir = sol_mir.t
xs_mir = [s[1] for s in sol_mir.u]
mus_mir = [get_pitch_angle(s) for s in sol_mir.u]
inv_mir = [(1.0 - mu^2) / norm(B_bottle(SA[x, 0.0, 0.0])) for (x, mu) in zip(xs_mir, mus_mir)]
inv0 = (1.0 - mu0_mir^2) / B0
x_mirror = L_bottle * sqrt(mu0_mir^2 / (1.0 - mu0_mir^2))

@printf "Mirror point: max(x) = %.3f (exact %.3f),  max |Δinv|/inv₀ = %.2e\n" maximum(xs_mir) x_mirror maximum(abs.(inv_mir .- inv0)) / inv0

# ## 2. Pitch-Angle Scattering Under QLT Diffusion
#
# Now turn on quasi-linear pitch-angle diffusion ([`QLTPitchAngleDiffusion`](@ref)) in a
# uniform magnetic field with [`PitchAngleBounds`](@ref) reflecting ``\mu`` at ``\pm 1``.
# Starting from a narrow beam at ``\mu_0 = 0.85``, we track the relaxation of the pitch-angle
# distribution toward the uniform isotropic steady state ``P(\mu) = 1/2``.

B_unif(x) = SA[1.0, 0.0, 0.0]
dmu_qlt = QLTPitchAngleDiffusion(; D0 = 0.4, p0 = 1.0, gamma = 5 / 3, h0 = 0.2)
param_iso = prepare(;
    B = B_unif, model = FocusedTransport(), dmu = dmu_qlt, dims = 1,
    q = 1.0, m = 1.0, c = Inf,
)

u0_iso = make_state(SA[0.0, 0.0, 0.0], 1.0, 0.85)
snap_times = (0.3, 1.5, 5.0)
mu_snaps = Vector{Vector{Float64}}()
cb_mu = PitchAngleBounds()

for (k, T_end) in enumerate(snap_times)
    Random.seed!(400 + k)
    prob_iso = TransportProblem(u0_iso, (0.0, T_end), param_iso)
    sim_iso = solve(
        EnsembleProblem(prob_iso), EM(), EnsembleThreads();
        dt = 1.0e-2, trajectories = 3_000, callback = cb_mu, save_everystep = false,
    )
    push!(mu_snaps, [get_pitch_angle(s.u[end]) for s in sim_iso.u])
end

@printf "t = %.1f: mean(μ) = %.3f (exact 0.000),  std(μ) = %.3f (exact %.3f)\n" snap_times[end] mean(mu_snaps[end]) std(mu_snaps[end]) (1 / sqrt(3))

# ## Visualization

fig = Figure(; size = (1150, 460), fontsize = 14)

ax1 = Axis(
    fig[1, 1];
    title = "Magnetic Mirroring in B(x) = B₀(1 + x²/L²)",
    xlabel = "Position x",
    ylabel = "Pitch-angle cosine μ",
)
lines!(ax1, xs_mir, mus_mir; color = :royalblue, linewidth = 2.0, label = "FTE trajectory")
vlines!(
    ax1, [-x_mirror, x_mirror];
    color = :crimson, linestyle = :dash, linewidth = 1.6, label = "Exact mirror points ±xₘ",
)
axislegend(ax1; position = :rb, framevisible = false)

ax2 = Axis(
    fig[1, 2];
    title = "QLT Pitch-Angle Isotropization (μ₀ = 0.85)",
    xlabel = "Pitch-angle cosine μ",
    ylabel = "PDF P(μ)",
    limits = ((-1.05, 1.05), (0.0, 1.4)),
)
colors = Makie.wong_colors()
for (k, (T_end, mus)) in enumerate(zip(snap_times, mu_snaps))
    hist!(
        ax2, mus;
        bins = range(-1.0, 1.0; length = 25), normalization = :pdf,
        color = (colors[k], 0.35), strokecolor = colors[k], strokewidth = 1.4,
        label = "t = $T_end",
    )
end
hlines!(
    ax2, [0.5];
    color = :black, linestyle = :dash, linewidth = 2.0, label = "Isotropic P(μ) = 1/2",
)
axislegend(ax2; position = :lt, framevisible = false)

fig = DisplayAs.PNG(fig) #hide
