# # Parallel MHD Shock: `TestParticle.jl` vs. `ParticleTransport.jl`
#
# In [`TestParticle.jl`'s Shock demo](https://henry2004y.github.io/TestParticle.jl/dev/applications/demo_shock/),
# tracing collisionless protons across a **parallel** MHD shock
# (``\mathbf{B} \parallel \mathbf{u} \parallel \hat{\mathbf{x}}``) reveals a fundamental
# physical limitation: because the motional electric field vanishes identically
# (``\mathbf{E} = -\mathbf{u} \times \mathbf{B} = \mathbf{0}``), collisionless test
# particles simply free-stream across the shock with **zero energy change**.
#
# In real space and astrophysical parallel shocks, self-generated Alfvénic turbulence
# scatters energetic particles in pitch angle on both sides of the shock. This diffusive
# scattering is precisely what Parker's transport equation models via ``\kappa_\parallel``,
# enabling **Diffusive Shock Acceleration (DSA)**!
#
# Below we use the **exact MHD parallel shock parameters** from `TestParticle.jl`'s
# `demo_shock.jl` and run both solvers side by side.

import DisplayAs #hide
import TestParticle as TP
import ParticleTransport as PT
using OrdinaryDiffEqVerner: ODEProblem, Vern7, solve
using StochasticDiffEq: EM
using StaticArrays: SA
using LinearAlgebra: norm, ×
using Random: Random, MersenneTwister
using Statistics: mean
using Printf
using CairoMakie
CairoMakie.activate!(; type = "png") #hide

# ## 1. Parallel MHD Shock Parameters (from `TestParticle.jl/docs/examples/applications/demo_shock.jl`)
#
# Upstream lies at ``x > 0`` (plasma flows in the ``-x`` direction toward the shock at
# ``x = 0``), and downstream lies at ``x < 0``:
# - Upstream speed: ``|V_1| = 545.15\,\mathrm{km/s}``
# - Downstream speed: ``|V_2| = 149.92\,\mathrm{km/s}``
# - Compression ratio: ``r = V_1 / V_2 \approx 3.636``
# - Magnetic field: ``\mathbf{B} = (5\,\mathrm{nT}, 0, 0)`` (uniform across the parallel shock)
# - Ideal MHD electric field: ``\mathbf{E} = -\mathbf{V} \times \mathbf{B} = \mathbf{0}``

const V1_mag = 545.1484121835928e3  # [m/s] upstream speed
const V2_mag = 149.91581335048804e3 # [m/s] downstream speed
const r_shock = V1_mag / V2_mag     # compression ratio ≈ 3.636
const q_theory = 3 * r_shock / (r_shock - 1)
const B0_val = 5.0e-9               # [T]

@printf "Parallel shock: r = %.3f,  exact DSA spectral index q = 3r/(r-1) = %.3f\n" r_shock q_theory

# ## 2. Collisionless Orbit Tracing with `TestParticle.jl`
#
# We trace a 10 keV proton launched upstream at ``x = +5000\,\mathrm{km}`` heading toward
# the shock at ``x = 0``. Because ``\mathbf{E} = \mathbf{0}`` everywhere, the Lorentz force
# ``q(\mathbf{v}\times\mathbf{B})`` does zero work and the kinetic energy is strictly
# conserved across the shock.

B_para(x) = SA[B0_val, 0.0, 0.0]
E_para(x) = SA[0.0, 0.0, 0.0]

param_tp = TP.prepare(; E = E_para, B = B_para, species = TP.Proton)

E0_keV = 10.0
v0_mag = sqrt(2 * E0_keV * 1.0e3 * PT.eV / PT.Proton.m)
## Pitch angle 135° (v_x < 0 so the proton travels from upstream x > 0 to downstream x < 0)
v0_tp = SA[v0_mag * cosd(135.0), v0_mag * sind(135.0), 0.0]
u0_tp = [5_000.0e3, 0.0, 0.0, v0_tp...]

sol_tp = solve(ODEProblem(TP.trace!, u0_tp, (0.0, 12.0), param_tp), Vern7(); saveat = 0.05)
ke_tp = [0.5 * PT.Proton.m * (u[4]^2 + u[5]^2 + u[6]^2) / (1.0e3 * PT.eV) for u in sol_tp.u]

@printf "TestParticle.jl: E_init = %.3f keV,  E_final = %.3f keV  (ΔE/E₀ = %.1e)\n" ke_tp[1] ke_tp[end] ((ke_tp[end] - ke_tp[1]) / ke_tp[1])

# ## 3. Diffusive Shock Acceleration with `ParticleTransport.jl`
#
# Now we model the **same** parallel shock velocity jump in `ParticleTransport.jl` with a
# finite parallel diffusion coefficient ``\kappa_\parallel = 2 \times 10^{11}\,\mathrm{m^2/s}``
# (diffusion length ``L_{\mathrm{diff}} = \kappa_\parallel / |V_1| \approx 367\,\mathrm{km}``)
# and shock ramp thickness ``L_{\mathrm{sh}} = 15\,\mathrm{km}``.
#
# Working in normalized shock units (length ``L_0 = L_{\mathrm{diff}}``, velocity
# ``U_0 = |V_1|``, time ``T_0 = L_{\mathrm{diff}} / |V_1| \approx 0.67\,\mathrm{s}``,
# momentum ``p_0`` corresponding to ``E_0 = 10\,\mathrm{keV}``) keeps all numerical values
# ``O(1)``.

u1_norm = 1.0
u2_norm = 1.0 / r_shock
L_sh_norm = 15.0e3 / (2.0e11 / V1_mag) # ≈ 0.041 L_diff
kappa_norm = 1.0

## Upstream at x > 0 (flow along -x), downstream at x < 0
u_shock_norm(x) = SA[-(0.5 * (u1_norm + u2_norm) + 0.5 * (u1_norm - u2_norm) * tanh(x[1] / L_sh_norm)), 0.0, 0.0]

param_pt = PT.prepare(;
    u = u_shock_norm,
    B = B_para,
    kappa = PT.ConstantDiffusion(kappa_norm, 0.0),
    dims = 1,
)

## Trace 4 individual stochastic trajectories starting at the shock (x = 0, p = p0)
T_sample = 20.0  # ≈ 13.5 s in physical time
prob_sample = PT.TransportProblem(PT.make_state(0.0, 1.0), (0.0, T_sample), param_pt)
sample_sols = [
    PT.solve(prob_sample, EM(); dt = 0.02, seed = 500 + k) for k in 1:4
]

## Steady-state ensemble with continuous injection + particle splitting
T_max = 120.0
n_init = 2_000
rng = MersenneTwister(2026)
t_injs = rand(rng, n_init) .* T_max
prob_ens = PT.TransportProblem(PT.make_state(0.0, 1.0), (0.0, T_max), param_pt)
prob_func = (p_base, ctx) -> PT.remake(p_base; tspan = (t_injs[ctx.sim_id], T_max))

Random.seed!(2026)
rule = PT.SplitRule(; ratio = 2.0, nsplit = 2, max_generations = 6, max_particles = 80_000)
res_pt = PT.solve_with_splitting(
    prob_ens, EM(), PT.EnsembleThreads();
    rule, trajectories = n_init, prob_func, dt = 0.02
)

edges_p = 10.0 .^ range(log10(1.2), log10(20.0); length = 15)
spec_pt = PT.momentum_spectrum(res_pt, edges_p)

## Fit power-law slope over [1.5, 10] p0
fit_idx = findall(i -> spec_pt.f[i] > 0 && 1.5 <= spec_pt.centers[i] <= 10.0, eachindex(spec_pt.f))
lx = log.(spec_pt.centers[fit_idx])
ly = log.(spec_pt.f[fit_idx])
q_fitted = -(sum((lx .- mean(lx)) .* (ly .- mean(ly))) / sum((lx .- mean(lx)) .^ 2))
@printf "ParticleTransport.jl: fitted DSA index q = %.2f  (exact = %.2f)\n" q_fitted q_theory

# ## Visualization

T0_sec = (2.0e11 / V1_mag) / V1_mag  # physical time unit [s]

fig = Figure(; size = (1180, 470), fontsize = 14)

ax1 = Axis(
    fig[1, 1];
    title = "Proton Energy Evolution Across Parallel Shock",
    xlabel = "Time t [s]",
    ylabel = "Kinetic energy E / E₀",
    yscale = log10,
)
lines!(
    ax1, sol_tp.t, ke_tp ./ E0_keV;
    color = :black, linewidth = 3.0, linestyle = :dash,
    label = "TestParticle.jl (collisionless, E = 0)"
)
palette = Makie.wong_colors()
for (k, s) in enumerate(sample_sols)
    ## Non-relativistic E/E0 = (p/p0)^2
    E_ratio = [PT.get_momentum(u)^2 for u in s.u]
    lines!(
        ax1, s.t .* T0_sec, E_ratio;
        color = palette[k], linewidth = 1.8,
        label = k == 1 ? "ParticleTransport.jl (sample SDEs)" : nothing
    )
end
axislegend(ax1; position = :lt, framevisible = false)

ax2 = Axis(
    fig[1, 2];
    title = "Steady-State Momentum Spectrum (r = 3.64)",
    xlabel = "Momentum p / p₀",
    ylabel = "f(p) [arb. units]",
    xscale = log10,
    yscale = log10,
)
valid = findall(>(0.0), spec_pt.f)
scatterlines!(
    ax2, spec_pt.centers[valid], spec_pt.f[valid];
    color = :royalblue, markersize = 8, linewidth = 2.0,
    label = @sprintf("ParticleTransport.jl (q ≈ %.2f)", q_fitted)
)
p_ref = spec_pt.centers[valid[2]]
f_ref = spec_pt.f[valid[2]]
ps_line = 10.0 .^ range(log10(spec_pt.centers[valid[1]]), log10(spec_pt.centers[valid[end]]); length = 50)
lines!(
    ax2, ps_line, f_ref .* (ps_line ./ p_ref) .^ (-q_theory);
    color = :crimson, linestyle = :dash, linewidth = 2.2,
    label = @sprintf("Exact DSA: p^(-%.2f)", q_theory)
)
vlines!(
    ax2, [1.0];
    color = :black, linestyle = :dot, linewidth = 2.0,
    label = "TestParticle.jl (unaccelerated p = p₀)"
)
axislegend(ax2; position = :lb, framevisible = false)

fig = DisplayAs.PNG(fig) #hide
