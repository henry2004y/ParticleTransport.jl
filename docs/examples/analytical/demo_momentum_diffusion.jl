# # Stochastic Acceleration (Second-Order Fermi) & Leaky-Box Spectrum
#
# This example validates `ParticleTransport.jl`'s momentum diffusion
# ([`PowerLawMomentumDiffusion`](@ref)) against two exact analytical solutions for
# hard-sphere / Alfvén-wave turbulence, ``D_{pp}(p) = D_0 p^2``:
#
# 1. **Time-dependent log-normal Green's function** (using ``O(1)``-memory
#    [`streaming_spectrum_ensemble`](@ref)).
# 2. **Steady-state leaky-box power-law spectrum** inside a finite diffusive slab
#    ``[-L, L]`` with absorbing boundaries ([`TerminateOutside`](@ref)).

import DisplayAs #hide
using ParticleTransport
using StochasticDiffEq: EM
using Random: Random, MersenneTwister
using Statistics: mean
using Printf
using CairoMakie
CairoMakie.activate!(; type = "png") #hide

# ## 1. Time-Dependent Log-Normal Green's Function
#
# For pure momentum diffusion with ``D_{pp}(p) = D_0 p^2``, the log-momentum SDE
# (``y = \ln p``) has constant drift ``3 D_0`` and constant diffusion ``\sqrt{2 D_0}``:
#
# ```math
# dy = 3 D_0\,dt + \sqrt{2 D_0}\,dW_p.
# ```
#
# Starting from an impulsive mono-energetic source at ``p_0`` (``y_0 = \ln p_0``), the
# exact pseudo-particle distribution ``F(y, t) = dN / d\ln p`` is a Gaussian in ``y``:
#
# ```math
# F(y, t) = \frac{1}{\sqrt{4\pi D_0 t}}
# \exp\!\left(-\frac{(y - y_0 - 3 D_0 t)^2}{4 D_0 t}\right).
# ```
#
# We accumulate the spectrum directly in ``O(1)`` memory per trajectory using
# [`streaming_spectrum_ensemble`](@ref).

D0_td = 0.2
p0 = 1.0
n_traj = 20_000
param_td = prepare(; dpp = PowerLawMomentumDiffusion(; D0 = D0_td, p0, gamma = 2.0), dims = 1)
u0_td = make_state(0.0, p0)

edges_td = 10.0 .^ range(log10(0.1), log10(200.0); length = 55)
times_td = (0.5, 1.5, 3.0)
specs_td = MomentumSpectrum{Float64}[]

for (k, T_end) in enumerate(times_td)
    Random.seed!(200 + k)
    prob = TransportProblem(u0_td, (0.0, T_end), param_td)
    ep = streaming_spectrum_ensemble(prob, edges_td)
    sim = solve(
        ep, EM(), EnsembleThreads();
        dt = 1.0e-2, trajectories = n_traj, save_everystep = false
    )
    push!(specs_td, spectrum_from_counts(sim.u, edges_td))
end

function exact_dN_dlnp(p, t, p0_val, D0_val)
    dy = log(p / p0_val) - 3 * D0_val * t
    return exp(-dy^2 / (4 * D0_val * t)) / sqrt(4 * pi * D0_val * t)
end

## Verify peak momentum p_peak(t) = p0 * exp(3 D0 t) for each snapshot
for (T_end, spec) in zip(times_td, specs_td)
    p_peak_num = spec.centers[argmax(spec.dN_dlnp)]
    p_peak_exact = p0 * exp(3 * D0_td * T_end)
    @printf "t = %.1f: peak p = %.2f (exact %.2f)\n" T_end p_peak_num p_peak_exact
end

# ## 2. Steady-State Leaky-Box Spectrum in a Diffusive Slab
#
# Now add spatial diffusion ``\kappa`` inside a finite slab ``x \in [-L, L]`` with
# absorbing boundaries at ``x = \pm L``. Particles injected continuously at ``x = 0``,
# ``p = p_0`` escape with a fundamental-mode escape timescale
#
# ```math
# \tau_{\mathrm{esc}} = \frac{4 L^2}{\pi^2 \kappa}.
# ```
#
# Balancing momentum diffusion against leaky-box escape yields power-law wings
# ``f(p) \propto p^{\alpha_\pm}`` below and above ``p_0``:
#
# ```math
# \alpha_\pm = -\frac{3}{2} \pm \sqrt{\frac{9}{4} + \frac{1}{D_0 \tau_{\mathrm{esc}}}}.
# ```

L_slab = 1.0
kappa_lb = 0.2
D0_lb = 0.5
tau_esc = 4 * L_slab^2 / (pi^2 * kappa_lb)
disc = sqrt(2.25 + 1.0 / (D0_lb * tau_esc))
alpha_hi = -1.5 - disc  # high-energy wing (p > p0)
alpha_lo = -1.5 + disc  # low-energy wing (p < p0)

@printf "Leaky-box exponents: α₊ = %.2f (p < p₀),  α₋ = %.2f (p > p₀)\n" alpha_lo alpha_hi

param_lb = prepare(;
    kappa = IsotropicDiffusion(kappa_lb),
    dpp = PowerLawMomentumDiffusion(; D0 = D0_lb, p0, gamma = 2.0),
    dims = 1,
)
T_lb = 25.0
n_lb = 8_000
rng_lb = MersenneTwister(303)
t_injs = rand(rng_lb, n_lb) .* T_lb
prob_lb = TransportProblem(make_state(0.0, p0), (0.0, T_lb), param_lb)
cb_slab = TerminateOutside(((-L_slab, L_slab),))

prob_func_lb = (p_base, ctx) -> remake(p_base; tspan = (t_injs[ctx.sim_id], T_lb))

Random.seed!(303)
sim_lb = solve(
    EnsembleProblem(prob_lb; prob_func = prob_func_lb), EM(), EnsembleThreads();
    dt = 1.0e-2, trajectories = n_lb, callback = cb_slab,
    save_everystep = false
)

## Surviving particles still inside (-L, L) at t = T_lb represent the steady-state distribution
inside_states = [s.u[end] for s in sim_lb.u if abs(s.u[end][1]) < L_slab]
edges_lb = 10.0 .^ range(log10(0.15), log10(30.0); length = 26)
spec_lb = momentum_spectrum(inside_states, edges_lb)

hi_idx = findall(i -> spec_lb.f[i] > 0 && 1.5 <= spec_lb.centers[i] <= 20.0, eachindex(spec_lb.f))
lx_hi = log.(spec_lb.centers[hi_idx])
ly_hi = log.(spec_lb.f[hi_idx])
alpha_hi_est = sum((lx_hi .- mean(lx_hi)) .* (ly_hi .- mean(ly_hi))) / sum((lx_hi .- mean(lx_hi)) .^ 2)
@printf "Leaky-box high-p wing: fitted slope = %.2f (exact %.2f)\n" alpha_hi_est alpha_hi

# ## Visualization

fig = Figure(; size = (1150, 470), fontsize = 14)

ax1 = Axis(
    fig[1, 1];
    title = "Pure Momentum Diffusion: D_pp = D₀ p²",
    xlabel = "Momentum p / p₀",
    ylabel = "(1/N) dN / d(ln p)",
    xscale = log10,
)
colors = Makie.wong_colors()
ps_fine = 10.0 .^ range(log10(0.1), log10(200.0); length = 300)
for (k, (T_end, spec)) in enumerate(zip(times_td, specs_td))
    scatter!(
        ax1, spec.centers, spec.dN_dlnp ./ n_traj;
        color = colors[k], markersize = 6, label = "SDE (t = $T_end)"
    )
    lines!(
        ax1, ps_fine, exact_dN_dlnp.(ps_fine, T_end, p0, D0_td);
        color = colors[k], linewidth = 2.0, label = "Exact (t = $T_end)"
    )
end
axislegend(ax1; position = :rt, framevisible = false, nbanks = 2)

ax2 = Axis(
    fig[1, 2];
    title = "Steady-State Diffusive Slab (Leaky Box)",
    xlabel = "Momentum p / p₀",
    ylabel = "f(p) [arb. units]",
    xscale = log10,
    yscale = log10,
)
valid = findall(>(0.0), spec_lb.f)
scatterlines!(
    ax2, spec_lb.centers[valid], spec_lb.f[valid];
    color = :royalblue, markersize = 7, linewidth = 2.0, label = "SDE (slab)"
)

## Anchor analytical power-law wings on each side of p0 = 1.0
i_hi = findfirst(i -> spec_lb.centers[i] > 1.8 && spec_lb.f[i] > 0, eachindex(spec_lb.f))
ps_hi = 10.0 .^ range(log10(1.2), log10(22.0); length = 60)
lines!(
    ax2, ps_hi, spec_lb.f[i_hi] .* (ps_hi ./ spec_lb.centers[i_hi]) .^ alpha_hi;
    color = :crimson, linestyle = :dash, linewidth = 2.2,
    label = @sprintf("High-p wing: p^(%.2f)", alpha_hi)
)

i_lo = findlast(i -> spec_lb.centers[i] < 0.7 && spec_lb.f[i] > 0, eachindex(spec_lb.f))
ps_lo = 10.0 .^ range(log10(0.18), log10(0.85); length = 40)
lines!(
    ax2, ps_lo, spec_lb.f[i_lo] .* (ps_lo ./ spec_lb.centers[i_lo]) .^ alpha_lo;
    color = :seagreen, linestyle = :dash, linewidth = 2.2,
    label = @sprintf("Low-p wing: p^(+%.2f)", alpha_lo)
)
vlines!(ax2, [p0]; color = :gray, linestyle = :dot, linewidth = 1.4)
axislegend(ax2; position = :lb, framevisible = false)

fig = DisplayAs.PNG(fig) #hide
