# # QLT/NLGC Diffusion, Flow-Shear Acceleration & Local Diagnostics
#
# This demo showcases the physics-based transport models and multi-dimensional diagnostic
# tools in `ParticleTransport.jl`:
#
# 1. **QLT & NLGC spatial diffusion** ([`QLTDiffusion`](@ref), [`NLGCDiffusion`](@ref)):
#    Compares the distinct momentum scalings of parallel (``\kappa_\parallel \propto p^{4/3}``)
#    and NLGC perpendicular (``\kappa_\perp \propto p^{10/9}``) diffusion in Kolmogorov turbulence.
# 2. **Flow-shear viscous acceleration** ([`FlowShearMomentumDiffusion`](@ref)) with a
#    selective acceleration region (`acc_region`), [`SplitRule`](@ref) with `pmin_split`,
#    and 2D spatial–energy binning via [`local_distribution`](@ref).

import DisplayAs #hide
using ParticleTransport
using StochasticDiffEq: EM
using StaticArrays: SA
using Random: Random
using Printf
using CairoMakie
CairoMakie.activate!(; type = "png") #hide

# ## 1. QLT Parallel vs. NLGC Perpendicular Diffusion Scaling
#
# In Kolmogorov turbulence (``\gamma = 5/3``) in the resonant regime ``\Omega L_c / v \gg 1``,
# Giacalone & Jokipii (1999) QLT predicts ``\kappa_\parallel \propto p^{4/3}`` (non-relativistic),
# whereas Matthaeus et al. (2003) NLGC predicts ``\kappa_\perp \propto p^{(5-\gamma)/3} = p^{10/9}``.

qlt = QLTDiffusion(;
    sigma2 = 0.1, Lc = 1.0e4, gamma = 5 / 3,
    q = 1.0, m = 1.0, c = Inf,
)
nlgc = NLGCDiffusion(;
    kappa_para = qlt, sigma2_2d = 0.4, L_2d = 1.0e3,
    gamma = 5 / 3, m = 1.0, c = Inf,
)

ps_scan = 10.0 .^ range(-1.0, 2.0; length = 40)
B_ref = SA[1.0, 0.0, 0.0]
x_ref = SA[0.0, 0.0, 0.0]
kparas = [ParticleTransport.diffusion_coefficients(nlgc, B_ref, x_ref, p, 0.0)[1] for p in ps_scan]
kperps = [ParticleTransport.diffusion_coefficients(nlgc, B_ref, x_ref, p, 0.0)[2] for p in ps_scan]

slope_para = log(kparas[end] / kparas[1]) / log(ps_scan[end] / ps_scan[1])
slope_perp = log(kperps[end] / kperps[1]) / log(ps_scan[end] / ps_scan[1])
@printf "Momentum slopes: κ∥ ~ p^(%.3f) (exact 1.333),  κ⊥ ~ p^(%.3f) (exact 1.111)\n" slope_para slope_perp

# ## 2. 2D Flow-Shear Acceleration, Selective Region Mask & Local Map
#
# Consider a 2D incompressible shear flow ``\mathbf{u}(x, y) = (S \tanh(y / L), 0, 0)``
# (``\nabla \cdot \mathbf{u} = 0``), where particles are accelerated solely by viscous
# flow-shear momentum diffusion ``D_{pp} = \Gamma \tau p^2`` inside the shear layer.
# We also restrict acceleration to ``|x| \le 1.5`` using `acc_region`, apply
# [`SplitRule`](@ref) with `pmin_split = 2.0`, and accumulate a 2D spatial–momentum map
# using [`local_distribution`](@ref).

S_shear = 4.0
L_shear = 0.4
u_shear(x) = SA[S_shear * tanh(x[2] / L_shear), 0.0, 0.0]
B_shear(x) = SA[1.0, 0.0, 0.0]

param_sh = prepare(;
    u = u_shear, B = B_shear,
    kappa = ConstantDiffusion(0.15, 0.05),
    dpp = FlowShearMomentumDiffusion(; tau0 = 0.15, p0 = 1.0, alpha = 0.0),
    dims = 2,
    acc_region = ((-1.5, 1.5), (-1.5, 1.5)),
)

prob_sh = TransportProblem(make_state(SA[0.0, 0.0, 0.0], 1.0), (0.0, 1.5), param_sh)
rule_sh = SplitRule(; ratio = 2.0, nsplit = 2, pmin_split = 2.0, max_generations = 4)

Random.seed!(606)
res_sh = solve_with_splitting(
    prob_sh, EM(), EnsembleThreads();
    rule = rule_sh, trajectories = 800, dt = 5.0e-3,
    callback = PeriodicBoundary(((-3.0, 3.0), (-2.0, 2.0))),
)

x_edges = collect(range(-3.0, 3.0; length = 25))
y_edges = collect(range(-2.0, 2.0; length = 17))
p_edges = [0.2, 2.0, 50.0] # low-p (p < 2) vs high-p (p >= 2)
ld = local_distribution(res_sh, (x_edges, y_edges), p_edges)

@printf "Splitting: 800 initial -> %d final particles, total weight = %.1f\n" length(res_sh.states) sum(res_sh.weights)

# ## Visualization

fig = Figure(; size = (1180, 460), fontsize = 14)

ax1 = Axis(
    fig[1, 1];
    title = "QLT (κ∥) & NLGC (κ⊥) Scaling in Kolmogorov Turbulence",
    xlabel = "Momentum p",
    ylabel = "Diffusion coefficient κ",
    xscale = log10,
    yscale = log10,
)
lines!(ax1, ps_scan, kparas; color = :royalblue, linewidth = 2.4, label = "QLT κ∥")
lines!(
    ax1, ps_scan, kparas[1] .* (ps_scan ./ ps_scan[1]) .^ (4 / 3);
    color = :black, linestyle = :dash, linewidth = 1.6, label = "p^(4/3)",
)
lines!(ax1, ps_scan, kperps; color = :crimson, linewidth = 2.4, label = "NLGC κ⊥")
lines!(
    ax1, ps_scan, kperps[1] .* (ps_scan ./ ps_scan[1]) .^ (10 / 9);
    color = :gray, linestyle = :dot, linewidth = 1.8, label = "p^(10/9)",
)
axislegend(ax1; position = :lt, framevisible = false)

ax2 = Axis(
    fig[1, 2];
    title = "2D Local Distribution: High-Energy Particles (p ≥ 2)",
    xlabel = "x",
    ylabel = "y",
)
xc = 0.5 .* (x_edges[1:(end - 1)] .+ x_edges[2:end])
yc = 0.5 .* (y_edges[1:(end - 1)] .+ y_edges[2:end])
hm = heatmap!(ax2, xc, yc, ld.counts[:, :, 2]; colormap = :inferno)
lines!(
    ax2, [-1.5, 1.5, 1.5, -1.5, -1.5], [-1.5, -1.5, 1.5, 1.5, -1.5];
    color = :cyan, linestyle = :dash, linewidth = 1.8, label = "acc_region",
)
Colorbar(fig[1, 3], hm; label = "Weighted count")
axislegend(ax2; position = :rt, framevisible = false, labelcolor = :white)

fig = DisplayAs.PNG(fig) #hide
