# # 1D Diffusive Shock Acceleration (DSA)
#
# Diffusive shock acceleration (first-order Fermi acceleration) is the classic analytical
# benchmark of Parker's transport equation (Axford et al. 1977; Krymsky 1977; Bell 1978;
# Blandford & Ostriker 1978).
#
# Consider a 1D planar shock at ``x = 0`` with upstream speed ``u_1`` (``x < 0``),
# downstream speed ``u_2 = u_1 / r`` (``x > 0``), compression ratio ``r = u_1 / u_2 > 1``,
# and constant spatial diffusion ``\kappa``. For steady mono-energetic injection at
# ``x = 0``, ``p = p_0``, Parker's transport equation has the exact steady-state solution:
#
# ```math
# f(x, p) = f_0 \left(\frac{p}{p_0}\right)^{-q}
# \begin{cases}
#   \exp\!\left(\dfrac{u_1 x}{\kappa}\right), & x < 0, \\
#   1, & x \ge 0,
# \end{cases}
# \qquad q = \frac{3r}{r - 1}.
# ```
#
# In particular:
# - For a strong shock (``r = 4``), the spectral index is ``q = 4`` (``dN/d\ln p \propto p^{-1}``).
# - For a moderate shock (``r = 2.5``), the spectral index is ``q = 5`` (``dN/d\ln p \propto p^{-2}``).
# - Upstream (``x < 0``), the spatial profile decays exponentially with diffusion length
#   ``L_{\mathrm{diff}} = \kappa / u_1``.
#
# Here we solve both shock compression ratios using continuous injection and generational
# particle splitting ([`solve_with_splitting`](@ref)) to resolve the power-law tail over
# more than a decade in momentum.

import DisplayAs #hide
using ParticleTransport
using StochasticDiffEq: EM
using StaticArrays: SA
using Random: Random, MersenneTwister
using Statistics: mean
using Printf
using CairoMakie
CairoMakie.activate!(; type = "png") #hide

# ## Simulation Setup
#
# We smooth the shock discontinuity over a thin ramp width ``L_{\mathrm{sh}} = 0.02 \ll
# L_{\mathrm{diff}} = \kappa / u_1 = 0.2`` using a ``\tanh`` profile so that
# ``\nabla\cdot\mathbf{u}`` is finite and automatically differentiated by `ForwardDiff.jl`.

function run_dsa(r; u1 = 1.0, kappa = 0.2, L_sh = 0.02, T_max = 60.0, n_init = 2000, seed = 2026)
    u2 = u1 / r
    u_shock(x) = SA[0.5 * (u1 + u2) - 0.5 * (u1 - u2) * tanh(x[1] / L_sh), 0.0, 0.0]
    param = prepare(; u = u_shock, kappa = IsotropicDiffusion(kappa), dims = 1)

    u0 = make_state(0.0, 1.0)
    prob = TransportProblem(u0, (0.0, T_max), param)

    rng = MersenneTwister(seed)
    t_injs = rand(rng, n_init) .* T_max
    prob_func = (p_base, ctx) -> remake(p_base; tspan = (t_injs[ctx.sim_id], T_max))

    Random.seed!(seed)
    rule = SplitRule(; ratio = 2.0, nsplit = 2, max_generations = 6, max_particles = 80_000)
    res = solve_with_splitting(
        prob, EM(), EnsembleThreads();
        rule, trajectories = n_init, prob_func, dt = 0.02
    )
    return res
end

res_r4 = run_dsa(4.0; seed = 2026)
res_r25 = run_dsa(2.5; seed = 2027)

println("Strong shock (r = 4.0):   total weight = ", round(sum(res_r4.weights); digits = 1),
    ", final particles = ", length(res_r4.states))
println("Moderate shock (r = 2.5): total weight = ", round(sum(res_r25.weights); digits = 1),
    ", final particles = ", length(res_r25.states))

# ## Spectral Index & Upstream Exponential Precursor

edges_p = 10.0 .^ range(log10(1.2), log10(25.0); length = 16)
spec_r4 = momentum_spectrum(res_r4, edges_p)
spec_r25 = momentum_spectrum(res_r25, edges_p)

function fit_powerlaw_index(spec; pmin = 1.5, pmax = 12.0)
    idx = findall(i -> spec.f[i] > 0 && pmin <= spec.centers[i] <= pmax, eachindex(spec.f))
    lx = log.(spec.centers[idx])
    ly = log.(spec.f[idx])
    mx, my = mean(lx), mean(ly)
    slope = sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
    return -slope, idx
end

q_est_r4, idx_r4 = fit_powerlaw_index(spec_r4; pmin = 1.5, pmax = 15.0)
q_est_r25, idx_r25 = fit_powerlaw_index(spec_r25; pmin = 1.5, pmax = 8.0)

@printf "r = 4.0: fitted q = %.2f  (exact = 4.00)\n" q_est_r4
@printf "r = 2.5: fitted q = %.2f  (exact = 5.00)\n" q_est_r25

## Upstream & downstream spatial profile of accelerated particles (p > 1.5) for r = 4
acc_mask = [get_momentum(s) > 1.5 for s in res_r4.states]
x_edges = range(-1.5, 2.5; length = 41)
prof_x = spatial_density(
    res_r4.states[acc_mask], x_edges; weights = res_r4.weights[acc_mask]
)

## Normalize to the downstream plateau (0.3 < x < 1.5)
down_idx = findall(x -> 0.3 < x < 1.5, prof_x.centers)
norm_down = mean(prof_x.density[down_idx]);

# ## Visualization

fig = Figure(; size = (1150, 480), fontsize = 14)

ax_p = Axis(
    fig[1, 1];
    title = "DSA Momentum Spectrum (with Particle Splitting)",
    xlabel = "Momentum p / p₀",
    ylabel = "f(p) [arb. units]",
    xscale = log10,
    yscale = log10,
)

for (spec, q_exact, q_est, col, lbl) in [
        (spec_r4, 4.0, q_est_r4, :royalblue, "r = 4.0"),
        (spec_r25, 5.0, q_est_r25, :crimson, "r = 2.5"),
    ]
    valid = findall(>(0.0), spec.f)
    scatterlines!(
        ax_p, spec.centers[valid], spec.f[valid];
        color = col, markersize = 8, linewidth = 2.0,
        label = @sprintf("%s: SDE (q ≈ %.2f)", lbl, q_est)
    )
    ## Anchor theoretical power-law at the second valid bin
    p_ref = spec.centers[valid[2]]
    f_ref = spec.f[valid[2]]
    ps_line = 10.0 .^ range(log10(spec.centers[valid[1]]), log10(spec.centers[valid[end]]); length = 50)
    lines!(
        ax_p, ps_line, f_ref .* (ps_line ./ p_ref) .^ (-q_exact);
        color = col, linestyle = :dash, linewidth = 1.8,
        label = @sprintf("%s: exact p^(-%.1f)", lbl, q_exact)
    )
end
axislegend(ax_p; position = :lb, framevisible = false)

ax_x = Axis(
    fig[1, 2];
    title = "Spatial Profile Around Shock (r = 4, p > 1.5 p₀)",
    xlabel = "Position x",
    ylabel = "Normalized density n(x) / n_down",
    yscale = log10,
    limits = ((-1.5, 2.5), (1.0e-3, 2.0)),
)
pos_x = findall(>(0.0), prof_x.density)
scatterlines!(
    ax_x, prof_x.centers[pos_x], prof_x.density[pos_x] ./ norm_down;
    color = :royalblue, markersize = 7, linewidth = 2.0, label = "SDE"
)
xs_th = range(-1.5, 2.5; length = 200)
n_th = [x < 0.0 ? exp(1.0 * x / 0.2) : 1.0 for x in xs_th]
lines!(
    ax_x, xs_th, n_th;
    color = :black, linestyle = :dash, linewidth = 2.2,
    label = "Exact: exp(u₁ x / κ) for x < 0"
)
vlines!(ax_x, [0.0]; color = :gray, linestyle = :dot, linewidth = 1.5)
axislegend(ax_x; position = :rb, framevisible = false)

fig = DisplayAs.PNG(fig) #hide
