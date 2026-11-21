# # 1D Convection–Diffusion, Absorbing Boundary & Adiabatic Cooling
#
# This example validates `ParticleTransport.jl` against three fundamental closed-form
# solutions of Parker's transport equation in 1D:
#
# 1. **Uniform convection–diffusion**: Gaussian Green's function in a uniform flow.
# 2. **Diffusion with an absorbing boundary**: Method-of-images solution enforced with
#    [`TerminateOutside`](@ref).
# 3. **Adiabatic heating and cooling**: Exact exponential momentum evolution in a linearly
#    compressing or expanding flow.

import DisplayAs #hide
using ParticleTransport
using StochasticDiffEq: EM
using StaticArrays: SA
using Random: Random, MersenneTwister
using Statistics: mean, var
using Printf
using CairoMakie
CairoMakie.activate!(; type = "png") #hide

# ## 1. Uniform Convection–Diffusion Green's Function
#
# For a constant bulk flow ``u(x) = u_0`` and constant spatial diffusion ``\kappa``, an
# impulsive point source at ``x = 0`` evolves according to
#
# ```math
# n(x, t) = \frac{1}{\sqrt{4\pi \kappa t}}
# \exp\!\left(-\frac{(x - u_0 t)^2}{4\kappa t}\right).
# ```

u_val = 1.5
kappa_cd = 0.25
u_uniform(x) = SA[u_val, 0.0, 0.0]

param_cd = prepare(; u = u_uniform, kappa = IsotropicDiffusion(kappa_cd), dims = 1)
u0_cd = make_state(0.0, 1.0)
n_traj = 12_000

function run_convdiff(T_end; seed = 42)
    Random.seed!(seed)
    prob = TransportProblem(u0_cd, (0.0, T_end), param_cd)
    ep = EnsembleProblem(prob)
    return solve(
        ep, EM(), EnsembleThreads();
        dt = 1.0e-2, trajectories = n_traj, save_everystep = false
    )
end

sim_t1 = run_convdiff(0.5; seed = 42)
sim_t2 = run_convdiff(2.0; seed = 43)

gaussian_pdf(x, t, u0, k) = exp(-(x - u0 * t)^2 / (4 * k * t)) / sqrt(4 * pi * k * t)

x_edges_cd = range(-2.0, 6.0; length = 65)
dens_t1 = spatial_density(sim_t1, x_edges_cd)
dens_t2 = spatial_density(sim_t2, x_edges_cd)

xs_t2 = [s.u[end][1] for s in sim_t2.u]
@printf "t = 2.0: mean(x) = %.3f (exact %.3f),  var(x) = %.3f (exact %.3f)\n" mean(xs_t2) (u_val * 2.0) var(xs_t2) (2 * kappa_cd * 2.0)

# ## 2. Diffusion with an Absorbing Boundary (Method of Images)
#
# On the half-line ``x > 0`` with an absorbing wall at ``x = 0`` (``n(0, t) = 0``), an
# impulsive source at ``x_0 > 0`` has the exact method-of-images solution
#
# ```math
# n(x, t) = \frac{1}{\sqrt{4\pi \kappa t}} \left[
#   \exp\!\left(-\frac{(x - x_0)^2}{4\kappa t}\right)
#   - \exp\!\left(-\frac{(x + x_0)^2}{4\kappa t}\right)
# \right], \qquad x > 0.
# ```
#
# We enforce the absorbing boundary at ``x = 0`` with [`TerminateOutside`](@ref).

x0_abs = 1.0
kappa_abs = 0.5
T_abs = 1.0

param_abs = prepare(; kappa = IsotropicDiffusion(kappa_abs), dims = 1)
u0_abs = make_state(x0_abs, 1.0)
cb_abs = TerminateOutside(((0.0, Inf),))
prob_abs = TransportProblem(u0_abs, (0.0, T_abs), param_abs)

Random.seed!(101)
sim_abs = solve(
    EnsembleProblem(prob_abs), EM(), EnsembleThreads();
    dt = 2.0e-3, trajectories = n_traj, callback = cb_abs,
    save_everystep = false
)

## Keep only trajectories that survived until T_abs (did not hit x = 0)
survivors = [s.u[end] for s in sim_abs.u if s.u[end][1] > 0.0]
x_edges_abs = range(0.0, 5.0; length = 51)
dens_abs = spatial_density(survivors, x_edges_abs)

function images_pdf(x, t, x0, k)
    return (exp(-(x - x0)^2 / (4 * k * t)) - exp(-(x + x0)^2 / (4 * k * t))) /
        sqrt(4 * pi * k * t)
end

surv_frac = length(survivors) / n_traj
dx_grid = 1.0e-3
xs_int = range(0.0, 10.0; step = dx_grid)
surv_exact = sum(images_pdf.(xs_int, T_abs, x0_abs, kappa_abs)) * dx_grid
@printf "Absorbing wall (t = %.1f): survival fraction = %.3f (exact %.3f)\n" T_abs surv_frac surv_exact

# ## 3. Exact Adiabatic Heating & Cooling
#
# For a linear 1D flow ``u(x) = a x`` (``\nabla\cdot\mathbf{u} = a``) with zero spatial
# diffusion, the momentum SDE reduces to ``d(\ln p)/dt = -a/3``, giving the exact solution
#
# ```math
# p(t) = p_0 \exp\!\left(-\frac{a t}{3}\right).
# ```
#
# Compression (``a < 0``) heats the particle, while expansion (``a > 0``) cools it.

p0_ad = 1.0
T_ad = 3.0

sol_heat = solve(
    TransportProblem(
        make_state(1.0, p0_ad), (0.0, T_ad),
        prepare(; u = x -> SA[-0.6 * x[1], 0.0, 0.0], kappa = ZeroDiffusion(), dims = 1)
    ),
    EM(); dt = 1.0e-2
)
sol_cool = solve(
    TransportProblem(
        make_state(1.0, p0_ad), (0.0, T_ad),
        prepare(; u = x -> SA[0.6 * x[1], 0.0, 0.0], kappa = ZeroDiffusion(), dims = 1)
    ),
    EM(); dt = 1.0e-2
)

@printf "Adiabatic heating (a = -0.6): p(T) = %.4f (exact %.4f)\n" get_momentum(sol_heat.u[end]) (p0_ad * exp(0.6 * T_ad / 3))
@printf "Adiabatic cooling (a = +0.6): p(T) = %.4f (exact %.4f)\n" get_momentum(sol_cool.u[end]) (p0_ad * exp(-0.6 * T_ad / 3))

# ## Visualization

fig = Figure(; size = (1350, 430), fontsize = 14)

ax1 = Axis(
    fig[1, 1];
    title = "1D Convection–Diffusion",
    xlabel = "x",
    ylabel = "n(x, t)",
)
xs_fine = range(-2.0, 6.0; length = 300)
barplot!(
    ax1, dens_t1.centers, dens_t1.density ./ n_traj;
    color = (:steelblue, 0.45), strokecolor = :steelblue, strokewidth = 0.6,
    label = "SDE (t = 0.5)"
)
lines!(
    ax1, xs_fine, gaussian_pdf.(xs_fine, 0.5, u_val, kappa_cd);
    color = :navy, linewidth = 2.2, label = "Exact (t = 0.5)"
)
barplot!(
    ax1, dens_t2.centers, dens_t2.density ./ n_traj;
    color = (:coral, 0.45), strokecolor = :coral, strokewidth = 0.6,
    label = "SDE (t = 2.0)"
)
lines!(
    ax1, xs_fine, gaussian_pdf.(xs_fine, 2.0, u_val, kappa_cd);
    color = :darkred, linewidth = 2.2, label = "Exact (t = 2.0)"
)
axislegend(ax1; position = :rt, framevisible = false)

ax2 = Axis(
    fig[1, 2];
    title = "Absorbing Wall at x = 0 (t = 1.0)",
    xlabel = "x",
    ylabel = "n(x, t)",
)
xs_abs = range(0.0, 5.0; length = 300)
barplot!(
    ax2, dens_abs.centers, dens_abs.density ./ n_traj;
    color = (:seagreen, 0.45), strokecolor = :seagreen, strokewidth = 0.6,
    label = "SDE (TerminateOutside)"
)
lines!(
    ax2, xs_abs, images_pdf.(xs_abs, T_abs, x0_abs, kappa_abs);
    color = :darkgreen, linewidth = 2.4, label = "Method of images"
)
axislegend(ax2; position = :rt, framevisible = false)

ax3 = Axis(
    fig[1, 3];
    title = "Adiabatic Heating & Cooling",
    xlabel = "t",
    ylabel = "p(t) / p₀",
)
ts_ad = range(0.0, T_ad; length = 100)
lines!(
    ax3, sol_heat.t, [get_momentum(u) for u in sol_heat.u];
    color = :crimson, linewidth = 2.5, label = "SDE (a = -0.6)"
)
lines!(
    ax3, ts_ad, p0_ad .* exp.(0.6 .* ts_ad ./ 3);
    color = :black, linestyle = :dash, linewidth = 1.8, label = "Exact"
)
lines!(
    ax3, sol_cool.t, [get_momentum(u) for u in sol_cool.u];
    color = :royalblue, linewidth = 2.5, label = "SDE (a = +0.6)"
)
lines!(
    ax3, ts_ad, p0_ad .* exp.(-0.6 .* ts_ad ./ 3);
    color = :black, linestyle = :dash, linewidth = 1.8
)
axislegend(ax3; position = :lt, framevisible = false)

fig = DisplayAs.PNG(fig) #hide
