# # 3D Anisotropic Diffusion & Curved-Field Itô Drift
#
# In magnetized plasmas, spatial diffusion is strongly anisotropic
# (``\kappa_\parallel \gg \kappa_\perp``) and tied to the local magnetic field direction
# ``\hat{\mathbf{b}}(\mathbf{x})``. This example demonstrates two key analytical tests:
#
# 1. **3D anisotropic diffusion in a tilted uniform field**: Verifies the exact symmetric
#    square root ``\boldsymbol{\sigma} = \sqrt{2\kappa_\perp}\mathbf{I} +
#    (\sqrt{2\kappa_\parallel} - \sqrt{2\kappa_\perp})\hat{\mathbf{b}}\hat{\mathbf{b}}^T``.
# 2. **Pure parallel diffusion in a circular magnetic field**: Demonstrates why the Itô
#    divergence drift ``\nabla\cdot\boldsymbol{\kappa}`` is indispensable — it cancels the
#    outward geometric curvature push so that particles stay on the circular field line
#    ``r(t) = r_0`` and diffuse purely in azimuth ``\theta(t)``.

import DisplayAs #hide
using ParticleTransport
using StochasticDiffEq: EM
using StaticArrays: SA
using LinearAlgebra: dot, norm
using Random: Random
using Statistics: mean, var
using Printf
using CairoMakie
CairoMakie.activate!(; type = "png") #hide

# ## 1. 3D Anisotropic Diffusion in a Tilted Field
#
# Along ``\hat{\mathbf{b}} = (1, 2, 2)/3`` with ``\kappa_\parallel = 1.0`` and
# ``\kappa_\perp = 0.1``, an impulsive point source at the origin spreads into an
# anisotropic 3D Gaussian with variances ``\sigma_\parallel^2 = 2\kappa_\parallel t`` and
# ``\sigma_\perp^2 = 2\kappa_\perp t``.

b_hat = SA[1.0, 2.0, 2.0] / 3.0
B_tilted(x) = 2.0 * b_hat
kpara_3d = 1.0
kperp_3d = 0.1
T_3d = 1.5

param_3d = prepare(; B = B_tilted, kappa = ConstantDiffusion(kpara_3d, kperp_3d), dims = 3)
prob_3d = TransportProblem(make_state(SA[0.0, 0.0, 0.0], 1.0), (0.0, T_3d), param_3d)

Random.seed!(77)
sim_3d = solve(
    EnsembleProblem(prob_3d), EM(), EnsembleThreads();
    dt = 1.0e-2, trajectories = 8_000, save_everystep = false
)

e1 = SA[2.0, -1.0, 0.0] / sqrt(5.0)
x_para = [dot(get_position(s.u[end]), b_hat) for s in sim_3d.u]
x_perp1 = [dot(get_position(s.u[end]), e1) for s in sim_3d.u]

@printf "3D tilted field: var(x∥) = %.3f (exact %.3f),  var(x⊥) = %.3f (exact %.3f)\n" var(x_para) (2 * kpara_3d * T_3d) var(x_perp1) (2 * kperp_3d * T_3d)

# ## 2. Curved Azimuthal Field: Exact Invariance via ``\nabla\cdot\boldsymbol{\kappa}``
#
# Consider a 2D circular magnetic field ``\mathbf{B}(x, y) = (-y, x, 0) / \sqrt{x^2 + y^2}``
# with purely parallel diffusion (``\kappa_\parallel > 0, \kappa_\perp = 0``). Since
# ``\nabla\cdot\hat{\mathbf{b}} = 0`` and ``(\hat{\mathbf{b}}\cdot\nabla)\hat{\mathbf{b}} =
# -\hat{\mathbf{r}}/r``, the Itô drift is
#
# ```math
# \nabla\cdot\boldsymbol{\kappa} = -\frac{\kappa_\parallel}{r}\,\hat{\mathbf{r}}.
# ```
#
# By Itô's lemma, ``d(r^2) = 2\mathbf{x}\cdot d\mathbf{x} + (dx)^2 + (dy)^2 =
# (-2\kappa_\parallel + 2\kappa_\parallel)\,dt = 0``: the inward Itô drift **exactly**
# cancels the outward tangent-step push, keeping ``r(t) = r_0`` invariant while the
# azimuthal angle diffuses as ``\theta(t) \sim \mathcal{N}(0,\; 2\kappa_\parallel t / r_0^2)``.

B_circ(x) = begin
    r = hypot(x[1], x[2])
    SA[-x[2] / r, x[1] / r, 0.0]
end

r0 = 2.0
kpara_c = 0.5
T_c = 1.0

param_circ = prepare(; B = B_circ, kappa = ConstantDiffusion(kpara_c, 0.0), dims = 2)
prob_circ = TransportProblem(make_state(SA[r0, 0.0, 0.0], 1.0), (0.0, T_c), param_circ)

Random.seed!(88)
sim_circ = solve(
    EnsembleProblem(prob_circ), EM(), EnsembleThreads();
    dt = 5.0e-4, trajectories = 5_000, save_everystep = false
)

xs_c = [s.u[end][1] for s in sim_circ.u]
ys_c = [s.u[end][2] for s in sim_circ.u]
rs_c = hypot.(xs_c, ys_c)
thetas_c = atan.(ys_c, xs_c)
sigma_theta = sqrt(2 * kpara_c * T_c) / r0

@printf "Circular field:  mean(r) = %.4f (exact %.4f),  std(θ) = %.3f (exact %.3f)\n" mean(rs_c) r0 sqrt(var(thetas_c)) sigma_theta

# ## Visualization

fig = Figure(; size = (1350, 440), fontsize = 14)

## Panel 1: 3D anisotropic cloud projected onto (x∥, x⊥1)
ax1 = Axis(
    fig[1, 1];
    title = "3D Tilted Field (κ∥ = 1.0, κ⊥ = 0.1)",
    xlabel = "x∥",
    ylabel = "x⊥",
    aspect = DataAspect(),
    limits = ((-6.0, 6.0), (-3.0, 3.0)),
)
scatter!(ax1, x_para[1:2500], x_perp1[1:2500]; color = (:steelblue, 0.25), markersize = 4)
phi = range(0.0, 2pi; length = 200)
for (nsig, ls) in [(1.0, :solid), (2.0, :dash)]
    lines!(
        ax1,
        nsig * sqrt(2 * kpara_3d * T_3d) .* cos.(phi),
        nsig * sqrt(2 * kperp_3d * T_3d) .* sin.(phi);
        color = :crimson, linestyle = ls, linewidth = 2.0,
        label = "$(Int(nsig))σ analytical"
    )
end
axislegend(ax1; position = :rt, framevisible = false)

## Panel 2: Particles diffusing along circular field line
ax2 = Axis(
    fig[1, 2];
    title = "Curved B-Field (κ⊥ = 0): r(t) = r₀ Invariance",
    xlabel = "x",
    ylabel = "y",
    aspect = DataAspect(),
    limits = ((-2.6, 2.6), (-2.6, 2.6)),
)
lines!(
    ax2, r0 .* cos.(phi), r0 .* sin.(phi);
    color = :black, linestyle = :dash, linewidth = 1.8, label = "Field line r = r₀"
)
scatter!(
    ax2, xs_c[1:1200], ys_c[1:1200];
    color = (:royalblue, 0.35), markersize = 5, label = "SDE final positions"
)
scatter!(ax2, [r0], [0.0]; color = :crimson, markersize = 12, label = "Injection (r₀, 0)")
axislegend(ax2; position = :lb, framevisible = false)

## Panel 3: Angular distribution θ(T) vs exact Gaussian
ax3 = Axis(
    fig[1, 3];
    title = "Angular Diffusion Along Field Line",
    xlabel = "Azimuth θ [rad]",
    ylabel = "PDF p(θ, T)",
)
hist!(
    ax3, thetas_c;
    normalization = :pdf, bins = 45,
    color = (:royalblue, 0.45), strokecolor = :royalblue, strokewidth = 0.6,
    label = "SDE"
)
th_grid = range(-2.0, 2.0; length = 200)
lines!(
    ax3, th_grid, exp.(-th_grid .^ 2 ./ (2 * sigma_theta^2)) ./ (sqrt(2 * pi) * sigma_theta);
    color = :crimson, linewidth = 2.4, label = "Exact N(0, 2κ∥T/r₀²)"
)
axislegend(ax3; position = :rt, framevisible = false)

fig = DisplayAs.PNG(fig) #hide
