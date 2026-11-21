# # Magnetic Drift: `TestParticle.jl` vs. `ParticleTransport.jl`
#
# This example directly compares microscopic full-orbit tracing in
# [`TestParticle.jl`](https://github.com/JuliaSpacePhysics/TestParticle.jl) (mirroring its
# `demo_magnetic_drift.jl` example) with the macroscopic isotropic magnetic drift in
# `ParticleTransport.jl`.
#
# ## Physical Connection
#
# In the 2D azimuthal magnetic field
#
# ```math
# \mathbf{B}(x, y) = \frac{B_0}{x^2 + y^2} \begin{bmatrix} y \\ -x \\ 0 \end{bmatrix},
# ```
#
# both the field-line curvature (``R_c = r``) and the magnitude gradient (``|\mathbf{B}| = B_0/r``)
# drive guiding-center drift along ``-\hat{\mathbf{z}}``. For a single non-relativistic
# particle of speed ``v_0`` and pitch angle ``\alpha`` (``\mu = \cos\alpha``), the
# combined curvature and grad-``B`` drift is:
#
# ```math
# v_z(\mu) = -\frac{m v_0^2}{q B_0}\left(\mu^2 + \frac{1 - \mu^2}{2}\right).
# ```
#
# Averaging over an **isotropic** pitch-angle distribution (``\mu \sim \mathcal{U}(-1, 1)``,
# so ``\langle \mu^2 \rangle = \tfrac{1}{3}``) gives
#
# ```math
# \langle v_z \rangle_{\mathrm{iso}}
# = -\frac{2}{3}\frac{m v_0^2}{q B_0}
# = \left[\frac{p v_0}{3 q} \nabla \times \left(\frac{\mathbf{B}}{B^2}\right)\right]_z,
# ```
#
# which is **identically** the magnetic drift velocity ``\mathbf{v}_d`` solved by
# `ParticleTransport.jl`!

import DisplayAs #hide
import TestParticle as TP
import ParticleTransport as PT
using OrdinaryDiffEqVerner: ODEProblem, Vern9, solve
using StochasticDiffEq: EM
using StaticArrays: SA
using LinearAlgebra: normalize
using Statistics: mean
using Printf
using CairoMakie
CairoMakie.activate!(; type = "png") #hide

# ## 1. Field Setup (Identical to `TestParticle.jl`'s `demo_magnetic_drift.jl`)

const Bmag0 = 1.0e-7  # [T ⋅ m]

function curve_B(x)
    r2 = x[1]^2 + x[2]^2
    return SA[x[2] / r2, -x[1] / r2, 0.0] * Bmag0
end

x0 = SA[1.0, 0.0, 0.0]
b0 = normalize(curve_B(x0))  # (0, -1, 0) at x0
eperp = SA[0.0, 0.0, 1.0]
v0 = 1.0                      # [m/s]
tspan = (0.0, 30.0)
q_p, m_p = PT.Proton.q, PT.Proton.m

# ## 2. Microscopic Pitch-Angle Scan with `TestParticle.jl`
#
# We sample 11 values of ``\mu = \cos\alpha`` uniformly across ``[0, 1]`` (by symmetry
# ``v_z(-\mu) = v_z(\mu)``), integrate full proton orbits with `TestParticle.jl` (`Vern9`),
# and extract the guiding-center drift velocity ``v_z(\mu)`` using `TP.get_gc`.

param_tp = TP.prepare(; B = curve_B, species = TP.Proton)
gc_func = xu -> TP.get_gc(xu, param_tp)

μs = collect(range(0.0, 1.0; length = 11))
ts = collect(range(tspan...; length = 300))
A_fit = hcat(ones(length(ts)), ts)

vz_tp = Vector{Float64}(undef, length(μs))
z_gc_all = Matrix{Float64}(undef, length(ts), length(μs))

for (i, μ) in enumerate(μs)
    vpar = v0 * μ
    vperp = v0 * sqrt(max(0.0, 1.0 - μ^2))
    v_init = vpar * b0 + vperp * eperp
    prob_tp = ODEProblem(TP.trace!, [x0..., v_init...], tspan, param_tp)
    sol_tp = solve(prob_tp, Vern9(); abstol = 1.0e-10, reltol = 1.0e-10)
    z_gc = [gc_func(sol_tp(t))[3] for t in ts]
    z_gc_all[:, i] .= z_gc
    vz_tp[i] = (A_fit \ z_gc)[2]
end

## Trapezoidal average over μ ∈ [0, 1] (equivalent to isotropic average over [-1, 1])
w_trap = fill(1.0 / (length(μs) - 1), length(μs))
w_trap[1] *= 0.5
w_trap[end] *= 0.5
vz_tp_iso = sum(w_trap .* vz_tp)
z_gc_iso = vec(z_gc_all * w_trap);

# ## 3. Macroscopic Isotropic Drift with `ParticleTransport.jl`
#
# Now we run `ParticleTransport.jl` in the exact same magnetic field with `drift = true`
# and `kappa = ZeroDiffusion()`. Because ``p = m_p v_0`` (non-relativistic limit),
# `ParticleTransport.jl` computes ``\mathbf{v}_d = \frac{pv}{3q}\nabla\times(\mathbf{B}/B^2)``
# automatically via `ForwardDiff.jl`.

param_pt = PT.prepare(;
    B = curve_B, species = PT.Proton, dims = 3, drift = true, kappa = PT.ZeroDiffusion()
)
p0 = m_p * v0
u0_pt = PT.make_state(x0, p0)
prob_pt = PT.TransportProblem(u0_pt, tspan, param_pt)
sol_pt = PT.solve(prob_pt, EM(); dt = 0.05)

vz_pt = (sol_pt.u[end][3] - sol_pt.u[1][3]) / (tspan[2] - tspan[1])
vz_exact_iso = -(2 / 3) * (m_p * v0^2) / (q_p * Bmag0)

@printf "TestParticle.jl isotropic average:  ⟨v_z⟩ = %.6f m/s\n" vz_tp_iso
@printf "ParticleTransport.jl magnetic drift: v_d,z = %.6f m/s\n" vz_pt
@printf "Exact analytical isotropic drift:    v_d,z = %.6f m/s\n" vz_exact_iso

# ## Visualization

fig = Figure(; size = (1150, 460), fontsize = 14)

ax1 = Axis(
    fig[1, 1];
    title = "Pitch-Angle Dependence vs. Isotropic Drift",
    xlabel = "Pitch-angle cosine |μ| = |cos α|",
    ylabel = "Drift velocity v_z [m/s]",
)
μ_fine = range(0.0, 1.0; length = 100)
vz_analytic(μ) = -(m_p * v0^2 / (q_p * Bmag0)) * (μ^2 + 0.5 * (1.0 - μ^2))

lines!(
    ax1, μ_fine, vz_analytic.(μ_fine);
    color = :gray40, linewidth = 2.0, label = "Analytic GC v_z(μ)"
)
scatter!(
    ax1, μs, vz_tp;
    color = :royalblue, markersize = 10, label = "TestParticle.jl (full orbit)"
)
hlines!(
    ax1, [vz_tp_iso];
    color = :seagreen, linewidth = 2.2, label = "TestParticle.jl ⟨v_z⟩_iso"
)
hlines!(
    ax1, [vz_pt];
    color = :crimson, linestyle = :dash, linewidth = 2.4,
    label = "ParticleTransport.jl v_d,z"
)
axislegend(ax1; position = :lb, framevisible = false)

ax2 = Axis(
    fig[1, 2];
    title = "Isotropic Mean Vertical Displacement z(t)",
    xlabel = "Time t [s]",
    ylabel = "z [m]",
)
lines!(
    ax2, ts, z_gc_iso;
    color = :royalblue, linewidth = 3.0, label = "TestParticle.jl ⟨z_gc(t)⟩_iso"
)
lines!(
    ax2, sol_pt.t, [u[3] for u in sol_pt.u];
    color = :crimson, linestyle = :dash, linewidth = 2.4,
    label = "ParticleTransport.jl z(t)"
)
axislegend(ax2; position = :lb, framevisible = false)

fig = DisplayAs.PNG(fig) #hide
