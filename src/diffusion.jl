# Spatial diffusion tensor (κ_∥, κ_⊥), momentum diffusion (D_pp),
# and pitch-angle diffusion (D_μμ) models.

# Zero-dependency Lanczos approximation for Γ(x), x > 0 (accurate to < 1e-14).
@inline function _gamma(z::T) where {T <: AbstractFloat}
    g = T(7)
    p = (
        T(0.99999999999980993),
        T(676.5203681218851),
        T(-1259.1392167224028),
        T(771.32342877765313),
        T(-176.61502916214059),
        T(12.507343278686905),
        T(-0.13857109526572012),
        T(9.9843695780195716e-6),
        T(1.5056327351493116e-7),
    )
    if z < T(0.5)
        return T(pi) / (sin(T(pi) * z) * _gamma(one(T) - z))
    end
    zm = z - one(T)
    x = p[1]
    @inbounds for i in 2:9
        x += p[i] / (zm + T(i - 1))
    end
    t = zm + g + T(0.5)
    return sqrt(2 * T(pi)) * t^(zm + T(0.5)) * exp(-t) * x
end

@inline _eval_scalar_field(val::Real, x, t) = val
@inline _eval_scalar_field(f, x, t) = f(x, t)

@inline _grad_scalar_field(val::Real, x, t) = zero(SVector{3, float(typeof(val))})
@inline function _grad_scalar_field(f, x, t)
    return SVector{3}(ForwardDiff.gradient(r -> f(r, t), SVector{3}(x)))
end

# --- Spatial diffusion models ---

"""
    ZeroDiffusion <: AbstractSpatialDiffusion

Zero spatial diffusion (`κ_∥ = κ_⊥ = 0`).
"""
struct ZeroDiffusion <: AbstractSpatialDiffusion end

"""
    IsotropicDiffusion(kappa) <: AbstractSpatialDiffusion

Constant isotropic spatial diffusion (`κ_∥ = κ_⊥ = kappa`).
"""
struct IsotropicDiffusion{T <: AbstractFloat} <: AbstractSpatialDiffusion
    kappa::T
    function IsotropicDiffusion(kappa::Real)
        kappa >= zero(kappa) || throw(ArgumentError("kappa must be non-negative; got $kappa."))
        kf = float(kappa)
        return new{typeof(kf)}(kf)
    end
end

"""
    ConstantDiffusion(kpara, kperp) <: AbstractSpatialDiffusion

Spatially uniform anisotropic diffusion with parallel coefficient `kpara` and
perpendicular coefficient `kperp`.
"""
struct ConstantDiffusion{T <: AbstractFloat} <: AbstractSpatialDiffusion
    kpara::T
    kperp::T
    function ConstantDiffusion(kpara::Real, kperp::Real)
        kpara >= zero(kpara) || throw(ArgumentError("kpara must be non-negative; got $kpara."))
        kperp >= zero(kperp) || throw(ArgumentError("kperp must be non-negative; got $kperp."))
        kp, kr = promote(float(kpara), float(kperp))
        return new{typeof(kp)}(kp, kr)
    end
end

"""
    PowerLawDiffusion(; kappa0, p0 = 1.0, alpha = 0.0, B0 = 1.0, beta = 0.0, eta = 0.0)

Power-law spatial diffusion model:
```math
\\kappa_\\parallel(\\mathbf{x}, p) = \\kappa_0 (p / p_0)^\\alpha (B_0 / |\\mathbf{B}|)^\\beta,
\\qquad \\kappa_\\perp = \\eta \\kappa_\\parallel.
```
"""
struct PowerLawDiffusion{T <: AbstractFloat} <: AbstractSpatialDiffusion
    kappa0::T
    p0::T
    alpha::T
    B0::T
    beta::T
    eta::T
    function PowerLawDiffusion(;
            kappa0::Real, p0::Real = 1.0, alpha::Real = 0.0,
            B0::Real = 1.0, beta::Real = 0.0, eta::Real = 0.0
        )
        kappa0 >= zero(kappa0) || throw(ArgumentError("kappa0 must be non-negative."))
        p0 > zero(p0) || throw(ArgumentError("p0 must be positive."))
        B0 > zero(B0) || throw(ArgumentError("B0 must be positive."))
        eta >= zero(eta) || throw(ArgumentError("eta must be non-negative."))
        vals = promote(float(kappa0), float(p0), float(alpha), float(B0), float(beta), float(eta))
        return new{typeof(vals[1])}(vals...)
    end
end

"""
    QLTDiffusion(; sigma2, Lc, gamma = 5/3, eta = 0.0, species = Proton, q = nothing, m = nothing, c = C_SI)

Quasi-linear theory (Giacalone & Jokipii 1999) parallel diffusion model:
```math
\\kappa_\\parallel(v, B) =
\\frac{v^3 \\csc(\\pi / \\gamma)}{4 \\Omega^2 \\sigma^2 \\gamma L_c}
\\left[1 + \\frac{8}{(2 - \\gamma)(4 - \\gamma)} \\left(\\frac{\\Omega L_c}{v}\\right)^\\gamma\\right],
\\qquad \\kappa_\\perp = \\eta \\kappa_\\parallel,
```
where `Ω = |q| |B| v / p` is the relativistic gyrofrequency. `sigma2` (`⟨δB²⟩/B²`) and
`Lc` (turbulence correlation length) may be positive scalars or callables `(x, t) -> val`.
"""
struct QLTDiffusion{S, L, T <: AbstractFloat} <: AbstractSpatialDiffusion
    sigma2::S
    Lc::L
    gamma::T
    eta::T
    prefac::T
    res_coeff::T
    absq::T
    m::T
    c::T
    function QLTDiffusion{S, L, T}(
            sigma2::S, Lc::L, gamma::T, eta::T, prefac::T, res_coeff::T,
            absq::T, m::T, c::T
        ) where {S, L, T <: AbstractFloat}
        return new{S, L, T}(sigma2, Lc, gamma, eta, prefac, res_coeff, absq, m, c)
    end
    function QLTDiffusion(;
            sigma2, Lc, gamma::Real = 5 / 3, eta::Real = 0.0,
            species::Particle = Proton, q = nothing, m = nothing, c::Real = C_SI
        )
        one(gamma) < gamma < 2 * one(gamma) ||
            throw(ArgumentError("`gamma` must satisfy 1 < gamma < 2; got $gamma."))
        eta >= zero(eta) || throw(ArgumentError("`eta` must be non-negative; got $eta."))
        sigma2 isa Real && (sigma2 > zero(sigma2) || throw(ArgumentError("`sigma2` must be positive.")))
        Lc isa Real && (Lc > zero(Lc) || throw(ArgumentError("`Lc` must be positive.")))
        q_val = isnothing(q) ? abs(species.q) : abs(q)
        m_val = isnothing(m) ? species.m : m
        T = promote_type(Float64, typeof(float(gamma)), typeof(float(eta)), typeof(float(q_val)), typeof(float(m_val)), typeof(float(c)))
        g = T(gamma)
        prefac = inv(4 * g * sin(T(pi) / g))
        res_coeff = T(8) / ((T(2) - g) * (T(4) - g))
        s2_v = sigma2 isa Real ? T(sigma2) : sigma2
        lc_v = Lc isa Real ? T(Lc) : Lc
        return new{typeof(s2_v), typeof(lc_v), T}(
            s2_v, lc_v, g, T(eta), prefac, res_coeff, T(q_val), T(m_val), T(c)
        )
    end
end

"""
    NLGCDiffusion(; kappa_para, sigma2_2d, L_2d, gamma = 5/3, alpha = sqrt(1/3), species = Proton, m = nothing, c = C_SI)

Nonlinear Guiding Center (NLGC; Matthaeus et al. 2003) spatial diffusion model pairing
any parallel diffusion model `kappa_para::AbstractSpatialDiffusion` with perpendicular
diffusion
```math
\\lambda_\\perp = \\lambda_\\parallel^{1/3}
\\left(\\alpha^2 \\sqrt{3\\pi} \\frac{2\\nu - 1}{\\nu}
\\frac{\\Gamma(\\nu)}{\\Gamma(\\nu - 1/2)} \\sigma_{\\mathrm{2D}}^2 l_{\\mathrm{2D}}\\right)^{2/3},
\\qquad \\kappa_\\perp = \\frac{v}{3} \\lambda_\\perp, \\quad \\nu = \\frac{\\gamma}{2}.
```
`sigma2_2d` and `L_2d` may be positive scalars or callables `(x, t) -> val`.
"""
struct NLGCDiffusion{K <: AbstractSpatialDiffusion, S, L, T <: AbstractFloat} <:
       AbstractSpatialDiffusion
    kappa_para::K
    sigma2_2d::S
    L_2d::L
    c_nlgc::T
    m::T
    c::T
    function NLGCDiffusion{K, S, L, T}(
            kappa_para::K, sigma2_2d::S, L_2d::L, c_nlgc::T, m::T, c::T
        ) where {K <: AbstractSpatialDiffusion, S, L, T <: AbstractFloat}
        return new{K, S, L, T}(kappa_para, sigma2_2d, L_2d, c_nlgc, m, c)
    end
    function NLGCDiffusion(;
            kappa_para::AbstractSpatialDiffusion, sigma2_2d, L_2d,
            gamma::Real = 5 / 3, alpha::Real = sqrt(1 / 3),
            species::Particle = Proton, m = nothing, c::Real = C_SI
        )
        gamma > one(gamma) || throw(ArgumentError("`gamma` must be > 1; got $gamma."))
        alpha > zero(alpha) || throw(ArgumentError("`alpha` must be positive; got $alpha."))
        sigma2_2d isa Real &&
            (sigma2_2d > zero(sigma2_2d) || throw(ArgumentError("`sigma2_2d` must be positive.")))
        L_2d isa Real && (L_2d > zero(L_2d) || throw(ArgumentError("`L_2d` must be positive.")))
        m_val = isnothing(m) ? species.m : m
        T = promote_type(
            Float64, typeof(float(gamma)), typeof(float(alpha)),
            typeof(float(m_val)), typeof(float(c)),
        )
        ν = T(gamma) / 2
        c_nlgc = T(alpha)^2 * sqrt(3 * T(pi)) * ((2 * ν - one(T)) / ν) *
            (_gamma(ν) / _gamma(ν - T(0.5)))
        s2_v = sigma2_2d isa Real ? T(sigma2_2d) : sigma2_2d
        l2_v = L_2d isa Real ? T(L_2d) : L_2d
        return new{typeof(kappa_para), typeof(s2_v), typeof(l2_v), T}(
            kappa_para, s2_v, l2_v, c_nlgc, T(m_val), T(c)
        )
    end
end

"""
    CustomDiffusion(f) <: AbstractSpatialDiffusion

Wrap a user callable `f(x, p, t) -> (kpara, kperp)` as a spatial diffusion model.
"""
struct CustomDiffusion{F} <: AbstractSpatialDiffusion
    f::F
end

Adapt.adapt_structure(
    to, d::Union{ZeroDiffusion, IsotropicDiffusion, ConstantDiffusion, PowerLawDiffusion}
) = d
Adapt.adapt_structure(to, d::CustomDiffusion) = CustomDiffusion(Adapt.adapt(to, d.f))
function Adapt.adapt_structure(to, d::QLTDiffusion)
    s2 = Adapt.adapt(to, d.sigma2)
    lc = Adapt.adapt(to, d.Lc)
    return QLTDiffusion{typeof(s2), typeof(lc), typeof(d.gamma)}(
        s2, lc, d.gamma, d.eta, d.prefac, d.res_coeff, d.absq, d.m, d.c
    )
end
function Adapt.adapt_structure(to, d::NLGCDiffusion)
    kp = Adapt.adapt(to, d.kappa_para)
    s2 = Adapt.adapt(to, d.sigma2_2d)
    l2 = Adapt.adapt(to, d.L_2d)
    return NLGCDiffusion{typeof(kp), typeof(s2), typeof(l2), typeof(d.c_nlgc)}(
        kp, s2, l2, d.c_nlgc, d.m, d.c
    )
end

"""
    diffusion_coefficients(model::AbstractSpatialDiffusion, B_vec, x, p, t)

Return `(kpara, kperp)` at `(x, p, t)` given local magnetic field vector `B_vec`.
"""
@inline diffusion_coefficients(::ZeroDiffusion, B_vec, x, p, t) = (zero(p), zero(p))
@inline diffusion_coefficients(d::IsotropicDiffusion, B_vec, x, p, t) = (d.kappa, d.kappa)
@inline diffusion_coefficients(d::ConstantDiffusion, B_vec, x, p, t) = (d.kpara, d.kperp)

@inline function diffusion_coefficients(d::PowerLawDiffusion, B_vec, x, p, t)
    kp = d.kappa0 * (p / d.p0)^d.alpha
    if d.beta != zero(d.beta)
        Bmag = norm(B_vec)
        if Bmag > zero(Bmag)
            kp *= (d.B0 / Bmag)^d.beta
        end
    end
    return (kp, d.eta * kp)
end

@inline function diffusion_coefficients(d::QLTDiffusion, B_vec, x, p, t)
    Bmag = norm(B_vec)
    T = promote_type(typeof(p), eltype(B_vec), typeof(d.gamma))
    if Bmag <= zero(Bmag)
        return (zero(T), zero(T))
    end
    v = momentum2velocity(p, d.m, d.c)
    # Relativistic gyrofrequency Ω = |q| B v / p => Ω / v = |q| B / p = 1 / R_L
    inv_RL = (d.absq * Bmag) / p
    s2 = _eval_scalar_field(d.sigma2, x, t)
    lc = _eval_scalar_field(d.Lc, x, t)
    x_res = inv_RL * lc
    kp = (d.prefac * v / (inv_RL * inv_RL * s2 * lc)) * (one(T) + d.res_coeff * x_res^d.gamma)
    return (T(kp), T(d.eta * kp))
end

@inline function diffusion_coefficients(d::NLGCDiffusion, B_vec, x, p, t)
    kp, _ = diffusion_coefficients(d.kappa_para, B_vec, x, p, t)
    T = promote_type(typeof(kp), typeof(p), typeof(d.c_nlgc))
    if kp <= zero(kp)
        return (zero(T), zero(T))
    end
    v = momentum2velocity(p, d.m, d.c)
    λ_para = 3 * kp / v
    s2 = _eval_scalar_field(d.sigma2_2d, x, t)
    l2 = _eval_scalar_field(d.L_2d, x, t)
    λ_perp = cbrt(λ_para) * cbrt((d.c_nlgc * s2 * l2)^2)
    kr = (v / 3) * λ_perp
    return (T(kp), T(kr))
end

@inline function diffusion_coefficients(d::CustomDiffusion, B_vec, x, p, t)
    return d.f(x, p, t)
end

"""
    diffusion_tensor(model, B_field, x, p, t)

Compute the 3×3 symmetric spatial diffusion tensor
`κ_ij = κ_⊥ δ_ij + (κ_∥ - κ_⊥) b_i b_j`.
"""
@inline function diffusion_tensor(model::AbstractSpatialDiffusion, B_field, x, p, t)
    B = B_field(x, t)
    kpara, kperp = diffusion_coefficients(model, B, x, p, t)
    Bmag = norm(B)
    T = promote_type(typeof(kpara), typeof(kperp), eltype(B))
    I3 = SMatrix{3, 3, T}(1, 0, 0, 0, 1, 0, 0, 0, 1)
    if Bmag == zero(Bmag)
        return T(kpara) * I3
    end
    b = B / Bmag
    return T(kperp) * I3 + T(kpara - kperp) * (b * b')
end

"""
    diffusion_sqrt(model, B_field, x, p, t)

Compute the exact symmetric positive-semidefinite square root `σ` such that `σ * σ' = 2 κ`:
`σ = √(2 κ_⊥) I + (√(2 κ_∥) - √(2 κ_⊥)) b b'`.
"""
@inline function diffusion_sqrt(::ZeroDiffusion, B_field, x::AbstractVector{T}, p, t) where {T}
    return zero(SMatrix{3, 3, promote_type(T, typeof(p))})
end

@inline function diffusion_sqrt(model::IsotropicDiffusion, B_field, x, p, t)
    s = sqrt(2 * model.kappa)
    T = typeof(s)
    return SMatrix{3, 3, T}(s, 0, 0, 0, s, 0, 0, 0, s)
end

@inline function diffusion_sqrt(model::AbstractSpatialDiffusion, B_field, x, p, t)
    B = B_field(x, t)
    kpara, kperp = diffusion_coefficients(model, B, x, p, t)
    s_para = sqrt(2 * max(kpara, zero(kpara)))
    s_perp = sqrt(2 * max(kperp, zero(kperp)))
    T = promote_type(typeof(s_para), typeof(s_perp), eltype(B))
    I3 = SMatrix{3, 3, T}(1, 0, 0, 0, 1, 0, 0, 0, 1)
    Bmag = norm(B)
    if Bmag == zero(Bmag)
        return T(s_para) * I3
    end
    b = B / Bmag
    return T(s_perp) * I3 + T(s_para - s_perp) * (b * b')
end

"""
    perp_diffusion_sqrt(model, B_field, x, p, t)

Compute the exact symmetric square root `σ_⊥ = √(2 κ_⊥) (I - b b')` of the cross-field
diffusion tensor `2 κ_⊥ (I - b b')` for focused transport.
"""
@inline function perp_diffusion_sqrt(
        ::ZeroDiffusion, B_field, x::AbstractVector{T}, p, t
    ) where {T}
    return zero(SMatrix{3, 3, promote_type(T, typeof(p))})
end

@inline function perp_diffusion_sqrt(model::AbstractSpatialDiffusion, B_field, x, p, t)
    B = B_field(x, t)
    _, kperp = diffusion_coefficients(model, B, x, p, t)
    s_perp = sqrt(2 * max(kperp, zero(kperp)))
    T = promote_type(typeof(s_perp), eltype(B))
    I3 = SMatrix{3, 3, T}(1, 0, 0, 0, 1, 0, 0, 0, 1)
    Bmag = norm(B)
    if Bmag == zero(Bmag)
        return T(s_perp) * I3
    end
    b = B / Bmag
    return T(s_perp) * (I3 - b * b')
end

@inline function _grad_diffusion_coeffs(
        ::Union{ZeroDiffusion, IsotropicDiffusion, ConstantDiffusion},
        B, J_B, b, Bmag, kpara, kperp, x, p, t
    )
    z = zero(SVector{3, promote_type(typeof(kpara), typeof(kperp), eltype(B))})
    return (z, z)
end

@inline function _grad_diffusion_coeffs(
        d::PowerLawDiffusion, B, J_B, b, Bmag, kpara, kperp, x, p, t
    )
    T = promote_type(typeof(kpara), typeof(kperp), eltype(B))
    if d.beta == zero(d.beta) || Bmag == zero(Bmag)
        z = zero(SVector{3, T})
        return (z, z)
    end
    grad_Bmag = J_B' * b
    grad_kp = (-d.beta * kpara / Bmag) * grad_Bmag
    return (grad_kp, d.eta * grad_kp)
end

@inline function _grad_diffusion_coeffs(
        d::QLTDiffusion, B, J_B, b, Bmag, kpara, kperp, x, p, t
    )
    T = promote_type(typeof(kpara), typeof(kperp), eltype(B))
    if Bmag <= zero(Bmag) || kpara <= zero(kpara)
        z = zero(SVector{3, T})
        return (z, z)
    end
    inv_RL = (d.absq * Bmag) / p
    s2 = _eval_scalar_field(d.sigma2, x, t)
    lc = _eval_scalar_field(d.Lc, x, t)
    x_res = inv_RL * lc
    res_term = d.res_coeff * x_res^d.gamma
    denom = one(T) + res_term
    # ∂ln κ_∥ / ∂ln B = -2 + γ * res_term / (1 + res_term)
    # ∂ln κ_∥ / ∂ln L_c = -1 + γ * res_term / (1 + res_term)
    # ∂ln κ_∥ / ∂ln σ² = -1
    dln_dB = (-2 + d.gamma * res_term / denom) / Bmag
    dln_dLc = (-1 + d.gamma * res_term / denom) / lc
    grad_Bmag = J_B' * b
    grad_s2 = _grad_scalar_field(d.sigma2, x, t)
    grad_lc = _grad_scalar_field(d.Lc, x, t)
    grad_kp = kpara * (dln_dB * grad_Bmag - (inv(s2)) * grad_s2 + dln_dLc * grad_lc)
    return (grad_kp, d.eta * grad_kp)
end

@inline function _grad_diffusion_coeffs(
        d::NLGCDiffusion, B, J_B, b, Bmag, kpara, kperp, x, p, t
    )
    T = promote_type(typeof(kpara), typeof(kperp), eltype(B))
    g_kp, _ = _grad_diffusion_coeffs(d.kappa_para, B, J_B, b, Bmag, kpara, zero(T), x, p, t)
    if kpara <= zero(kpara) || kperp <= zero(kperp)
        return (g_kp, zero(SVector{3, T}))
    end
    s2 = _eval_scalar_field(d.sigma2_2d, x, t)
    l2 = _eval_scalar_field(d.L_2d, x, t)
    g_s2 = _grad_scalar_field(d.sigma2_2d, x, t)
    g_l2 = _grad_scalar_field(d.L_2d, x, t)
    g_kr = kperp * (
        (g_kp / (3 * kpara)) + (2 / (3 * s2)) * g_s2 + (2 / (3 * l2)) * g_l2
    )
    return (g_kp, g_kr)
end

@inline function _grad_diffusion_coeffs(
        d::CustomDiffusion, B, J_B, b, Bmag, kpara, kperp, x, p, t
    )
    g_kp = SVector{3}(ForwardDiff.gradient(r -> d.f(r, p, t)[1], SVector{3}(x)))
    g_kr = SVector{3}(ForwardDiff.gradient(r -> d.f(r, p, t)[2], SVector{3}(x)))
    return (g_kp, g_kr)
end

@inline function _div_tensor_from_coeffs(
        kpara, kperp, g_kp, g_kr, b, J_b, ::Val{D}, ::Val{TRACK3D} = Val(false)
    ) where {D, TRACK3D}
    T = promote_type(typeof(kpara), typeof(kperp), eltype(b))
    kdiff = kpara - kperp
    g_kdiff = g_kp - g_kr
    @inbounds if D == 1
        div_b = J_b[1, 1]
        b_dot_grad_b = SVector{3}(b[1] * J_b[1, 1], b[1] * J_b[2, 1], b[1] * J_b[3, 1])
        b_dot_g_kdiff = b[1] * g_kdiff[1]
        res = SVector{3}(g_kr[1], zero(T), zero(T)) + b_dot_g_kdiff * b +
            kdiff * (div_b * b + b_dot_grad_b)
        return SVector{3}(res[1], zero(T), zero(T))
    elseif D == 2
        div_b = J_b[1, 1] + J_b[2, 2]
        b_dot_grad_b = SVector{3}(
            b[1] * J_b[1, 1] + b[2] * J_b[1, 2],
            b[1] * J_b[2, 1] + b[2] * J_b[2, 2],
            b[1] * J_b[3, 1] + b[2] * J_b[3, 2],
        )
        b_dot_g_kdiff = b[1] * g_kdiff[1] + b[2] * g_kdiff[2]
        res = SVector{3}(g_kr[1], g_kr[2], zero(T)) + b_dot_g_kdiff * b +
            kdiff * (div_b * b + b_dot_grad_b)
        return TRACK3D ? res : SVector{3}(res[1], res[2], zero(T))
    else
        div_b = J_b[1, 1] + J_b[2, 2] + J_b[3, 3]
        b_dot_grad_b = J_b * b
        b_dot_g_kdiff = dot(b, g_kdiff)
        return g_kr + b_dot_g_kdiff * b + kdiff * (div_b * b + b_dot_grad_b)
    end
end

"""
    div_diffusion_tensor(model, B_field, x, p, t, ::Val{D} = Val(3), ::Val{TRACK3D} = Val(false))

Compute the spatial divergence `(∇ ⋅ κ)_i = ∑_{j=1}^D ∂κ_ij / ∂x_j` analytically from
`B`, its spatial Jacobian `J_B`, and the gradients of `κ_∥`, `κ_⊥`.
"""
@inline function div_diffusion_tensor(
        ::Union{ZeroDiffusion, IsotropicDiffusion}, B_field, x::AbstractVector{T}, p, t,
        ::Val{D} = Val(3), ::Val{TRACK3D} = Val(false)
    ) where {T, D, TRACK3D}
    return zero(SVector{3, promote_type(T, typeof(p))})
end

@inline function div_diffusion_tensor(
        model::AbstractSpatialDiffusion, B_field, x, p, t,
        ::Val{D} = Val(3), ::Val{TRACK3D} = Val(false)
    ) where {D, TRACK3D}
    B = B_field(x, t)
    kpara, kperp = diffusion_coefficients(model, B, x, p, t)
    Bmag = norm(B)
    T = promote_type(typeof(kpara), typeof(kperp), eltype(B))

    if Bmag == zero(Bmag)
        J_B = zero(SMatrix{3, 3, T})
        b = zero(SVector{3, T})
        g_kp, _ = _grad_diffusion_coeffs(model, B, J_B, b, Bmag, kpara, kperp, x, p, t)
        return _mask_vec(g_kp, Val(D), Val(TRACK3D))
    end

    b = B / Bmag
    J_B = jacobian(B_field, x, t)
    g_kp, g_kr = _grad_diffusion_coeffs(model, B, J_B, b, Bmag, kpara, kperp, x, p, t)

    invB = inv(Bmag)
    bdotJ = J_B' * b
    J_b = (J_B - b * bdotJ') * invB
    return _div_tensor_from_coeffs(kpara, kperp, g_kp, g_kr, b, J_b, Val(D), Val(TRACK3D))
end

"""
    div_perp_diffusion_tensor(model, B_field, x, p, t, ::Val{D} = Val(3), ::Val{TRACK3D} = Val(false))

Compute `∇ ⋅ κ_⊥` for the cross-field tensor `κ_⊥ (I - b b')` (i.e. `κ_∥ = 0`).
"""
@inline function div_perp_diffusion_tensor(
        ::ZeroDiffusion, B_field, x::AbstractVector{T}, p, t,
        ::Val{D} = Val(3), ::Val{TRACK3D} = Val(false)
    ) where {T, D, TRACK3D}
    return zero(SVector{3, promote_type(T, typeof(p))})
end

@inline function div_perp_diffusion_tensor(
        model::AbstractSpatialDiffusion, B_field, x, p, t,
        ::Val{D} = Val(3), ::Val{TRACK3D} = Val(false)
    ) where {D, TRACK3D}
    B = B_field(x, t)
    kpara, kperp = diffusion_coefficients(model, B, x, p, t)
    Bmag = norm(B)
    T = promote_type(typeof(kpara), typeof(kperp), eltype(B))

    if Bmag == zero(Bmag)
        J_B = zero(SMatrix{3, 3, T})
        b = zero(SVector{3, T})
        _, g_kr = _grad_diffusion_coeffs(model, B, J_B, b, Bmag, kpara, kperp, x, p, t)
        return _mask_vec(g_kr, Val(D), Val(TRACK3D))
    end

    b = B / Bmag
    J_B = jacobian(B_field, x, t)
    _, g_kr = _grad_diffusion_coeffs(model, B, J_B, b, Bmag, kpara, kperp, x, p, t)

    invB = inv(Bmag)
    bdotJ = J_B' * b
    J_b = (J_B - b * bdotJ') * invB
    return _div_tensor_from_coeffs(
        zero(T), kperp, zero(SVector{3, T}), g_kr, b, J_b, Val(D), Val(TRACK3D)
    )
end

@inline function _spherical_tensor_and_derivs(
        kpara, kperp, g_kp, g_kr, b, J_b
    )
    T = promote_type(typeof(kpara), typeof(kperp), eltype(b))
    I3 = SMatrix{3, 3, T}(1, 0, 0, 0, 1, 0, 0, 0, 1)
    kdiff = kpara - kperp
    g_kdiff = g_kp - g_kr
    κ = T(kperp) * I3 + T(kdiff) * (b * b')
    @inbounds begin
        jb1 = SVector{3}(J_b[1, 1], J_b[2, 1], J_b[3, 1])
        jb2 = SVector{3}(J_b[1, 2], J_b[2, 2], J_b[3, 2])
        jb3 = SVector{3}(J_b[1, 3], J_b[2, 3], J_b[3, 3])
        dκ_dr = g_kr[1] * I3 + g_kdiff[1] * (b * b') + kdiff * (jb1 * b' + b * jb1')
        dκ_dθ = g_kr[2] * I3 + g_kdiff[2] * (b * b') + kdiff * (jb2 * b' + b * jb2')
        dκ_dφ = g_kr[3] * I3 + g_kdiff[3] * (b * b') + kdiff * (jb3 * b' + b * jb3')
    end
    return (κ, dκ_dr, dκ_dθ, dκ_dφ)
end

@inline function spherical_div_diffusion_tensor(
        ::ZeroDiffusion, B_field, x::AbstractVector{T}, p, t, ::Val{D} = Val(3)
    ) where {T, D}
    return zero(SVector{3, promote_type(T, typeof(p))})
end

@inline function spherical_div_diffusion_tensor(
        model::AbstractSpatialDiffusion, B_field, x, p, t, ::Val{D} = Val(3);
        perp_only::Bool = false
    ) where {D}
    B = B_field(x, t)
    kpara_raw, kperp = diffusion_coefficients(model, B, x, p, t)
    Bmag = norm(B)
    T = promote_type(typeof(kpara_raw), typeof(kperp), eltype(B))
    kpara = perp_only ? zero(T) : T(kpara_raw)

    if Bmag == zero(Bmag)
        J_B = zero(SMatrix{3, 3, T})
        b = zero(SVector{3, T})
        g_kp_raw, g_kr = _grad_diffusion_coeffs(
            model, B, J_B, b, Bmag, kpara_raw, kperp, x, p, t
        )
        g_iso = perp_only ? g_kr : g_kp_raw
        k_iso = perp_only ? T(kperp) : T(kpara_raw)
        I3 = SMatrix{3, 3, T}(1, 0, 0, 0, 1, 0, 0, 0, 1)
        return spherical_diffusion_drift(
            k_iso * I3, g_iso[1] * I3, g_iso[2] * I3, g_iso[3] * I3, x, Val(D)
        )
    end

    b = B / Bmag
    J_B = jacobian(B_field, x, t)
    g_kp_raw, g_kr = _grad_diffusion_coeffs(
        model, B, J_B, b, Bmag, kpara_raw, kperp, x, p, t
    )
    g_kp = perp_only ? zero(SVector{3, T}) : g_kp_raw
    invB = inv(Bmag)
    bdotJ = J_B' * b
    J_b = (J_B - b * bdotJ') * invB
    κ, dκ_dr, dκ_dθ, dκ_dφ = _spherical_tensor_and_derivs(kpara, kperp, g_kp, g_kr, b, J_b)
    return spherical_diffusion_drift(κ, dκ_dr, dκ_dθ, dκ_dφ, x, Val(D))
end

@inline function _mask_vec(
        v::SVector{3, T}, ::Val{D}, ::Val{TRACK3D} = Val(false)
    ) where {T, D, TRACK3D}
    @inbounds if D == 1
        return SVector{3, T}(v[1], zero(T), zero(T))
    elseif D == 2 && !TRACK3D
        return SVector{3, T}(v[1], v[2], zero(T))
    else
        return v
    end
end

# --- Momentum diffusion D_pp ---

"""
    ZeroMomentumDiffusion <: AbstractMomentumDiffusion

Zero momentum-space diffusion (`D_pp = 0`).
"""
struct ZeroMomentumDiffusion <: AbstractMomentumDiffusion end

"""
    PowerLawMomentumDiffusion(; D0, p0 = 1.0, gamma = 2.0)

Power-law momentum diffusion `D_pp(p) = D0 * (p / p0)^gamma`.
"""
struct PowerLawMomentumDiffusion{T <: AbstractFloat} <: AbstractMomentumDiffusion
    D0::T
    p0::T
    gamma::T
    function PowerLawMomentumDiffusion(; D0::Real, p0::Real = 1.0, gamma::Real = 2.0)
        D0 >= zero(D0) || throw(ArgumentError("D0 must be non-negative; got $D0."))
        p0 > zero(p0) || throw(ArgumentError("p0 must be positive; got $p0."))
        vals = promote(float(D0), float(p0), float(gamma))
        return new{typeof(vals[1])}(vals...)
    end
end

"""
    AlfvenWaveMomentumDiffusion(; vA, kappa = nothing)

Alfvén-wave stochastic acceleration (Skilling 1975):
```math
D_{pp}(\\mathbf{x}, p, t) = \\frac{p^2 v_A(\\mathbf{x}, t)^2}{9 \\kappa_\\parallel(\\mathbf{x}, p, t)}.
```
`vA` may be a positive scalar or a callable `(x, t) -> vA`. When `kappa === nothing`,
uses the spatial diffusion model from `param.kappa`.
"""
struct AlfvenWaveMomentumDiffusion{V, K} <: AbstractMomentumDiffusion
    vA::V
    kappa::K
    function AlfvenWaveMomentumDiffusion{V, K}(vA::V, kappa::K) where {V, K}
        return new{V, K}(vA, kappa)
    end
    function AlfvenWaveMomentumDiffusion(;
            vA, kappa::Union{Nothing, AbstractSpatialDiffusion} = nothing
        )
        vA isa Real && (vA >= zero(vA) || throw(ArgumentError("`vA` must be non-negative.")))
        vA_v = vA isa Real ? float(vA) : vA
        return new{typeof(vA_v), typeof(kappa)}(vA_v, kappa)
    end
end

"""
    FlowShearMomentumDiffusion(; tau0 = nothing, p0 = 1.0, alpha = 0.0, use_kappa = false)

Viscous flow-shear momentum diffusion (Earl, Jokipii & Morfill 1988; Webb et al. 2018):
```math
D_{pp} = \\Gamma \\tau(p) p^2,
\\qquad \\Gamma = \\frac{2}{15} \\sum_{i,j} \\sigma_{ij}^2,
\\qquad \\sigma_{ij} = \\frac{1}{2}(\\partial_i u_j + \\partial_j u_i)
- \\frac{1}{3}(\\nabla \\cdot \\mathbf{u}) \\delta_{ij}.
```
Set `use_kappa = true` to compute the scattering time `τ = 3 κ_∥ / v²` from `param.kappa`,
or provide `tau0` with power-law scaling `τ(p) = tau0 * (p0 / p)^alpha`.
"""
struct FlowShearMomentumDiffusion{T <: AbstractFloat} <: AbstractMomentumDiffusion
    tau0::T
    p0::T
    alpha::T
    use_kappa::Bool
    function FlowShearMomentumDiffusion(;
            tau0::Union{Nothing, Real} = nothing, p0::Real = 1.0,
            alpha::Real = 0.0, use_kappa::Bool = false
        )
        if !use_kappa
            isnothing(tau0) && throw(
                ArgumentError("Provide `tau0` or set `use_kappa = true`.")
            )
            tau0 >= zero(tau0) || throw(ArgumentError("`tau0` must be non-negative."))
        end
        p0 > zero(p0) || throw(ArgumentError("`p0` must be positive."))
        t0 = isnothing(tau0) ? 0.0 : float(tau0)
        vals = promote(t0, float(p0), float(alpha))
        return new{typeof(vals[1])}(vals[1], vals[2], vals[3], use_kappa)
    end
end

"""
    CustomMomentumDiffusion(f) <: AbstractMomentumDiffusion

Wrap a user callable `f(x, p, t) -> D_pp` as a momentum diffusion model.
"""
struct CustomMomentumDiffusion{F} <: AbstractMomentumDiffusion
    f::F
end

Adapt.adapt_structure(
    to, d::Union{ZeroMomentumDiffusion, PowerLawMomentumDiffusion, FlowShearMomentumDiffusion}
) = d
Adapt.adapt_structure(to, d::CustomMomentumDiffusion) =
    CustomMomentumDiffusion(Adapt.adapt(to, d.f))
function Adapt.adapt_structure(to, d::AlfvenWaveMomentumDiffusion)
    vA_a = Adapt.adapt(to, d.vA)
    k_a = Adapt.adapt(to, d.kappa)
    return AlfvenWaveMomentumDiffusion{typeof(vA_a), typeof(k_a)}(vA_a, k_a)
end

"""
    momentum_diffusion_terms(model::AbstractMomentumDiffusion, x, p, t)
    momentum_diffusion_terms(model::AbstractMomentumDiffusion, x, p, t, param)

Return `(drift_y, sigma_y)` for the `y = ln(p)` SDE:
- `drift_y = (1/p²) ∂(p D_pp) / ∂p`
- `sigma_y = √(2 D_pp) / p`
"""
@inline function momentum_diffusion_terms(::ZeroMomentumDiffusion, x, p, t)
    z = zero(p)
    return (z, z)
end

@inline function momentum_diffusion_terms(d::PowerLawMomentumDiffusion, x, p, t)
    Dpp = d.D0 * (p / d.p0)^d.gamma
    invp = inv(p)
    drift_y = (d.gamma + one(d.gamma)) * Dpp * invp * invp
    sigma_y = sqrt(2 * Dpp) * invp
    return (drift_y, sigma_y)
end

@inline function momentum_diffusion_terms(d::CustomMomentumDiffusion, x, p, t)
    Dpp = d.f(x, p, t)
    dDpp_dp = ForwardDiff.derivative(q -> d.f(x, q, t), p)
    invp = inv(p)
    drift_y = (Dpp * invp + dDpp_dp) * invp
    sigma_y = sqrt(2 * max(Dpp, zero(Dpp))) * invp
    return (drift_y, sigma_y)
end

@inline function momentum_diffusion_terms(
        model::AbstractMomentumDiffusion, x, p, t, param::AbstractTransportParams
    )
    return momentum_diffusion_terms(model, x, p, t)
end

@inline function momentum_diffusion_terms(
        d::AlfvenWaveMomentumDiffusion, x, p, t, param::AbstractTransportParams
    )
    vA = _eval_scalar_field(d.vA, x, t)
    B_vec = param.B(x, t)
    kmodel = isnothing(d.kappa) ? param.kappa : d.kappa
    kpara, _ = diffusion_coefficients(kmodel, B_vec, x, p, t)
    T = promote_type(typeof(p), typeof(vA), typeof(kpara))
    if kpara <= zero(kpara) || vA == zero(vA)
        return (zero(T), zero(T))
    end
    dpp_of_q = q -> begin
        kp_q, _ = diffusion_coefficients(kmodel, B_vec, x, q, t)
        (q * q * vA * vA) / (9 * kp_q)
    end
    Dpp = dpp_of_q(p)
    dDpp_dp = ForwardDiff.derivative(dpp_of_q, p)
    invp = inv(p)
    drift_y = (Dpp * invp + dDpp_dp) * invp
    sigma_y = sqrt(2 * max(Dpp, zero(Dpp))) * invp
    return (T(drift_y), T(sigma_y))
end

@inline function _shear_gamma_cartesian(J_u, ::Val{D}) where {D}
    @inbounds begin
        j11 = J_u[1, 1]
        j21 = J_u[2, 1]
        j31 = J_u[3, 1]
        j12 = D >= 2 ? J_u[1, 2] : zero(j11)
        j22 = D >= 2 ? J_u[2, 2] : zero(j11)
        j32 = D >= 2 ? J_u[3, 2] : zero(j11)
        j13 = D == 3 ? J_u[1, 3] : zero(j11)
        j23 = D == 3 ? J_u[2, 3] : zero(j11)
        j33 = D == 3 ? J_u[3, 3] : zero(j11)
        div_u = j11 + j22 + j33
        third = div_u / 3
        s11 = j11 - third
        s22 = j22 - third
        s33 = j33 - third
        s12 = (j12 + j21) / 2
        s13 = (j13 + j31) / 2
        s23 = (j23 + j32) / 2
        sum_sq = s11 * s11 + s22 * s22 + s33 * s33 + 2 * (s12 * s12 + s13 * s13 + s23 * s23)
        return (2 * sum_sq) / 15
    end
end

@inline function momentum_diffusion_terms(
        d::FlowShearMomentumDiffusion, x, p, t, param::AbstractTransportParams
    )
    D = spatial_dims(param)
    J_u = jacobian(param.u, x, t)
    Γ = _shear_gamma_cartesian(J_u, Val(D))
    T = promote_type(typeof(p), typeof(Γ), typeof(d.tau0))
    if Γ <= zero(Γ)
        return (zero(T), zero(T))
    end
    if !d.use_kappa
        τ = d.tau0 * (d.p0 / p)^d.alpha
        Dpp = Γ * τ * p * p
        # D_pp ∝ p^(2 - alpha), so (1/p²) ∂(p D_pp)/∂p = (3 - alpha) Γ τ
        drift_y = (3 - d.alpha) * Γ * τ
        sigma_y = sqrt(2 * max(Dpp, zero(Dpp))) / p
        return (T(drift_y), T(sigma_y))
    end
    B_vec = param.B(x, t)
    dpp_of_q = q -> begin
        kp_q, _ = diffusion_coefficients(param.kappa, B_vec, x, q, t)
        v_q = momentum2velocity(q, param.m, param.c)
        τ_q = v_q > zero(v_q) ? (3 * kp_q) / (v_q * v_q) : zero(kp_q)
        Γ * τ_q * q * q
    end
    Dpp = dpp_of_q(p)
    dDpp_dp = ForwardDiff.derivative(dpp_of_q, p)
    invp = inv(p)
    drift_y = (Dpp * invp + dDpp_dp) * invp
    sigma_y = sqrt(2 * max(Dpp, zero(Dpp))) * invp
    return (T(drift_y), T(sigma_y))
end

# --- Pitch-angle diffusion D_μμ (for Focused Transport) ---

"""
    ZeroPitchAngleDiffusion <: AbstractPitchAngleDiffusion

Zero pitch-angle diffusion (`D_μμ = 0`).
"""
struct ZeroPitchAngleDiffusion <: AbstractPitchAngleDiffusion end

"""
    ConstantPitchAngleDiffusion(D0) <: AbstractPitchAngleDiffusion

Isotropic pitch-angle diffusion `D_μμ(μ) = D0 * (1 - μ²)`.
"""
struct ConstantPitchAngleDiffusion{T <: AbstractFloat} <: AbstractPitchAngleDiffusion
    D0::T
    function ConstantPitchAngleDiffusion(D0::Real)
        D0 >= zero(D0) || throw(ArgumentError("D0 must be non-negative; got $D0."))
        df = float(D0)
        return new{typeof(df)}(df)
    end
end

"""
    QLTPitchAngleDiffusion(; D0, p0 = 1.0, gamma = 5/3, h0 = 0.2, B0 = 1.0, beta = 0.0)

Quasi-linear pitch-angle diffusion with resonance-gap parameter `h0` (Kong et al. 2022):
```math
D_{\\mu\\mu}(\\mathbf{x}, p, \\mu) =
D_0 (p / p_0)^{\\gamma - 1} (B_0 / |\\mathbf{B}|)^\\beta (1 - \\mu^2)(|\\mu|^{\\gamma - 1} + h_0).
```
"""
struct QLTPitchAngleDiffusion{T <: AbstractFloat} <: AbstractPitchAngleDiffusion
    D0::T
    p0::T
    gamma::T
    h0::T
    B0::T
    beta::T
    function QLTPitchAngleDiffusion(;
            D0::Real, p0::Real = 1.0, gamma::Real = 5 / 3,
            h0::Real = 0.2, B0::Real = 1.0, beta::Real = 0.0
        )
        D0 >= zero(D0) || throw(ArgumentError("D0 must be non-negative; got $D0."))
        p0 > zero(p0) || throw(ArgumentError("p0 must be positive; got $p0."))
        B0 > zero(B0) || throw(ArgumentError("B0 must be positive; got $B0."))
        h0 >= zero(h0) || throw(ArgumentError("h0 must be non-negative; got $h0."))
        gamma > one(gamma) || throw(ArgumentError("gamma must be > 1; got $gamma."))
        vals = promote(
            float(D0), float(p0), float(gamma), float(h0), float(B0), float(beta)
        )
        return new{typeof(vals[1])}(vals...)
    end
end

"""
    CustomPitchAngleDiffusion(f) <: AbstractPitchAngleDiffusion

Wrap a user callable `f(x, p, mu, t) -> D_μμ` as a pitch-angle diffusion model.
"""
struct CustomPitchAngleDiffusion{F} <: AbstractPitchAngleDiffusion
    f::F
end

Adapt.adapt_structure(
    to, d::Union{ZeroPitchAngleDiffusion, ConstantPitchAngleDiffusion, QLTPitchAngleDiffusion}
) = d
Adapt.adapt_structure(to, d::CustomPitchAngleDiffusion) =
    CustomPitchAngleDiffusion(Adapt.adapt(to, d.f))

"""
    pitch_angle_diffusion_terms(model::AbstractPitchAngleDiffusion, B_vec, x, p, mu, t)

Return `(dDmu_dmu, sigma_mu)` for the pitch-angle SDE:
`dμ = (... + ∂D_μμ/∂μ) dt + √(2 D_μμ) dW_μ`.
"""
@inline function pitch_angle_diffusion_terms(::ZeroPitchAngleDiffusion, B_vec, x, p, mu, t)
    z = zero(mu)
    return (z, z)
end

@inline function pitch_angle_diffusion_terms(
        d::ConstantPitchAngleDiffusion, B_vec, x, p, mu, t
    )
    one_m_mu2 = max(one(mu) - mu * mu, zero(mu))
    dD = -2 * d.D0 * mu
    σ = sqrt(2 * d.D0 * one_m_mu2)
    return (dD, σ)
end

@inline function pitch_angle_diffusion_terms(
        d::QLTPitchAngleDiffusion, B_vec, x, p, mu, t
    )
    T = promote_type(typeof(p), typeof(mu), eltype(B_vec), typeof(d.D0))
    amp = d.D0 * (p / d.p0)^(d.gamma - one(d.gamma))
    if d.beta != zero(d.beta)
        Bmag = norm(B_vec)
        if Bmag > zero(Bmag)
            amp *= (d.B0 / Bmag)^d.beta
        end
    end
    gm1 = d.gamma - one(d.gamma)
    amu = abs(mu)
    one_m_mu2 = max(one(T) - mu * mu, zero(T))
    # Smooth guard at |μ| → 0 for |μ|^(γ - 2) when γ < 2
    amu_reg = max(amu, T(1.0e-12))
    term = amu^gm1 + d.h0
    Dmu = amp * one_m_mu2 * term
    dterm_dmu = gm1 * sign(mu) * amu_reg^(d.gamma - 2 * one(d.gamma))
    dD = amp * (-2 * mu * term + one_m_mu2 * dterm_dmu)
    σ = sqrt(2 * max(Dmu, zero(T)))
    return (T(dD), T(σ))
end

@inline function pitch_angle_diffusion_terms(
        d::CustomPitchAngleDiffusion, B_vec, x, p, mu, t
    )
    Dmu = d.f(x, p, mu, t)
    dD = ForwardDiff.derivative(m -> d.f(x, p, m, t), mu)
    σ = sqrt(2 * max(Dmu, zero(Dmu)))
    return (dD, σ)
end
