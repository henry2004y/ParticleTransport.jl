# Spatial diffusion tensor (κ_∥, κ_⊥) and momentum diffusion (D_pp) models.

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
    CustomDiffusion(f) <: AbstractSpatialDiffusion

Wrap a user callable `f(x, p, t) -> (kpara, kperp)` as a spatial diffusion model.
"""
struct CustomDiffusion{F} <: AbstractSpatialDiffusion
    f::F
end

Adapt.adapt_structure(to, d::Union{ZeroDiffusion, IsotropicDiffusion, ConstantDiffusion, PowerLawDiffusion}) = d
Adapt.adapt_structure(to, d::CustomDiffusion) = CustomDiffusion(Adapt.adapt(to, d.f))

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
    # ∇(|B|) = J_B' * b, so ∇κ_∥ = -β (κ_∥ / |B|) (J_B' * b)
    grad_Bmag = J_B' * b
    grad_kp = (-d.beta * kpara / Bmag) * grad_Bmag
    return (grad_kp, d.eta * grad_kp)
end

@inline function _grad_diffusion_coeffs(
        d::CustomDiffusion, B, J_B, b, Bmag, kpara, kperp, x, p, t
    )
    g_kp = SVector{3}(ForwardDiff.gradient(r -> d.f(r, p, t)[1], SVector{3}(x)))
    g_kr = SVector{3}(ForwardDiff.gradient(r -> d.f(r, p, t)[2], SVector{3}(x)))
    return (g_kp, g_kr)
end

"""
    div_diffusion_tensor(model, B_field, x, p, t, ::Val{D} = Val(3))

Compute the spatial divergence `(∇ ⋅ κ)_i = ∑_{j=1}^D ∂κ_ij / ∂x_j` analytically from
`B`, its spatial Jacobian `J_B`, and the gradients of `κ_∥`, `κ_⊥`.
"""
@inline function div_diffusion_tensor(
        ::Union{ZeroDiffusion, IsotropicDiffusion}, B_field, x::AbstractVector{T}, p, t,
        ::Val{D} = Val(3)
    ) where {T, D}
    return zero(SVector{3, promote_type(T, typeof(p))})
end

@inline function div_diffusion_tensor(
        model::AbstractSpatialDiffusion, B_field, x, p, t, ::Val{D} = Val(3)
    ) where {D}
    B = B_field(x, t)
    kpara, kperp = diffusion_coefficients(model, B, x, p, t)
    Bmag = norm(B)
    T = promote_type(typeof(kpara), typeof(kperp), eltype(B))

    if Bmag == zero(Bmag)
        J_B = zero(SMatrix{3, 3, T})
        b = zero(SVector{3, T})
        g_kp, _ = _grad_diffusion_coeffs(model, B, J_B, b, Bmag, kpara, kperp, x, p, t)
        return _mask_vec(g_kp, Val(D))
    end

    b = B / Bmag
    J_B = jacobian(B_field, x, t)
    g_kp, g_kr = _grad_diffusion_coeffs(model, B, J_B, b, Bmag, kpara, kperp, x, p, t)

    # ∂b_i / ∂x_j = (J_B[i,j] - b_i * (b ⋅ J_B[:,j])) / |B|
    invB = inv(Bmag)
    bdotJ = J_B' * b # vector of length 3: (b ⋅ ∂B/∂x_j)
    J_b = (J_B - b * bdotJ') * invB

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
        return SVector{3}(res[1], res[2], zero(T))
    else
        div_b = J_b[1, 1] + J_b[2, 2] + J_b[3, 3]
        b_dot_grad_b = J_b * b
        b_dot_g_kdiff = dot(b, g_kdiff)
        return g_kr + b_dot_g_kdiff * b + kdiff * (div_b * b + b_dot_grad_b)
    end
end

@inline function _mask_vec(v::SVector{3, T}, ::Val{D}) where {T, D}
    @inbounds if D == 1
        return SVector{3, T}(v[1], zero(T), zero(T))
    elseif D == 2
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
    CustomMomentumDiffusion(f) <: AbstractMomentumDiffusion

Wrap a user callable `f(x, p, t) -> D_pp` as a momentum diffusion model.
"""
struct CustomMomentumDiffusion{F} <: AbstractMomentumDiffusion
    f::F
end

Adapt.adapt_structure(to, d::Union{ZeroMomentumDiffusion, PowerLawMomentumDiffusion}) = d
Adapt.adapt_structure(to, d::CustomMomentumDiffusion) =
    CustomMomentumDiffusion(Adapt.adapt(to, d.f))

"""
    momentum_diffusion_terms(model::AbstractMomentumDiffusion, x, p, t)

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
