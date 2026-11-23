# Focused transport equation (FTE) Itô SDE drift and diffusion kernels in
# (x, y, z, ln p, μ) or spherical (r, θ, φ, ln p, μ).

@inline function _time_deriv_u(u_field::AbstractField{false}, x, t)
    return zero(SVector{3, eltype(x)})
end

@inline function _time_deriv_u(u_field::AbstractFieldInterpolator, x, t)
    return zero(SVector{3, eltype(x)})
end

@inline function _time_deriv_u(u_field, x, t)
    return SVector{3}(ForwardDiff.derivative(τ -> u_field(x, τ), t))
end

@inline function _cartesian_curl(J)
    @inbounds return SVector{3}(J[3, 2] - J[2, 3], J[1, 3] - J[3, 1], J[2, 1] - J[1, 2])
end

"""
    focused_magnetic_drift(B_field, x, p, mu, t, param::FocusedTransportParams)

Compute the pitch-angle-dependent guiding-center drift velocity (grad-B + curvature + parallel):
```math
\\mathbf{v}_d(\\mu) = \\frac{p v}{q B} \\left[
  \\frac{1 - \\mu^2}{2} \\frac{\\mathbf{B} \\times \\nabla B}{B^2}
  + \\mu^2 \\frac{\\mathbf{B} \\times [(\\mathbf{B}\\cdot\\nabla)\\mathbf{B}]}{B^3}
  + \\frac{1 - \\mu^2}{2} \\frac{\\mathbf{B}(\\mathbf{B}\\cdot\\nabla\\times\\mathbf{B})}{B^3}
\\right].
```
"""
@inline function focused_magnetic_drift(
        B_field, x, p, mu, t, param::FocusedTransportParams{D, CartesianCoordinates}
    ) where {D}
    B = B_field(x, t)
    B2 = dot(B, B)
    T = promote_type(typeof(p), typeof(mu), eltype(B))
    B2 > zero(T) || return zero(SVector{3, T})
    Bmag = sqrt(B2)
    b = B / Bmag
    J_B = jacobian(B_field, x, t)

    grad_B = J_B' * b
    B_dot_grad_B = J_B * B
    curl_B = _cartesian_curl(J_B)

    v = momentum2velocity(p, param.m, param.c)
    invB = inv(Bmag)
    invB2 = inv(B2)
    invB3 = invB2 * invB
    half_1m = (one(T) - mu * mu) / 2
    mu2 = mu * mu

    term_grad = half_1m * _cross3(B, grad_B) * invB2
    term_curv = mu2 * _cross3(B, B_dot_grad_B) * invB3
    term_para = half_1m * B * (dot(B, curl_B) * invB3)

    return ((p * v) / (param.q * Bmag)) * (term_grad + term_curv + term_para)
end

@inline function focused_magnetic_drift(
        B_field, x, p, mu, t, param::FocusedTransportParams{D, SphericalCoordinates}
    ) where {D}
    B = B_field(x, t)
    B2 = dot(B, B)
    T = promote_type(typeof(p), typeof(mu), eltype(B))
    B2 > zero(T) || return zero(SVector{3, T})
    Bmag = sqrt(B2)
    b = B / Bmag
    J_B = jacobian(B_field, x, t)

    grad_B = spherical_grad_scalar(J_B' * b, x, Val(D))
    B_dot_grad_B = spherical_v_dot_grad_w(B, B, J_B, x, Val(D))
    curl_B = spherical_curl(B, J_B, x, Val(D))

    v = momentum2velocity(p, param.m, param.c)
    invB = inv(Bmag)
    invB2 = inv(B2)
    invB3 = invB2 * invB
    half_1m = (one(T) - mu * mu) / 2
    mu2 = mu * mu

    term_grad = half_1m * _cross3(B, grad_B) * invB2
    term_curv = mu2 * _cross3(B, B_dot_grad_B) * invB3
    term_para = half_1m * B * (dot(B, curl_B) * invB3)

    return ((p * v) / (param.q * Bmag)) * (term_grad + term_curv + term_para)
end

"""
    focused_drift(u, param::FocusedTransportParams, t)

Out-of-place 5-component drift vector `SVector{5}` for the forward-in-time Focused
Transport Equation SDE in `u = (x, y, z, ln p, μ)` (Cartesian) or `(r, θ, φ, ln p, μ)`
(Spherical).
"""
@inline function focused_drift(
        u, param::FocusedTransportParams{D, CartesianCoordinates, DRIFT, TRACK3D}, t
    ) where {D, DRIFT, TRACK3D}
    x = get_position(u)
    p = get_momentum(u)
    mu = get_pitch_angle(u)

    v = momentum2velocity(p, param.m, param.c)
    B = param.B(x, t)
    Bmag = norm(B)
    T0 = promote_type(typeof(p), typeof(mu), eltype(B))
    has_B = Bmag > zero(T0)
    b = has_B ? B / Bmag : zero(SVector{3, T0})

    u_flow = param.u(x, t)
    J_u = jacobian(param.u, x, t)
    div_u = @inbounds (D == 1 ? J_u[1, 1] : D == 2 ? J_u[1, 1] + J_u[2, 2] :
        J_u[1, 1] + J_u[2, 2] + J_u[3, 3])

    div_kperp = div_perp_diffusion_tensor(
        param.kappa, param.B, x, p, t, Val(D), Val(TRACK3D)
    )
    v_tot = v * mu * b + u_flow + div_kperp
    if DRIFT
        v_tot = v_tot + focused_magnetic_drift(param.B, x, p, mu, t, param)
    end
    v_masked = _mask_vec(v_tot, Val(D), Val(TRACK3D))

    # Adiabatic focusing and momentum change terms
    bJb = has_B ? dot(b, J_u * b) : zero(T0)
    dudt = _time_deriv_u(param.u, x, t) + J_u * u_flow
    b_dudt = has_B ? dot(b, dudt) : zero(T0)
    inv_v = v > zero(v) ? inv(v) : zero(v)

    b_dot_grad_lnB = if has_B
        J_B = jacobian(param.B, x, t)
        grad_B = J_B' * b
        dot(b, grad_B) / Bmag
    else
        zero(T0)
    end

    one_m_mu2 = max(one(T0) - mu * mu, zero(T0))
    if in_acceleration_region(param.acc_region, x, t)
        dpp_drift, _ = momentum_diffusion_terms(param.dpp, x, p, t, param)
        dy = -(
            (one_m_mu2 / 2) * (div_u - bJb) + mu * mu * bJb + mu * inv_v * b_dudt
        ) + dpp_drift
    else
        dy = zero(div_u)
    end

    dmu_det = (one_m_mu2 / 2) * (
        -v * b_dot_grad_lnB + mu * div_u - 3 * mu * bJb - 2 * inv_v * b_dudt
    )
    dDmu_dmu, _ = pitch_angle_diffusion_terms(param.dmu, B, x, p, mu, t)
    dmu = dmu_det + dDmu_dmu

    T = promote_type(eltype(v_masked), typeof(dy), typeof(dmu))
    return @inbounds SVector{5, T}(v_masked[1], v_masked[2], v_masked[3], dy, dmu)
end

@inline function focused_drift(
        u, param::FocusedTransportParams{D, SphericalCoordinates, DRIFT, TRACK3D}, t
    ) where {D, DRIFT, TRACK3D}
    x = get_position(u)
    p = get_momentum(u)
    mu = get_pitch_angle(u)

    v = momentum2velocity(p, param.m, param.c)
    B = param.B(x, t)
    Bmag = norm(B)
    T0 = promote_type(typeof(p), typeof(mu), eltype(B))
    has_B = Bmag > zero(T0)
    b = has_B ? B / Bmag : zero(SVector{3, T0})

    u_flow = param.u(x, t)
    J_u = jacobian(param.u, x, t)
    div_u = spherical_divergence(u_flow, J_u, x, Val(D))

    v_phys = v * mu * b + u_flow
    if DRIFT
        v_phys = v_phys + focused_magnetic_drift(param.B, x, p, mu, t, param)
    end
    v_coord = spherical_coord_velocity(v_phys, x, Val(D))
    diff_drift = spherical_div_diffusion_tensor(
        param.kappa, param.B, x, p, t, Val(D); perp_only = true
    )
    v_masked = _mask_vec(v_coord + diff_drift, Val(D), Val(TRACK3D))

    bJb = has_B ? dot(b, spherical_v_dot_grad_w(b, u_flow, J_u, x, Val(D))) : zero(T0)
    dudt = _time_deriv_u(param.u, x, t) + spherical_v_dot_grad_w(u_flow, u_flow, J_u, x, Val(D))
    b_dudt = has_B ? dot(b, dudt) : zero(T0)
    inv_v = v > zero(v) ? inv(v) : zero(v)

    b_dot_grad_lnB = if has_B
        J_B = jacobian(param.B, x, t)
        grad_B = spherical_grad_scalar(J_B' * b, x, Val(D))
        dot(b, grad_B) / Bmag
    else
        zero(T0)
    end

    one_m_mu2 = max(one(T0) - mu * mu, zero(T0))
    if in_acceleration_region(param.acc_region, x, t)
        dpp_drift, _ = momentum_diffusion_terms(param.dpp, x, p, t, param)
        dy = -(
            (one_m_mu2 / 2) * (div_u - bJb) + mu * mu * bJb + mu * inv_v * b_dudt
        ) + dpp_drift
    else
        dy = zero(div_u)
    end

    dmu_det = (one_m_mu2 / 2) * (
        -v * b_dot_grad_lnB + mu * div_u - 3 * mu * bJb - 2 * inv_v * b_dudt
    )
    dDmu_dmu, _ = pitch_angle_diffusion_terms(param.dmu, B, x, p, mu, t)
    dmu = dmu_det + dDmu_dmu

    T = promote_type(eltype(v_masked), typeof(dy), typeof(dmu))
    return @inbounds SVector{5, T}(v_masked[1], v_masked[2], v_masked[3], dy, dmu)
end

"""
    focused_drift!(du, u, param::FocusedTransportParams, t)

In-place 5-component drift vector for the Focused Transport Equation SDE.
"""
@inline function focused_drift!(du, u, param::FocusedTransportParams, t)
    d = focused_drift(u, param, t)
    @inbounds for i in 1:5
        du[i] = d[i]
    end
    return nothing
end

@inline function _pack_focused_diffusion(
        σ, σ_y, σ_mu, ::Val{D}, ::Val{TRACK3D}
    ) where {D, TRACK3D}
    T = promote_type(eltype(σ), typeof(σ_y), typeof(σ_mu))
    z = zero(T)
    @inbounds if D == 1
        return SMatrix{5, 5, T}(
            σ[1, 1], z, z, z, z,
            z, z, z, z, z,
            z, z, z, z, z,
            z, z, z, σ_y, z,
            z, z, z, z, σ_mu,
        )
    elseif D == 2 && !TRACK3D
        return SMatrix{5, 5, T}(
            σ[1, 1], σ[2, 1], z, z, z,
            σ[1, 2], σ[2, 2], z, z, z,
            z, z, z, z, z,
            z, z, z, σ_y, z,
            z, z, z, z, σ_mu,
        )
    else
        return SMatrix{5, 5, T}(
            σ[1, 1], σ[2, 1], σ[3, 1], z, z,
            σ[1, 2], σ[2, 2], σ[3, 2], z, z,
            σ[1, 3], σ[2, 3], σ[3, 3], z, z,
            z, z, z, σ_y, z,
            z, z, z, z, σ_mu,
        )
    end
end

"""
    focused_diffusion(u, param::FocusedTransportParams, t)

Out-of-place 5×5 diffusion matrix `SMatrix{5, 5}` for the Focused Transport Equation SDE.
Upper-left 3×3 block is the cross-field square root `σ_⊥ = √(2 κ_⊥) (I - b b')`,
`(4, 4)` is `√(2 D_pp) / p`, and `(5, 5)` is `√(2 D_μμ)`.
"""
@inline function focused_diffusion(
        u, param::FocusedTransportParams{D, CartesianCoordinates, DRIFT, TRACK3D}, t
    ) where {D, DRIFT, TRACK3D}
    x = get_position(u)
    p = get_momentum(u)
    mu = get_pitch_angle(u)
    B = param.B(x, t)

    σ = perp_diffusion_sqrt(param.kappa, param.B, x, p, t)
    if in_acceleration_region(param.acc_region, x, t)
        _, σ_y = momentum_diffusion_terms(param.dpp, x, p, t, param)
    else
        σ_y = zero(p)
    end
    _, σ_mu = pitch_angle_diffusion_terms(param.dmu, B, x, p, mu, t)
    return _pack_focused_diffusion(σ, σ_y, σ_mu, Val(D), Val(TRACK3D))
end

@inline function focused_diffusion(
        u, param::FocusedTransportParams{D, SphericalCoordinates, DRIFT, TRACK3D}, t
    ) where {D, DRIFT, TRACK3D}
    x = get_position(u)
    p = get_momentum(u)
    mu = get_pitch_angle(u)
    B = param.B(x, t)

    σ_phys = perp_diffusion_sqrt(param.kappa, param.B, x, p, t)
    σ = spherical_diffusion_sqrt(σ_phys, x, Val(D))
    if in_acceleration_region(param.acc_region, x, t)
        _, σ_y = momentum_diffusion_terms(param.dpp, x, p, t, param)
    else
        σ_y = zero(p)
    end
    _, σ_mu = pitch_angle_diffusion_terms(param.dmu, B, x, p, mu, t)
    T = promote_type(eltype(σ), typeof(σ_y), typeof(σ_mu))
    z = zero(T)
    @inbounds return SMatrix{5, 5, T}(
        σ[1, 1], σ[2, 1], σ[3, 1], z, z,
        σ[1, 2], σ[2, 2], σ[3, 2], z, z,
        σ[1, 3], σ[2, 3], σ[3, 3], z, z,
        z, z, z, σ_y, z,
        z, z, z, z, σ_mu,
    )
end

"""
    focused_diffusion!(du, u, param::FocusedTransportParams, t)

In-place 5×5 diffusion matrix for the Focused Transport Equation SDE.
"""
@inline function focused_diffusion!(du, u, param::FocusedTransportParams, t)
    G = focused_diffusion(u, param, t)
    @inbounds for j in 1:5, i in 1:5
        du[i, j] = G[i, j]
    end
    return nothing
end
