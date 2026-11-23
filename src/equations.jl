# Parker transport Itô SDE drift and diffusion kernels in (x, y, z, ln p)
# or spherical (r, θ, φ, ln p).

"""
    magnetic_drift(B_field, x, p, t, param::TransportParams)

Compute the isotropic guiding-center magnetic drift velocity
`v_d = (p * v / (3 * q)) * ∇ × (B / B²)`.
"""
@inline function magnetic_drift(
        B_field, x, p, t, param::TransportParams{D, CartesianCoordinates}
    ) where {D}
    v = momentum2velocity(p, param.m, param.c)
    coeff = (p * v) / (3 * param.q)
    return coeff * curl_b_over_b2(B_field, x, t)
end

@inline function magnetic_drift(
        B_field, x, p, t, param::TransportParams{D, SphericalCoordinates}
    ) where {D}
    v = momentum2velocity(p, param.m, param.c)
    coeff = (p * v) / (3 * param.q)
    return coeff * spherical_curl_b_over_b2(B_field, x, t, Val(D))
end

"""
    parker_drift(u, param::TransportParams, t)

Out-of-place drift vector `SVector{4}` for the forward-in-time Parker transport SDE in
`u = (x, y, z, ln p)` (Cartesian) or `u = (r, θ, φ, ln p)` (Spherical):
- Cartesian: `dx/dt = u_flow + v_d + ∇ ⋅ κ`
- Spherical: coordinate rates for `F_1 = F r² sin θ`
- `d(ln p)/dt = -(1/3) ∇ ⋅ u_flow + (1/p²) ∂(p D_pp)/∂p` (when inside `param.acc_region`)
"""
@inline function parker_drift(
        u, param::TransportParams{D, CartesianCoordinates, DRIFT, TRACK3D}, t
    ) where {D, DRIFT, TRACK3D}
    x = get_position(u)
    p = get_momentum(u)

    u_flow = param.u(x, t)
    div_u = divergence(param.u, x, t, Val(D))
    div_k = div_diffusion_tensor(param.kappa, param.B, x, p, t, Val(D), Val(TRACK3D))

    v_tot = u_flow + div_k
    if DRIFT
        v_tot = v_tot + magnetic_drift(param.B, x, p, t, param)
    end
    v_masked = _mask_vec(v_tot, Val(D), Val(TRACK3D))

    if in_acceleration_region(param.acc_region, x, t)
        dpp_drift, _ = momentum_diffusion_terms(param.dpp, x, p, t, param)
        dy = -(div_u / 3) + dpp_drift
    else
        dy = zero(div_u)
    end

    T = promote_type(eltype(v_masked), typeof(dy))
    return @inbounds SVector{4, T}(v_masked[1], v_masked[2], v_masked[3], dy)
end

@inline function parker_drift(
        u, param::TransportParams{D, SphericalCoordinates, DRIFT, TRACK3D}, t
    ) where {D, DRIFT, TRACK3D}
    x = get_position(u)
    p = get_momentum(u)

    u_flow = param.u(x, t)
    J_u = jacobian(param.u, x, t)
    div_u = spherical_divergence(u_flow, J_u, x, Val(D))

    v_phys = u_flow
    if DRIFT
        v_phys = v_phys + magnetic_drift(param.B, x, p, t, param)
    end
    v_coord = spherical_coord_velocity(v_phys, x, Val(D))
    diff_drift = spherical_div_diffusion_tensor(param.kappa, param.B, x, p, t, Val(D))
    v_masked = _mask_vec(v_coord + diff_drift, Val(D), Val(TRACK3D))

    if in_acceleration_region(param.acc_region, x, t)
        dpp_drift, _ = momentum_diffusion_terms(param.dpp, x, p, t, param)
        dy = -(div_u / 3) + dpp_drift
    else
        dy = zero(div_u)
    end

    T = promote_type(eltype(v_masked), typeof(dy))
    return @inbounds SVector{4, T}(v_masked[1], v_masked[2], v_masked[3], dy)
end

"""
    parker_drift!(du, u, param::TransportParams, t)

In-place drift vector for the forward-in-time Parker transport SDE.
"""
@inline function parker_drift!(du, u, param::TransportParams, t)
    d = parker_drift(u, param, t)
    @inbounds begin
        du[1] = d[1]
        du[2] = d[2]
        du[3] = d[3]
        du[4] = d[4]
    end
    return nothing
end

@inline function _pack_parker_diffusion(
        σ, σ_y, ::Val{D}, ::Val{TRACK3D}
    ) where {D, TRACK3D}
    T = promote_type(eltype(σ), typeof(σ_y))
    z = zero(T)
    @inbounds if D == 1
        return SMatrix{4, 4, T}(
            σ[1, 1], z, z, z,
            z, z, z, z,
            z, z, z, z,
            z, z, z, σ_y,
        )
    elseif D == 2 && !TRACK3D
        return SMatrix{4, 4, T}(
            σ[1, 1], σ[2, 1], z, z,
            σ[1, 2], σ[2, 2], z, z,
            z, z, z, z,
            z, z, z, σ_y,
        )
    else
        return SMatrix{4, 4, T}(
            σ[1, 1], σ[2, 1], σ[3, 1], z,
            σ[1, 2], σ[2, 2], σ[3, 2], z,
            σ[1, 3], σ[2, 3], σ[3, 3], z,
            z, z, z, σ_y,
        )
    end
end

"""
    parker_diffusion(u, param::TransportParams, t)

Out-of-place 4×4 diffusion matrix `SMatrix{4, 4}` for the forward-in-time Parker transport SDE.
The upper-left 3×3 block is the exact symmetric square root `σ` of `2 κ` (scaled by spherical
metrics when `SphericalCoordinates` is active), and entry `(4, 4)` is `√(2 D_pp) / p`.
"""
@inline function parker_diffusion(
        u, param::TransportParams{D, CartesianCoordinates, DRIFT, TRACK3D}, t
    ) where {D, DRIFT, TRACK3D}
    x = get_position(u)
    p = get_momentum(u)

    σ = diffusion_sqrt(param.kappa, param.B, x, p, t)
    if in_acceleration_region(param.acc_region, x, t)
        _, σ_y = momentum_diffusion_terms(param.dpp, x, p, t, param)
    else
        σ_y = zero(p)
    end
    return _pack_parker_diffusion(σ, σ_y, Val(D), Val(TRACK3D))
end

@inline function parker_diffusion(
        u, param::TransportParams{D, SphericalCoordinates, DRIFT, TRACK3D}, t
    ) where {D, DRIFT, TRACK3D}
    x = get_position(u)
    p = get_momentum(u)

    σ_phys = diffusion_sqrt(param.kappa, param.B, x, p, t)
    σ = spherical_diffusion_sqrt(σ_phys, x, Val(D))
    if in_acceleration_region(param.acc_region, x, t)
        _, σ_y = momentum_diffusion_terms(param.dpp, x, p, t, param)
    else
        σ_y = zero(p)
    end
    # Note: in spherical 1D/2D, row 1 (or rows 1:2) have non-zeros across all 3 columns of σ_phys,
    # so preserve all 3 noise columns in the packed 4×4 matrix.
    T = promote_type(eltype(σ), typeof(σ_y))
    z = zero(T)
    @inbounds return SMatrix{4, 4, T}(
        σ[1, 1], σ[2, 1], σ[3, 1], z,
        σ[1, 2], σ[2, 2], σ[3, 2], z,
        σ[1, 3], σ[2, 3], σ[3, 3], z,
        z, z, z, σ_y,
    )
end

"""
    parker_diffusion!(du, u, param::TransportParams, t)

In-place 4×4 diffusion matrix for the forward-in-time Parker transport SDE.
"""
@inline function parker_diffusion!(du, u, param::TransportParams, t)
    G = parker_diffusion(u, param, t)
    @inbounds for j in 1:4, i in 1:4
        du[i, j] = G[i, j]
    end
    return nothing
end
