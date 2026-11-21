# Parker transport Itô SDE drift and diffusion kernels in (x, y, z, ln p).

"""
    magnetic_drift(B_field, x, p, t, param::TransportParams)

Compute the isotropic guiding-center magnetic drift velocity
`v_d = (p * v / (3 * q)) * ∇ × (B / B²)`.
"""
@inline function magnetic_drift(B_field, x, p, t, param::TransportParams)
    v = momentum2velocity(p, param.m, param.c)
    coeff = (p * v) / (3 * param.q)
    return coeff * curl_b_over_b2(B_field, x, t)
end

"""
    parker_drift(u, param::TransportParams, t)

Out-of-place drift vector `SVector{4}` for the forward-in-time Parker transport SDE in
`u = (x, y, z, ln p)`:
- `dx/dt = u_flow + v_d + ∇ ⋅ κ` (masked to active `dims`)
- `d(ln p)/dt = -(1/3) ∇ ⋅ u_flow + (1/p²) ∂(p D_pp)/∂p`
"""
@inline function parker_drift(u, param::TransportParams{D, DRIFT}, t) where {D, DRIFT}
    x = get_position(u)
    p = get_momentum(u)

    u_flow = param.u(x, t)
    div_u = divergence(param.u, x, t, Val(D))
    div_k = div_diffusion_tensor(param.kappa, param.B, x, p, t, Val(D))

    v_tot = u_flow + div_k
    if DRIFT
        v_tot = v_tot + magnetic_drift(param.B, x, p, t, param)
    end
    v_masked = _mask_vec(v_tot, Val(D))

    dpp_drift, _ = momentum_diffusion_terms(param.dpp, x, p, t)
    dy = -(div_u / 3) + dpp_drift

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

"""
    parker_diffusion(u, param::TransportParams, t)

Out-of-place 4×4 diffusion matrix `SMatrix{4, 4}` for the forward-in-time Parker transport SDE.
The upper-left 3×3 block is the exact symmetric square root `σ` of `2 κ` (masked to active `dims`),
and entry `(4, 4)` is `√(2 D_pp) / p`.
"""
@inline function parker_diffusion(u, param::TransportParams{D}, t) where {D}
    x = get_position(u)
    p = get_momentum(u)

    σ = diffusion_sqrt(param.kappa, param.B, x, p, t)
    _, σ_y = momentum_diffusion_terms(param.dpp, x, p, t)

    T = promote_type(eltype(σ), typeof(σ_y))
    z = zero(T)

    @inbounds if D == 1
        return SMatrix{4, 4, T}(
            σ[1, 1], z, z, z,
            z, z, z, z,
            z, z, z, z,
            z, z, z, σ_y,
        )
    elseif D == 2
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
