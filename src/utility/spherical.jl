# Spherical-coordinate (r, θ, φ) differential operators and SDE helpers.
# Vector fields return physical orthonormal components (V_r, V_θ, V_φ) at x = (r, θ, φ).

@inline function _spherical_metrics(x::AbstractVector, ::Val{D}) where {D}
    @inbounds begin
        r = x[1]
        inv_r = inv(r)
        if D == 1
            z = zero(inv_r)
            return (inv_r, z, z, z)
        end
        θ = x[2]
        sθ, cθ = sincos(θ)
        inv_rsθ = inv(r * sθ)
        cotθ_r = cθ * inv_rsθ
        return (inv_r, inv_rsθ, cotθ_r, cθ / sθ)
    end
end

"""
    spherical_divergence(u_vec, J_u, x, ::Val{D} = Val(3))

Compute `∇ ⋅ u` in spherical coordinates `(r, θ, φ)` from physical components `u_vec`
and coordinate Jacobian `J_u[i, j] = ∂u_i / ∂x_j`.
"""
@inline function spherical_divergence(u_vec, J_u, x, ::Val{D} = Val(3)) where {D}
    inv_r, inv_rsθ, cotθ_r, _ = _spherical_metrics(x, Val(D))
    @inbounds if D == 1
        return J_u[1, 1] + 2 * u_vec[1] * inv_r
    elseif D == 2
        return J_u[1, 1] + 2 * u_vec[1] * inv_r + J_u[2, 2] * inv_r + u_vec[2] * cotθ_r
    else
        return J_u[1, 1] + 2 * u_vec[1] * inv_r + J_u[2, 2] * inv_r + u_vec[2] * cotθ_r +
            J_u[3, 3] * inv_rsθ
    end
end

"""
    spherical_curl(B, J_B, x, ::Val{D} = Val(3))

Compute `∇ × B` in spherical physical components.
"""
@inline function spherical_curl(B, J_B, x, ::Val{D} = Val(3)) where {D}
    inv_r, inv_rsθ, cotθ_r, _ = _spherical_metrics(x, Val(max(D, 2)))
    @inbounds begin
        dB_r_dθ = D >= 2 ? J_B[1, 2] : zero(B[1])
        dB_φ_dθ = D >= 2 ? J_B[3, 2] : zero(B[3])
        dB_r_dφ = D == 3 ? J_B[1, 3] : zero(B[1])
        dB_θ_dφ = D == 3 ? J_B[2, 3] : zero(B[2])
        c_r = dB_φ_dθ * inv_r + B[3] * cotθ_r - dB_θ_dφ * inv_rsθ
        c_θ = dB_r_dφ * inv_rsθ - J_B[3, 1] - B[3] * inv_r
        c_φ = J_B[2, 1] + B[2] * inv_r - dB_r_dθ * inv_r
        return SVector{3}(c_r, c_θ, c_φ)
    end
end

"""
    spherical_grad_scalar(g_coord, x, ::Val{D} = Val(3))

Convert coordinate gradient `(∂f/∂r, ∂f/∂θ, ∂f/∂φ)` to physical spherical gradient
`(∂f/∂r, (1/r) ∂f/∂θ, (1/(r sin θ)) ∂f/∂φ)`.
"""
@inline function spherical_grad_scalar(g_coord, x, ::Val{D} = Val(3)) where {D}
    inv_r, inv_rsθ, _, _ = _spherical_metrics(x, Val(D))
    @inbounds if D == 1
        z = zero(g_coord[1])
        return SVector{3}(g_coord[1], z, z)
    elseif D == 2
        return SVector{3}(g_coord[1], g_coord[2] * inv_r, zero(g_coord[1]))
    else
        return SVector{3}(g_coord[1], g_coord[2] * inv_r, g_coord[3] * inv_rsθ)
    end
end

"""
    spherical_curl_b_over_b2(B_field, x, t, ::Val{D} = Val(3))

Compute `∇ × (B / B²)` in spherical physical components.
"""
@inline function spherical_curl_b_over_b2(
        ::ZeroField, x::AbstractVector{T}, t, ::Val{D} = Val(3)
    ) where {T, D}
    return zero(SVector{3, T})
end

@inline function spherical_curl_b_over_b2(B_field, x, t, ::Val{D} = Val(3)) where {D}
    B = B_field(x, t)
    B2 = dot(B, B)
    T = eltype(B)
    B2 > zero(T) || return zero(SVector{3, T})
    Bmag = sqrt(B2)
    J_B = jacobian(B_field, x, t)
    curl_B = spherical_curl(B, J_B, x, Val(D))
    # Coordinate gradient of |B| is J_B' * (B / |B|)
    grad_B_coord = J_B' * (B / Bmag)
    grad_B = spherical_grad_scalar(grad_B_coord, x, Val(D))
    # ∇ × (B / B²) = (∇ × B)/B² - (2 / B³) (∇B × B)
    gBxB = _cross3(grad_B, B)
    invB2 = inv(B2)
    return curl_B * invB2 - (2 * invB2 / Bmag) * gBxB
end

@inline function _cross3(a, b)
    @inbounds return SVector{3}(
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    )
end

"""
    spherical_v_dot_grad_w(v, w, J_w, x, ::Val{D} = Val(3))

Compute the covariant directional derivative `(v ⋅ ∇) w` in spherical physical components,
given `J_w[i, j] = ∂w_i / ∂x_j`.
"""
@inline function spherical_v_dot_grad_w(v, w, J_w, x, ::Val{D} = Val(3)) where {D}
    inv_r, inv_rsθ, _, cotθ = _spherical_metrics(x, Val(max(D, 2)))
    @inbounds begin
        vr, vθ, vφ = v[1], v[2], v[3]
        wr, wθ, wφ = w[1], w[2], w[3]
        dθ_r = D >= 2 ? J_w[1, 2] * inv_r : zero(vr)
        dθ_θ = D >= 2 ? J_w[2, 2] * inv_r : zero(vr)
        dθ_φ = D >= 2 ? J_w[3, 2] * inv_r : zero(vr)
        dφ_r = D == 3 ? J_w[1, 3] * inv_rsθ : zero(vr)
        dφ_θ = D == 3 ? J_w[2, 3] * inv_rsθ : zero(vr)
        dφ_φ = D == 3 ? J_w[3, 3] * inv_rsθ : zero(vr)

        cr = vr * J_w[1, 1] + vθ * dθ_r + vφ * dφ_r - (vθ * wθ + vφ * wφ) * inv_r
        cθ = vr * J_w[2, 1] + vθ * dθ_θ + vφ * dφ_θ + (vθ * wr - cotθ * vφ * wφ) * inv_r
        cφ = vr * J_w[3, 1] + vθ * dθ_φ + vφ * dφ_φ + (vφ * wr + cotθ * vφ * wθ) * inv_r
        return SVector{3}(cr, cθ, cφ)
    end
end

"""
    spherical_coord_velocity(v_phys, x, ::Val{D} = Val(3))

Convert physical velocity `(v_r, v_θ, v_φ)` to coordinate rates `(dr/dt, dθ/dt, dφ/dt)`.
"""
@inline function spherical_coord_velocity(v_phys, x, ::Val{D} = Val(3)) where {D}
    inv_r, inv_rsθ, _, _ = _spherical_metrics(x, Val(D))
    @inbounds if D == 1
        z = zero(v_phys[1])
        return SVector{3}(v_phys[1], z, z)
    elseif D == 2
        return SVector{3}(v_phys[1], v_phys[2] * inv_r, zero(v_phys[1]))
    else
        return SVector{3}(v_phys[1], v_phys[2] * inv_r, v_phys[3] * inv_rsθ)
    end
end

"""
    spherical_diffusion_sqrt(σ_phys, x, ::Val{D} = Val(3))

Scale the physical diffusion square root `σ_phys` (`σ_phys * σ_phys' = 2 κ`) into
spherical coordinate noise matrix `P = diag(1, 1/r, 1/(r sin θ)) * σ_phys` so that
`P * P' = 2 κ_coord` identically without pole singularities.
"""
@inline function spherical_diffusion_sqrt(σ_phys::SMatrix{3, 3, T}, x, ::Val{D} = Val(3)) where {T, D}
    inv_r, inv_rsθ, _, _ = _spherical_metrics(x, Val(D))
    z = zero(T)
    @inbounds if D == 1
        return SMatrix{3, 3, T}(
            σ_phys[1, 1], z, z,
            σ_phys[1, 2], z, z,
            σ_phys[1, 3], z, z,
        )
    elseif D == 2
        return SMatrix{3, 3, T}(
            σ_phys[1, 1], σ_phys[2, 1] * inv_r, z,
            σ_phys[1, 2], σ_phys[2, 2] * inv_r, z,
            σ_phys[1, 3], σ_phys[2, 3] * inv_r, z,
        )
    else
        return SMatrix{3, 3, T}(
            σ_phys[1, 1], σ_phys[2, 1] * inv_r, σ_phys[3, 1] * inv_rsθ,
            σ_phys[1, 2], σ_phys[2, 2] * inv_r, σ_phys[3, 2] * inv_rsθ,
            σ_phys[1, 3], σ_phys[2, 3] * inv_r, σ_phys[3, 3] * inv_rsθ,
        )
    end
end

"""
    spherical_diffusion_drift(κ, dκ_dr, dκ_dθ, dκ_dφ, x, ::Val{D} = Val(3))

Compute the Fokker–Planck coordinate drift `(dr/dt, dθ/dt, dφ/dt)` due to spatial
diffusion `κ` in spherical coordinates for `F_1 = F r² sin θ` (Pei et al. 2010).
"""
@inline function spherical_diffusion_drift(κ, dκ_dr, dκ_dθ, dκ_dφ, x, ::Val{D} = Val(3)) where {D}
    inv_r, inv_rsθ, cotθ_r, _ = _spherical_metrics(x, Val(D))
    inv_r2 = inv_r * inv_r
    @inbounds if D == 1
        dr = dκ_dr[1, 1] + 2 * inv_r * κ[1, 1]
        z = zero(dr)
        return SVector{3}(dr, z, z)
    elseif D == 2
        dr = dκ_dr[1, 1] + 2 * inv_r * κ[1, 1] + inv_r * dκ_dθ[1, 2] + cotθ_r * κ[1, 2]
        dθ = inv_r * dκ_dr[1, 2] + inv_r2 * κ[1, 2] + inv_r2 * dκ_dθ[2, 2] +
            cotθ_r * inv_r * κ[2, 2]
        return SVector{3}(dr, dθ, zero(dr))
    else
        dr = dκ_dr[1, 1] + 2 * inv_r * κ[1, 1] + inv_r * dκ_dθ[1, 2] + cotθ_r * κ[1, 2] +
            inv_rsθ * dκ_dφ[1, 3]
        dθ = inv_r * dκ_dr[1, 2] + inv_r2 * κ[1, 2] + inv_r2 * dκ_dθ[2, 2] +
            cotθ_r * inv_r * κ[2, 2] + inv_r * inv_rsθ * dκ_dφ[2, 3]
        dφ = inv_rsθ * dκ_dr[1, 3] + inv_r * inv_rsθ * κ[1, 3] +
            inv_r * inv_rsθ * dκ_dθ[2, 3] + inv_rsθ * inv_rsθ * dκ_dφ[3, 3]
        return SVector{3}(dr, dθ, dφ)
    end
end
