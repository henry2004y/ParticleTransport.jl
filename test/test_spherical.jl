module test_spherical

using Test
using ParticleTransport
using StochasticDiffEq: EM
using StaticArrays: SA, SMatrix
using LinearAlgebra: norm
using Statistics: mean, var

@testset "Spherical Parker 1D: Radial Solar Wind Adiabatic Cooling" begin
    # Constant radial solar wind u_r = u0 => ∇⋅u = 2 u0 / r => p(t) = p0 * (r0 / r(t))^(2/3)
    let
        u0_sw = 2.0
        u(x) = SA[u0_sw, 0.0, 0.0]
        B(x) = SA[1.0 / x[1]^2, 0.0, 0.0]
        param = prepare(;
            u, B, coords = SphericalCoordinates(), dims = 1,
            q = 1.0, m = 1.0, c = Inf,
        )
        r0 = 1.0
        p0 = 3.0
        u_init = make_state(SA[r0, pi / 2, 0.0], p0)
        prob = TransportProblem(u_init, (0.0, 1.5), param)
        sol = solve(prob, EM(); dt = 1.0e-3)
        r_end = get_position(sol.u[end])[1]
        p_end = get_momentum(sol.u[end])
        @test r_end ≈ r0 + u0_sw * 1.5 atol = 1.0e-10
        @test p_end ≈ p0 * (r0 / r_end)^(2 / 3) rtol = 1.0e-3
    end
end

@testset "Spherical Parker 1D: Radial Diffusion Drift & Noise" begin
    # Constant radial κ => dr = (2 κ / r) dt + √(2 κ) dW_r
    let
        κ0 = 0.25
        B(x) = SA[1.0 / x[1]^2, 0.0, 0.0]
        param = prepare(;
            B, kappa = IsotropicDiffusion(κ0),
            coords = SphericalCoordinates(), dims = 1,
            q = 1.0, m = 1.0, c = Inf,
        )
        x = SA[2.0, pi / 2, 0.0]
        u_st = make_state(x, 1.0)
        d = parker_drift(u_st, param, 0.0)
        G = parker_diffusion(u_st, param, 0.0)
        @test d[1] ≈ 2 * κ0 / x[1] atol = 1.0e-12
        @test d[2] == 0.0 && d[3] == 0.0 && d[4] == 0.0
        # P * P' upper-left (1,1) entry must equal 2 * κ0
        PPt = G[1:3, 1:3] * G[1:3, 1:3]'
        @test PPt[1, 1] ≈ 2 * κ0 atol = 1.0e-12
        @test (@allocated parker_drift(u_st, param, 0.0)) == 0
        @test (@allocated parker_diffusion(u_st, param, 0.0)) == 0
    end
end

@testset "Spherical 3D: Parker Spiral Curl, Drift & P*P' Exactness" begin
    # Parker spiral: B_r = B0 / r^2, B_θ = 0, B_φ = -B0 * Ω * sin(θ) / (V_sw * r)
    # ∇ ⋅ B = 0; check spherical_curl and P * P' = S * (2 κ) * S
    let
        B0 = 1.0
        a = 0.5 # Ω / V_sw
        B(x) = SA[B0 / x[1]^2, 0.0, -B0 * a * sin(x[2]) / x[1]]
        param = prepare(;
            B, kappa = ConstantDiffusion(0.4, 0.1),
            coords = SphericalCoordinates(), dims = 3, drift = true,
            q = 1.0, m = 1.0, c = Inf,
        )
        r, θ, φ = 1.5, 1.1, 0.7
        x = SA[r, θ, φ]
        u_st = make_state(x, 2.0)

        # Analytical curl of Parker spiral:
        # (∇ × B)_r = (1/(r sin θ)) ∂_θ (sin θ * B_φ) = -2 B0 a cos θ / r^2
        # (∇ × B)_θ = -(1/r) ∂_r (r B_φ) = 0
        # (∇ × B)_φ = 0
        J_B = ParticleTransport.jacobian(param.B, x, 0.0)
        curlB = ParticleTransport.spherical_curl(param.B(x, 0.0), J_B, x, Val(3))
        @test curlB ≈ SA[-2 * B0 * a * cos(θ) / r^2, 0.0, 0.0] atol = 1.0e-12

        # Verify P * P' = diag(1, 1/r, 1/(r sin θ)) * 2κ * diag(1, 1/r, 1/(r sin θ))
        G = parker_diffusion(u_st, param, 0.0)
        P = G[1:3, 1:3]
        κ = ParticleTransport.diffusion_tensor(param.kappa, param.B, x, 2.0, 0.0)
        S = SMatrix{3, 3, Float64}(1.0, 0, 0, 0, 1.0 / r, 0, 0, 0, 1.0 / (r * sin(θ)))
        @test P * P' ≈ S * (2 * κ) * S atol = 1.0e-12

        @test (@allocated parker_drift(u_st, param, 0.0)) == 0
        @test (@allocated parker_diffusion(u_st, param, 0.0)) == 0
    end
end

@testset "Spherical Focused Transport: Radial Mirroring in B_r ∝ 1/r²" begin
    # In radial field B_r = B0 / r^2, b ⋅ ∇ ln B = -2 / r => dμ/dt = (1 - μ^2) v / r
    # Adiabatic invariant (1 - μ^2) * r^2 is conserved!
    let
        B(x) = SA[1.0 / x[1]^2, 0.0, 0.0]
        param = prepare(;
            B, model = FocusedTransport(), coords = SphericalCoordinates(),
            dims = 1, q = 1.0, m = 1.0, c = Inf,
        )
        r0 = 1.0
        mu0 = 0.2
        u0 = make_state(SA[r0, pi / 2, 0.0], 1.0, mu0)
        inv0 = (1.0 - mu0^2) * r0^2
        prob = TransportProblem(u0, (0.0, 2.0), param)
        sol = solve(prob, EM(); dt = 1.0e-3)
        r_end = get_position(sol.u[end])[1]
        mu_end = get_pitch_angle(sol.u[end])
        inv_end = (1.0 - mu_end^2) * r_end^2
        @test inv_end ≈ inv0 rtol = 5.0e-3
        @test mu_end > mu0 # focusing towards mu -> +1 as B decreases outward
    end
end

end
