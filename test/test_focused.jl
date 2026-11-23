module test_focused

using Test
using ParticleTransport
using StochasticDiffEq: EM, SOSRI
using StaticArrays: SA, SVector, SMatrix
using LinearAlgebra: dot, norm
using Statistics: mean, std
const Adapt = ParticleTransport.Adapt

@testset "Focused Transport: Parallel Streaming & Magnetic Mirroring" begin
    # 1. Exact parallel streaming in uniform B along x: x(t) = x0 + v * mu * t
    let
        B(x) = SA[1.0, 0.0, 0.0]
        param = prepare(;
            B, model = FocusedTransport(), dims = 3,
            q = 1.0, m = 1.0, c = Inf,
        )
        u0 = make_state(SA[0.0, 0.0, 0.0], 2.0, 0.5)
        prob = TransportProblem(u0, (0.0, 3.0), param)
        sol = solve(prob, EM(); dt = 1.0e-2)
        # v = p/m = 2.0, mu = 0.5 => vx = 1.0 => x(3) = 3.0
        @test get_position(sol.u[end]) ≈ SA[3.0, 0.0, 0.0] atol = 1.0e-12
        @test get_momentum(sol.u[end]) ≈ 2.0 atol = 1.0e-12
        @test get_pitch_angle(sol.u[end]) ≈ 0.5 atol = 1.0e-12
    end

    # 2. Magnetic mirroring in a converging bottle B_x(x) = B0 * (1 + (x/L)^2)
    # Adiabatic invariant (1 - mu^2) / B(x) is conserved along deterministic trajectory
    let
        B0 = 1.0
        L = 2.0
        B(x) = SA[B0 * (1.0 + (x[1] / L)^2), 0.0, 0.0]
        param = prepare(;
            B, model = FocusedTransport(), dims = 1,
            q = 1.0, m = 1.0, c = Inf,
        )
        mu0 = 0.6
        p0 = 1.5
        u0 = make_state(SA[0.0, 0.0, 0.0], p0, mu0)
        inv0 = (1.0 - mu0^2) / norm(B(SA[0.0, 0.0, 0.0]))
        prob = TransportProblem(u0, (0.0, 4.0), param)
        sol = solve(prob, EM(); dt = 1.0e-3)
        x_end = get_position(sol.u[end])
        mu_end = get_pitch_angle(sol.u[end])
        inv_end = (1.0 - mu_end^2) / norm(B(x_end))
        @test inv_end ≈ inv0 rtol = 5.0e-3
        # Particle should have mirrored (mu flipped sign)
        @test mu_end < 0.0
    end
end

@testset "Focused Transport: Pitch-Angle Drift Identity & Isotropization" begin
    # Isotropic average of focused_magnetic_drift over mu in [-1, 1] equals Parker magnetic_drift
    let
        B(x) = SA[0.3 * x[2] + 1.0, -0.4 * x[1] + 0.5 * x[3], 1.2 + 0.2 * x[1] * x[2]]
        p_parker = prepare(; B, drift = true, q = 1.0, m = 1.0, c = Inf)
        p_fte = prepare(; B, model = FocusedTransport(), drift = true, q = 1.0, m = 1.0, c = Inf)
        x = SA[0.7, -0.3, 0.5]
        p = 2.3
        vd_parker = ParticleTransport.magnetic_drift(p_parker.B, x, p, 0.0, p_parker)
        # 2-point Gauss-Legendre on [-1, 1] integrates degree <= 3 polynomials in mu exactly
        mu_gl = 1.0 / sqrt(3.0)
        vd_avg = 0.5 * (
            ParticleTransport.focused_magnetic_drift(p_fte.B, x, p, mu_gl, 0.0, p_fte) +
            ParticleTransport.focused_magnetic_drift(p_fte.B, x, p, -mu_gl, 0.0, p_fte)
        )
        @test vd_avg ≈ vd_parker rtol = 1.0e-12
    end

    # Pitch-angle diffusion isotropizes mu uniformly on [-1, 1]
    let
        B(x) = SA[1.0, 0.0, 0.0]
        param = prepare(;
            B, model = FocusedTransport(),
            dmu = ConstantPitchAngleDiffusion(0.5),
            dims = 1, q = 1.0, m = 1.0, c = Inf,
        )
        u0 = make_state(SA[0.0, 0.0, 0.0], 1.0, 0.9)
        prob = TransportProblem(u0, (0.0, 4.0), param)
        ep = EnsembleProblem(prob)
        sim = solve(
            ep, EM(), EnsembleThreads();
            dt = 1.0e-2, trajectories = 1500, callback = PitchAngleBounds(),
            save_everystep = false,
        )
        mus = [get_pitch_angle(s.u[end]) for s in sim.u]
        @test all(m -> -1.0 <= m <= 1.0, mus)
        # Uniform[-1, 1] has mean 0, std 1/sqrt(3) ≈ 0.577
        @test abs(mean(mus)) < 0.08
        @test std(mus) ≈ 1.0 / sqrt(3.0) atol = 0.06
    end
end

@testset "Focused Transport: QLT D_μμ, In-Place, Adapt & Zero-Alloc" begin
    let
        B(x) = SA[1.0, 0.2 * x[1], 0.0]
        u_flow(x) = SA[0.5 * x[1], 0.0, 0.0]
        dmu = QLTPitchAngleDiffusion(; D0 = 0.2, p0 = 1.0, gamma = 5 / 3, h0 = 0.2)
        param = prepare(;
            u = u_flow, B, model = FocusedTransport(),
            kappa = ConstantDiffusion(0.0, 0.05), dmu, drift = true,
            q = 1.0, m = 1.0, c = Inf,
        )
        @test Adapt.adapt(identity, param) == param

        u0 = make_state(SA[0.4, 0.1, -0.2], 1.5, 0.3)
        d_oop = focused_drift(u0, param, 0.0)
        G_oop = focused_diffusion(u0, param, 0.0)

        # Cross-field diffusion square root check: σ_⊥ * σ_⊥' = 2 κ_⊥ (I - b b')
        σ = G_oop[1:3, 1:3]
        b = B(get_position(u0)) / norm(B(get_position(u0)))
        κ_perp_exact = 2 * 0.05 * (SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1) - b * b')
        @test σ * σ' ≈ κ_perp_exact atol = 1.0e-12

        # In-place matches out-of-place
        du = zeros(5)
        dG = zeros(5, 5)
        focused_drift!(du, collect(u0), param, 0.0)
        focused_diffusion!(dG, collect(u0), param, 0.0)
        @test du ≈ d_oop
        @test dG ≈ G_oop

        # Zero heap allocations on out-of-place kernels
        @test (@allocated focused_drift(u0, param, 0.0)) == 0
        @test (@allocated focused_diffusion(u0, param, 0.0)) == 0
    end
end

end
