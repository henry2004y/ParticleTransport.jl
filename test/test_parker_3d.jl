module test_parker_3d

using Test
using Random
using LinearAlgebra: dot, norm
using Statistics: mean, var
using StochasticDiffEq: EM
using StaticArrays: SA
using ParticleTransport

@testset "3D Anisotropic Diffusion in Tilted B" begin
    let
        Random.seed!(77)
        b_hat = SA[1.0, 2.0, 2.0] / 3.0
        B_tilted(x) = 2.0 * b_hat
        kpara = 1.0
        kperp = 0.1
        T_end = 1.5
        param = prepare(; B = B_tilted, kappa = ConstantDiffusion(kpara, kperp), dims = 3)
        u0 = make_state(SA[0.0, 0.0, 0.0], 1.0)
        prob = TransportProblem(u0, (0.0, T_end), param)
        ep = EnsembleProblem(prob)
        sim = solve(ep, EM(), EnsembleThreads(); dt = 1.0e-2, trajectories = 2500, save_everystep = false)

        # Project displacement onto b_hat and two orthogonal unit vectors
        e1 = SA[2.0, -1.0, 0.0] / sqrt(5.0)
        e2 = SA[b_hat[2] * e1[3] - b_hat[3] * e1[2],
            b_hat[3] * e1[1] - b_hat[1] * e1[3],
            b_hat[1] * e1[2] - b_hat[2] * e1[1]]

        x_para = [dot(get_position(s.u[end]), b_hat) for s in sim.u]
        x_perp1 = [dot(get_position(s.u[end]), e1) for s in sim.u]
        x_perp2 = [dot(get_position(s.u[end]), e2) for s in sim.u]

        @test var(x_para) ≈ 2 * kpara * T_end rtol = 0.12
        @test var(x_perp1) ≈ 2 * kperp * T_end rtol = 0.12
        @test var(x_perp2) ≈ 2 * kperp * T_end rtol = 0.12
    end
end

@testset "Momentum Diffusion D_pp = D0 * p^2 (Log-Normal Exact)" begin
    let
        Random.seed!(99)
        D0 = 0.2
        T_end = 2.0
        param = prepare(; dpp = PowerLawMomentumDiffusion(; D0, p0 = 1.0, gamma = 2.0), dims = 1)
        p0 = 1.5
        u0 = make_state(0.0, p0)
        prob = TransportProblem(u0, (0.0, T_end), param)
        ep = EnsembleProblem(prob)
        sim = solve(ep, EM(), EnsembleThreads(); dt = 1.0e-2, trajectories = 2500, save_everystep = false)

        dy = [s.u[end][4] - log(p0) for s in sim.u]
        @test mean(dy) ≈ 3 * D0 * T_end rtol = 0.08
        @test var(dy) ≈ 2 * D0 * T_end rtol = 0.10
    end
end

@testset "Magnetic Drift & In-Place Interface" begin
    let
        # B(x) = (0, 0, B0 * (1 + α x)) => v_d = (0, (p*v/(3q)) * α / (B0(1+αx)^2), 0)
        B0 = 1.0
        α = 0.2
        B_grad(x) = SA[0.0, 0.0, B0 * (1.0 + α * x[1])]
        param = prepare(; B = B_grad, q = 1.0, m = 1.0, c = Inf, dims = 3, drift = true)

        p0 = 3.0
        T_end = 2.0
        # Since v_d is purely along y, x stays at 0 and B(0) = B0 is constant along trajectory!
        vd_expected = (p0 * p0 / 3.0) * (α / B0)

        # Out-of-place (SVector)
        u0_oop = make_state(SA[0.0, 0.0, 0.0], p0)
        sol_oop = solve(TransportProblem(u0_oop, (0.0, T_end), param), EM(); dt = 1.0e-2)
        @test sol_oop.u[end][2] ≈ vd_expected * T_end rtol = 1.0e-6

        # In-place (Vector)
        u0_iip = collect(u0_oop)
        sol_iip = solve(TransportProblem(u0_iip, (0.0, T_end), param), EM(); dt = 1.0e-2)
        @test sol_iip.u[end][2] ≈ vd_expected * T_end rtol = 1.0e-6
    end
end

@testset "2D Curved Azimuthal Field Itô Drift Invariance" begin
    let
        # Pure parallel diffusion on circle r = r0: div(κ) = -(κ_∥/r) r̂ cancels curvature push
        B_circ(x) = begin
            r = hypot(x[1], x[2])
            SA[-x[2] / r, x[1] / r, 0.0]
        end
        r0 = 2.0
        kpara = 0.5
        T_end = 1.0
        param = prepare(; B = B_circ, kappa = ConstantDiffusion(kpara, 0.0), dims = 2)
        prob = TransportProblem(make_state(SA[r0, 0.0, 0.0], 1.0), (0.0, T_end), param)
        Random.seed!(88)
        sim = solve(
            EnsembleProblem(prob), EM(), EnsembleThreads();
            dt = 5.0e-4, trajectories = 1500, save_everystep = false
        )
        rs = [hypot(s.u[end][1], s.u[end][2]) for s in sim.u]
        thetas = [atan(s.u[end][2], s.u[end][1]) for s in sim.u]
        @test mean(rs) ≈ r0 rtol = 0.02
        @test var(thetas) ≈ 2 * kpara * T_end / r0^2 rtol = 0.15
    end
end

end
