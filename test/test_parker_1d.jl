module test_parker_1d

using Test
using Random
using Statistics: mean, var
using StochasticDiffEq: EM
using StaticArrays: SA
using ParticleTransport

@testset "1D Pure Spatial Diffusion" begin
    let
        Random.seed!(42)
        kappa = 0.5
        T_end = 2.0
        param = prepare(; kappa = IsotropicDiffusion(kappa), dims = 1)
        u0 = make_state(0.0, 1.0)
        prob = TransportProblem(u0, (0.0, T_end), param)
        ep = EnsembleProblem(prob)
        sim = solve(ep, EM(), EnsembleThreads(); dt = 1.0e-2, trajectories = 2000, save_everystep = false)

        xs = [s.u[end][1] for s in sim]
        @test abs(mean(xs)) < 0.12
        @test var(xs) ≈ 2 * kappa * T_end rtol = 0.1
        # Unused y, z, ln(p) must remain unchanged
        @test all(s -> s.u[end][2] == 0.0 && s.u[end][3] == 0.0 && s.u[end][4] == 0.0, sim.u)
    end
end

@testset "1D Convection-Diffusion" begin
    let
        Random.seed!(123)
        u_val = 1.5
        kappa = 0.25
        T_end = 1.5
        u_flow(x) = SA[u_val, 0.0, 0.0]
        param = prepare(; u = u_flow, kappa = IsotropicDiffusion(kappa), dims = 1)
        u0 = make_state(0.0, 1.0)
        prob = TransportProblem(u0, (0.0, T_end), param)
        ep = EnsembleProblem(prob)
        sim = solve(ep, EM(), EnsembleThreads(); dt = 1.0e-2, trajectories = 2000, save_everystep = false)

        xs = [s.u[end][1] for s in sim]
        @test mean(xs) ≈ u_val * T_end rtol = 0.05
        @test var(xs) ≈ 2 * kappa * T_end rtol = 0.1
    end
end

@testset "1D Adiabatic Cooling / Heating (Exact)" begin
    let
        # u(x) = a * x => ∇⋅u = a => p(t) = p0 * exp(-a*t/3)
        a = -0.6 # compression => heating
        u_flow(x) = SA[a * x[1], 0.0, 0.0]
        param = prepare(; u = u_flow, kappa = ZeroDiffusion(), dims = 1)
        p0 = 2.0
        T_end = 3.0
        u0 = make_state(1.0, p0)
        prob = TransportProblem(u0, (0.0, T_end), param)
        sol = solve(prob, EM(); dt = 1.0e-3)
        @test get_momentum(sol.u[end]) ≈ p0 * exp(-a * T_end / 3) rtol = 1.0e-3
    end
end

@testset "1D Diffusive Shock Acceleration (DSA) Power-Law" begin
    let
        # Strong shock: u1 = 1.0 (x < 0), u2 = 0.25 (x > 0), compression ratio r = 4.
        # Theoretical steady-state spectrum: f(p) ∝ p^(-q) with q = 3r/(r-1) = 4,
        # so dN/d(ln p) ∝ p^3 f(p) ∝ p^(-1).
        Random.seed!(2026)
        u1 = 1.0
        u2 = 0.25
        L_sh = 0.02
        u_shock(x) = SA[0.5 * (u1 + u2) - 0.5 * (u1 - u2) * tanh(x[1] / L_sh), 0.0, 0.0]
        kappa = 0.2
        param = prepare(; u = u_shock, kappa = IsotropicDiffusion(kappa), dims = 1)

        T_max = 60.0
        u0 = make_state(0.0, 1.0)
        prob = TransportProblem(u0, (0.0, T_max), param)

        # Continuous injection in time t_inj ∈ [0, T_max] at the shock x = 0, p = 1.0
        n_init = 1200
        t_injs = rand(MersenneTwister(2026), n_init) .* T_max
        prob_func = (p_base, i, repeat) -> remake(p_base; tspan = (t_injs[i], T_max))

        rule = SplitRule(; ratio = 2.0, nsplit = 2, max_generations = 5, max_particles = 50_000)
        res = solve_with_splitting(
            prob, EM(), EnsembleThreads();
            rule, trajectories = n_init, prob_func, dt = 0.02
        )

        # Conserved total weight
        @test sum(res.weights) ≈ n_init rtol = 1.0e-10

        edges = 10.0 .^ range(log10(1.2), log10(12.0); length = 11)
        spec = momentum_spectrum(res, edges)

        # Fit log(f(p)) = a - q * log(p) over bins with positive counts
        valid = findall(>(0.0), spec.f)
        @test length(valid) >= 5
        lx = log.(spec.centers[valid])
        ly = log.(spec.f[valid])
        mx, my = mean(lx), mean(ly)
        slope = sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
        q_est = -slope
        # Theoretical q = 4.0 (allow finite-time / finite-width shock & MC tolerance)
        @test 3.3 < q_est < 4.7
    end
end

end
