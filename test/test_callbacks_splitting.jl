module test_callbacks_splitting

using Test
using Random
using StochasticDiffEq: EM
using StaticArrays: SA
using ParticleTransport

@testset "Spatial & Momentum Boundary Callbacks" begin
    let
        u_flow(x) = SA[2.0, 0.0, 0.0]
        param = prepare(; u = u_flow, kappa = ZeroDiffusion(), dims = 1)
        u0 = make_state(0.0, 1.0)
        prob = TransportProblem(u0, (0.0, 5.0), param)

        # TerminateOutside
        sol_term = solve(prob, EM(); dt = 0.01, callback = TerminateOutside(((-1.0, 1.0),)))
        @test sol_term.t[end] < 1.0
        @test sol_term.u[end][1] >= 1.0

        # PeriodicBoundary: x(5) = 10 wrapped into [0, 3) => 1.0
        sol_per = solve(prob, EM(); dt = 0.01, callback = PeriodicBoundary(((0.0, 3.0),)))
        @test 0.0 <= sol_per.u[end][1] < 3.0
        @test sol_per.u[end][1] ≈ 1.0 atol = 0.05

        # ReflectingBoundary: x stays in [0, 3]
        sol_ref = solve(prob, EM(); dt = 0.01, callback = ReflectingBoundary(((0.0, 3.0),)))
        @test 0.0 <= sol_ref.u[end][1] <= 3.0

        # MomentumBounds (:terminate and :reflect)
        u_comp(x) = SA[-3.0 * x[1], 0.0, 0.0] # d(ln p)/dt = +1
        p_comp = prepare(; u = u_comp, kappa = ZeroDiffusion(), dims = 1)
        prob_p = TransportProblem(make_state(0.0, 1.0), (0.0, 5.0), p_comp)

        sol_pterm = solve(prob_p, EM(); dt = 0.01, callback = MomentumBounds(; pmax = 2.0, action = :terminate))
        @test sol_pterm.t[end] < 1.0

        sol_pref = solve(prob_p, EM(); dt = 0.01, callback = MomentumBounds(; pmin = 0.5, pmax = 2.0, action = :reflect))
        @test 0.5 <= get_momentum(sol_pref.u[end]) <= 2.0
    end
end

@testset "Particle Splitting Weight Conservation" begin
    let
        # Deterministic heating d(ln p)/dt = +1 => p(t) = exp(t), crosses 2, 4, 8 in t ∈ [0, 2.5]
        u_comp(x) = SA[-3.0 * x[1], 0.0, 0.0]
        param = prepare(; u = u_comp, kappa = ZeroDiffusion(), dims = 1)
        prob = TransportProblem(make_state(0.0, 1.0), (0.0, 2.5), param)

        rule = SplitRule(; ratio = 2.0, nsplit = 2, max_generations = 4)
        res = solve_with_splitting(prob, EM(), EnsembleSerial(); rule, trajectories = 5, dt = 0.01)

        @test length(res.states) > 5
        @test res.generations >= 2
        @test sum(res.weights) ≈ 5.0 rtol = 1.0e-12
    end
end

end
