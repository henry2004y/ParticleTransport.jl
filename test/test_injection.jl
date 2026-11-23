module test_injection

using Test
using ParticleTransport
using StochasticDiffEq: EM
using StaticArrays: SA, SVector
using Random: MersenneTwister
using Statistics: mean, var

@testset "Momentum & Spatial Samplers" begin
    rng = MersenneTwister(42)

    # 1. DeltaMomentum
    let
        dm = DeltaMomentum(2.5)
        @test ParticleTransport.sample_momentum(rng, dm) == 2.5
    end

    # 2. MaxwellianMomentum: E[p] = sqrt(8/pi) p_th, E[p^2] = 3 p_th^2
    let
        p_th = 1.5
        mw = MaxwellianMomentum(; p_th)
        ps = [ParticleTransport.sample_momentum(rng, mw) for _ in 1:20_000]
        @test mean(ps) ≈ sqrt(8 / pi) * p_th rtol = 0.03
        @test mean(p -> p^2, ps) ≈ 3 * p_th^2 rtol = 0.03

        mw_trunc = MaxwellianMomentum(; p_th = 1.0, pmin = 1.0, pmax = 2.0)
        ps_t = [ParticleTransport.sample_momentum(rng, mw_trunc) for _ in 1:500]
        @test all(p -> 1.0 <= p <= 2.0, ps_t)
    end

    # 3. PowerLawMomentum: dN/dp ∝ p^(-index)
    let
        # index = 3 => E[p] = 2 pmin pmax / (pmin + pmax)
        pl = PowerLawMomentum(; pmin = 1.0, pmax = 10.0, index = 3.0)
        ps = [ParticleTransport.sample_momentum(rng, pl) for _ in 1:20_000]
        @test all(p -> 1.0 <= p <= 10.0, ps)
        @test mean(ps) ≈ 2.0 * 1.0 * 10.0 / (1.0 + 10.0) rtol = 0.03

        # index = 1 (log-uniform): E[ln p] = (ln pmin + ln pmax)/2
        pl1 = PowerLawMomentum(; pmin = 1.0, pmax = 100.0, index = 1.0)
        lps = [log(ParticleTransport.sample_momentum(rng, pl1)) for _ in 1:10_000]
        @test mean(lps) ≈ 0.5 * log(100.0) rtol = 0.03
    end

    # 4. ConditionedInjection
    let
        cond = ConditionedInjection(((-1.0, 1.0), (-1.0, 1.0)), (x, t) -> x[1]^2 + x[2]^2 <= 0.25)
        for _ in 1:200
            x = ParticleTransport.sample_position(rng, cond, 0.0)
            @test x[1]^2 + x[2]^2 <= 0.25
            @test x[3] == 0.0
        end
    end
end

@testset "InjectionSource & Continuous Ensemble Workflow" begin
    let
        rng = MersenneTwister(123)
        src = InjectionSource(;
            space = UniformBoxInjection(((-1.0, 1.0),)),
            momentum = PowerLawMomentum(; pmin = 1.0, pmax = 4.0, index = 2.0),
            tspan = (0.0, 1.0),
        )
        u0_s, t_inj = sample_state(rng, src)
        @test u0_s isa SVector{4}
        @test -1.0 <= u0_s[1] <= 1.0
        @test 1.0 <= get_momentum(u0_s) <= 4.0
        @test 0.0 <= t_inj <= 1.0

        # Run a continuous injection EnsembleProblem with constant flow u_x = 1.0
        param = prepare(; u = x -> SA[1.0, 0.0, 0.0], B = x -> SA[1.0, 0.0, 0.0], dims = 1)
        prob = TransportProblem(make_state(0.0, 1.0), (0.0, 2.0), param)
        ep = EnsembleProblem(prob; prob_func = injection_prob_func(src; rng = MersenneTwister(7)))
        sim = solve(ep, EM(), EnsembleSerial(); dt = 0.05, trajectories = 100, save_everystep = false)
        # Particles injected uniformly in x ∈ [-1, 1] at t_inj ∈ [0, 1], advected to t = 2:
        # E[x_end] = E[x0] + E[2 - t_inj] = 0 + 1.5 = 1.5
        xs = [get_position(s.u[end])[1] for s in sim.u]
        @test mean(xs) ≈ 1.5 atol = 0.25

        # Focused 5D injection source
        src_fte = InjectionSource(;
            space = PointInjection(SA[0.0, 0.0, 0.0]),
            momentum = DeltaMomentum(1.0),
            pitch_angle = :isotropic,
        )
        u5, _ = sample_state(rng, src_fte)
        @test u5 isa SVector{5}
        @test -1.0 <= get_pitch_angle(u5) <= 1.0
    end
end

end
