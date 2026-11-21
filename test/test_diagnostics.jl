module test_diagnostics

using Test
using Random
using StochasticDiffEq: EM
using StaticArrays: SA
using ParticleTransport

@testset "MomentumSpectrum, SpatialDensity & Streaming Reduction" begin
    let
        D0 = 0.3
        param = prepare(;
            kappa = IsotropicDiffusion(0.2),
            dpp = PowerLawMomentumDiffusion(; D0, p0 = 1.0, gamma = 2.0),
            dims = 1,
        )
        u0 = make_state(0.0, 1.0)
        prob = TransportProblem(u0, (0.0, 1.0), param)
        edges = 10.0 .^ range(-1.0, 1.5; length = 15)

        # Run standard ensemble with deterministic per-trajectory seeds
        pf = (p_base, ctx) -> remake(p_base; seed = ctx.sim_id)
        sim = solve(EnsembleProblem(prob; prob_func = pf), EM(), EnsembleSerial(); dt = 0.02, trajectories = 200, save_everystep = false)
        spec_post = momentum_spectrum(sim, edges)

        # Run streaming O(1)-memory ensemble with identical seeds
        ep_stream = streaming_spectrum_ensemble(prob, edges; prob_func = pf)
        sim_stream = solve(ep_stream, EM(), EnsembleSerial(); dt = 0.02, trajectories = 200, save_everystep = false)
        spec_stream = spectrum_from_counts(sim_stream.u, edges)

        @test spec_stream.counts == spec_post.counts
        @test spec_stream.f ≈ spec_post.f
        @test sum(spec_post.counts) > 0

        # Spatial density
        x_edges = collect(range(-3.0, 3.0; length = 13))
        sd = spatial_density(sim, x_edges; dim = 1)
        @test length(sd.density) == 12
        @test sum(sd.counts) > 0
        @test sd.density ≈ sd.counts ./ diff(x_edges)
    end
end

end
