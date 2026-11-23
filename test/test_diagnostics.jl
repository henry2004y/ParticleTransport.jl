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

@testset "Escaped-Particle Diagnostics & Streaming" begin
    let
        # 1D pure diffusion on [0, 1] starting at x0 = 0.25:
        # Exact hitting probabilities: P(xmax) = x0 / L = 0.25, P(xmin) = 0.75
        bounds = ((0.0, 1.0),)
        param = prepare(; kappa = IsotropicDiffusion(0.5), dims = 1)
        prob = TransportProblem(make_state(0.25, 2.0), (0.0, 5.0), param)
        cb = TerminateOutside(bounds)
        pf = (p_base, ctx) -> remake(p_base; seed = 1000 + ctx.sim_id)
        p_edges = [0.5, 1.5, 3.0, 5.0]

        sim = solve(
            EnsembleProblem(prob; prob_func = pf), EM(), EnsembleSerial();
            dt = 2.0e-3, trajectories = 400, callback = cb, save_everystep = false,
        )
        esc = escaped_spectra(sim, bounds, p_edges)
        @test haskey(esc, :xmin) && haskey(esc, :xmax)
        n_left = sum(esc[:xmin].counts)
        n_right = sum(esc[:xmax].counts)
        @test n_left + n_right > 380
        @test n_right / (n_left + n_right) ≈ 0.25 atol = 0.07

        # Streaming escaped matches post-hoc escaped_spectra identically
        ep_esc = streaming_escaped_ensemble(prob, bounds, p_edges; prob_func = pf)
        sim_esc = solve(
            ep_esc, EM(), EnsembleSerial();
            dt = 2.0e-3, trajectories = 400, callback = cb, save_everystep = false,
        )
        @test sim_esc.u[:, 1] == esc[:xmin].counts
        @test sim_esc.u[:, 2] == esc[:xmax].counts

        @test escape_face(make_state(0.5, 1.0), bounds) === :interior
        @test escape_face(make_state(-0.1, 1.0), bounds) === :xmin
        @test escape_face(make_state(1.1, 1.0), bounds) === :xmax
    end
end

@testset "Multi-Dimensional LocalDistribution & Multi-Snapshot Streaming" begin
    let
        # 2D spatial + momentum local distribution
        states = [
            make_state(SA[0.2, 0.3, 0.0], 1.5),
            make_state(SA[0.2, 0.3, 0.0], 1.5),
            make_state(SA[0.8, 0.7, 0.0], 3.5),
        ]
        x_ed = [0.0, 0.5, 1.0]
        y_ed = [0.0, 0.5, 1.0]
        p_ed = [1.0, 2.0, 5.0]
        ld = local_distribution(states, (x_ed, y_ed), p_ed; weights = [1.0, 2.0, 0.5])
        @test size(ld.counts) == (2, 2, 2)
        @test ld.counts[1, 1, 1] ≈ 3.0
        @test ld.counts[2, 2, 2] ≈ 0.5
        @test sum(ld.counts) ≈ 3.5

        # 5D Focused Transport state with pitch-angle binning
        states_fte = [
            make_state(SA[0.2, 0.0, 0.0], 1.5, -0.5),
            make_state(SA[0.8, 0.0, 0.0], 3.5, 0.5),
        ]
        mu_ed = [-1.0, 0.0, 1.0]
        ld_fte = local_distribution(states_fte, x_ed, p_ed; mu_edges = mu_ed)
        @test size(ld_fte.counts) == (2, 2, 2)
        @test ld_fte.counts[1, 1, 1] ≈ 1.0
        @test ld_fte.counts[2, 2, 2] ≈ 1.0

        # Multi-snapshot streaming_distribution_ensemble
        param = prepare(; u = x -> SA[1.0, 0.0, 0.0], dims = 1)
        prob = TransportProblem(make_state(0.1, 1.5), (0.0, 1.0), param)
        ep_snap = streaming_distribution_ensemble(
            prob, [0.0, 0.5, 1.5], [1.0, 2.0]; saveat = [0.0, 1.0]
        )
        sim_snap = solve(ep_snap, EM(), EnsembleSerial(); dt = 0.05, trajectories = 10)
        @test size(sim_snap.u) == (2, 1, 2)
        # At t = 0: x = 0.1 ∈ [0.0, 0.5) => bin 1 has all 10 particles
        @test sim_snap.u[1, 1, 1] ≈ 10.0
        @test sim_snap.u[2, 1, 1] ≈ 0.0
        # At t = 1: x = 1.1 ∈ [0.5, 1.5) => bin 2 has all 10 particles
        @test sim_snap.u[1, 1, 2] ≈ 0.0
        @test sim_snap.u[2, 1, 2] ≈ 10.0
    end
end

end
