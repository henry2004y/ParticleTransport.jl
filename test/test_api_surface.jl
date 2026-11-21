module test_api_surface

using Test
using ParticleTransport
using StaticArrays: SA

@testset "Exports & Public Symbols" begin
    for sym in (
            :prepare, :TransportProblem, :solve,
            :Particle, :Proton, :Electron, :Positron,
            :ZeroDiffusion, :IsotropicDiffusion, :ConstantDiffusion,
            :PowerLawDiffusion, :CustomDiffusion,
            :ZeroMomentumDiffusion, :PowerLawMomentumDiffusion, :CustomMomentumDiffusion,
            :parker_drift, :parker_drift!, :parker_diffusion, :parker_diffusion!,
            :build_interpolator, :LazyTimeInterpolator,
            :TerminateOutside, :PeriodicBoundary, :ReflectingBoundary, :MomentumBounds,
            :SplitRule, :SplittingResult, :solve_with_splitting,
            :MomentumSpectrum, :momentum_spectrum, :spatial_density,
            :streaming_spectrum_ensemble, :spectrum_from_counts,
            :make_state, :get_position, :get_momentum, :get_energy,
            :momentum2velocity, :momentum2energy, :energy2momentum,
        )
        @test isdefined(ParticleTransport, sym)
        @test Base.isexported(ParticleTransport, sym)
    end

    for sym in (
            :AbstractField, :AbstractSpatialDiffusion, :AbstractMomentumDiffusion,
            :TransportParams, :Field, :ZeroField,
            :jacobian, :divergence, :curl_b_over_b2,
            :diffusion_tensor, :diffusion_sqrt, :div_diffusion_tensor,
            :momentum_diffusion_terms, :magnetic_drift,
        )
        @test isdefined(ParticleTransport, sym)
        @test Base.ispublic(ParticleTransport, sym)
    end
end

@testset "Kinematics Round-trip" begin
    let m = Proton.m, c_val = ParticleTransport.c
        for p in (1.0e-24, 1.0e-20, 1.0e-18, 1.0e-15)
            E = momentum2energy(p, m, c_val)
            @test energy2momentum(E, m, c_val) ≈ p rtol = 1.0e-12
        end
        # Non-relativistic limit (c = Inf)
        p_nr = 3.0
        E_nr = momentum2energy(p_nr, 2.0, Inf)
        @test E_nr ≈ 9.0 / 4.0
        @test energy2momentum(E_nr, 2.0, Inf) ≈ p_nr
        @test momentum2velocity(p_nr, 2.0, Inf) ≈ 1.5

        u = make_state(SA[1.0, 2.0, 3.0], 5.0)
        @test get_position(u) == SA[1.0, 2.0, 3.0]
        @test get_momentum(u) ≈ 5.0
    end
end

end
