module test_api_surface

using Test
using ParticleTransport
using StaticArrays: SA

@testset "Exports & Public Symbols" begin
    for sym in (
            :prepare, :TransportProblem, :solve,
            :Particle, :Proton, :Electron, :Positron,
            :ParkerTransport, :FocusedTransport,
            :CartesianCoordinates, :SphericalCoordinates,
            :BoxAccelerationRegion, :CustomAccelerationRegion,
            :ZeroDiffusion, :IsotropicDiffusion, :ConstantDiffusion,
            :PowerLawDiffusion, :QLTDiffusion, :NLGCDiffusion, :CustomDiffusion,
            :ZeroMomentumDiffusion, :PowerLawMomentumDiffusion,
            :AlfvenWaveMomentumDiffusion, :FlowShearMomentumDiffusion,
            :CustomMomentumDiffusion,
            :ZeroPitchAngleDiffusion, :ConstantPitchAngleDiffusion,
            :QLTPitchAngleDiffusion, :CustomPitchAngleDiffusion,
            :parker_drift, :parker_drift!, :parker_diffusion, :parker_diffusion!,
            :focused_drift, :focused_drift!, :focused_diffusion, :focused_diffusion!,
            :build_interpolator, :LazyTimeInterpolator,
            :TerminateOutside, :PeriodicBoundary, :ReflectingBoundary,
            :MomentumBounds, :PitchAngleBounds,
            :SplitRule, :SplittingResult, :solve_with_splitting,
            :DeltaMomentum, :MaxwellianMomentum, :PowerLawMomentum,
            :PointInjection, :UniformBoxInjection, :ConditionedInjection,
            :InjectionSource, :sample_state, :injection_prob_func,
            :MomentumSpectrum, :momentum_spectrum, :spatial_density,
            :streaming_spectrum_ensemble, :spectrum_from_counts,
            :escape_face, :escaped_spectra, :streaming_escaped_ensemble,
            :LocalDistribution, :local_distribution, :streaming_distribution_ensemble,
            :make_state, :get_position, :get_momentum, :get_energy, :get_pitch_angle,
            :momentum2velocity, :momentum2energy, :energy2momentum,
        )
        @test isdefined(ParticleTransport, sym)
        @test Base.isexported(ParticleTransport, sym)
    end

    for sym in (
            :AbstractField, :AbstractSpatialDiffusion, :AbstractMomentumDiffusion,
            :AbstractPitchAngleDiffusion, :AbstractTransportModel,
            :AbstractCoordinateSystem, :AbstractAccelerationRegion,
            :AbstractTransportParams, :TransportParams, :FocusedTransportParams,
            :AbstractMomentumSampler, :AbstractSpatialSampler,
            :Field, :ZeroField,
            :jacobian, :divergence, :curl_b_over_b2,
            :spherical_divergence, :spherical_curl, :spherical_curl_b_over_b2,
            :spherical_diffusion_sqrt, :spherical_diffusion_drift,
            :diffusion_tensor, :diffusion_sqrt, :div_diffusion_tensor,
            :perp_diffusion_sqrt, :div_perp_diffusion_tensor,
            :momentum_diffusion_terms, :pitch_angle_diffusion_terms,
            :magnetic_drift, :focused_magnetic_drift, :in_acceleration_region,
            :spatial_dims, :coordinate_system, :has_drift, :tracks_3rd_dim,
            :sample_momentum, :sample_position,
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

        u5 = make_state(SA[1.0, 2.0, 3.0], 5.0, -0.4)
        @test get_position(u5) == SA[1.0, 2.0, 3.0]
        @test get_momentum(u5) ≈ 5.0
        @test get_pitch_angle(u5) ≈ -0.4
    end
end

end
