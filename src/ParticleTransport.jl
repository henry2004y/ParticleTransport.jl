module ParticleTransport

using LinearAlgebra: dot, norm
using FastInterpolations: constant_interp, linear_interp, cardinal_interp,
    gradient, deriv1, FillExtrap, ClampExtrap, WrapExtrap, AbstractExtrap,
    PeriodicBC, NoBC, NoExtrap
import ForwardDiff
import Adapt
using StaticArrays: SVector, SMatrix, SA
using Random: AbstractRNG, default_rng, randn
using SciMLBase: SDEProblem, EnsembleProblem, EnsembleContext, EnsembleThreads,
    EnsembleSerial, EnsembleDistributed, EnsembleSplitThreads, EnsembleSolution,
    DiscreteCallback, CallbackSet, terminate!, remake, solve
using PlasmaPopulations: Particle, Proton, Electron, Positron, C_SI, E_SI
using PrecompileTools: @setup_workload, @compile_workload

export prepare, TransportProblem, solve
export Particle, Proton, Electron, Positron
export ParkerTransport, FocusedTransport
export CartesianCoordinates, SphericalCoordinates
export BoxAccelerationRegion, CustomAccelerationRegion
export ZeroDiffusion, IsotropicDiffusion, ConstantDiffusion, PowerLawDiffusion,
    QLTDiffusion, NLGCDiffusion, CustomDiffusion
export ZeroMomentumDiffusion, PowerLawMomentumDiffusion,
    AlfvenWaveMomentumDiffusion, FlowShearMomentumDiffusion, CustomMomentumDiffusion
export ZeroPitchAngleDiffusion, ConstantPitchAngleDiffusion,
    QLTPitchAngleDiffusion, CustomPitchAngleDiffusion
export parker_drift, parker_drift!, parker_diffusion, parker_diffusion!
export focused_drift, focused_drift!, focused_diffusion, focused_diffusion!
export build_interpolator, LazyTimeInterpolator
export TerminateOutside, PeriodicBoundary, ReflectingBoundary,
    MomentumBounds, PitchAngleBounds
export SplitRule, SplittingResult, solve_with_splitting
export DeltaMomentum, MaxwellianMomentum, PowerLawMomentum,
    PointInjection, UniformBoxInjection, ConditionedInjection,
    InjectionSource, sample_state, injection_prob_func
export MomentumSpectrum, momentum_spectrum, spatial_density,
    streaming_spectrum_ensemble, spectrum_from_counts,
    escape_face, escaped_spectra, streaming_escaped_ensemble,
    LocalDistribution, local_distribution, streaming_distribution_ensemble
export make_state, get_position, get_momentum, get_energy, get_pitch_angle,
    momentum2velocity, momentum2energy, energy2momentum
export EnsembleProblem, EnsembleSerial, EnsembleThreads, EnsembleDistributed,
    EnsembleSplitThreads, remake, DiscreteCallback, CallbackSet

public AbstractField, AbstractFieldInterpolator,
    AbstractSpatialDiffusion, AbstractMomentumDiffusion, AbstractPitchAngleDiffusion,
    AbstractTransportModel, AbstractCoordinateSystem, AbstractAccelerationRegion,
    AbstractTransportParams, TransportParams, FocusedTransportParams,
    AbstractMomentumSampler, AbstractSpatialSampler,
    Field, ZeroField, FieldInterpolator1D, FieldInterpolator2D, FieldInterpolator3D,
    is_time_dependent, jacobian, divergence, curl_b_over_b2,
    spherical_divergence, spherical_curl, spherical_curl_b_over_b2,
    spherical_diffusion_sqrt, spherical_diffusion_drift,
    diffusion_coefficients, diffusion_tensor, diffusion_sqrt, div_diffusion_tensor,
    perp_diffusion_sqrt, div_perp_diffusion_tensor,
    momentum_diffusion_terms, pitch_angle_diffusion_terms,
    magnetic_drift, focused_magnetic_drift, in_acceleration_region,
    spatial_dims, coordinate_system, has_drift, tracks_3rd_dim,
    sample_momentum, sample_position, c, eV,
    FillExtrap, ClampExtrap, WrapExtrap, PeriodicBC

include("types.jl")
include("utility/constants.jl")
include("utility/fields.jl")
include("utility/spherical.jl")
include("utility/interpolation.jl")
include("diffusion.jl")
include("prepare.jl")
include("equations.jl")
include("focused.jl")
include("problem.jl")
include("callbacks.jl")
include("splitting.jl")
include("injection.jl")
include("diagnostics.jl")
include("precompile.jl")

end
