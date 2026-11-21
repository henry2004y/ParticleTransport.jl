module ParticleTransport

using LinearAlgebra: dot, norm
using FastInterpolations: constant_interp, linear_interp, cardinal_interp,
    gradient, deriv1, FillExtrap, ClampExtrap, WrapExtrap, AbstractExtrap,
    PeriodicBC, NoBC, NoExtrap
import ForwardDiff
import Adapt
using StaticArrays: SVector, SMatrix, SA
using SciMLBase: SDEProblem, EnsembleProblem, EnsembleThreads, EnsembleSerial,
    EnsembleDistributed, EnsembleSplitThreads, EnsembleSolution,
    DiscreteCallback, CallbackSet, terminate!, remake, solve
using PlasmaPopulations: Particle, Proton, Electron, Positron, C_SI, E_SI
using PrecompileTools: @setup_workload, @compile_workload

export prepare, TransportProblem, solve
export Particle, Proton, Electron, Positron
export ZeroDiffusion, IsotropicDiffusion, ConstantDiffusion, PowerLawDiffusion,
    CustomDiffusion
export ZeroMomentumDiffusion, PowerLawMomentumDiffusion, CustomMomentumDiffusion
export parker_drift, parker_drift!, parker_diffusion, parker_diffusion!
export build_interpolator, LazyTimeInterpolator
export TerminateOutside, PeriodicBoundary, ReflectingBoundary, MomentumBounds
export SplitRule, SplittingResult, solve_with_splitting
export MomentumSpectrum, momentum_spectrum, spatial_density,
    streaming_spectrum_ensemble, spectrum_from_counts
export make_state, get_position, get_momentum, get_energy,
    momentum2velocity, momentum2energy, energy2momentum
export EnsembleProblem, EnsembleSerial, EnsembleThreads, EnsembleDistributed,
    EnsembleSplitThreads, remake, DiscreteCallback, CallbackSet

public AbstractField, AbstractFieldInterpolator,
    AbstractSpatialDiffusion, AbstractMomentumDiffusion,
    AbstractTransportModel, ParkerTransport, TransportParams,
    Field, ZeroField, FieldInterpolator1D, FieldInterpolator2D, FieldInterpolator3D,
    is_time_dependent, jacobian, divergence, curl_b_over_b2,
    diffusion_coefficients, diffusion_tensor, diffusion_sqrt, div_diffusion_tensor,
    momentum_diffusion_terms, magnetic_drift,
    spatial_dims, has_drift, c, eV,
    FillExtrap, ClampExtrap, WrapExtrap, PeriodicBC

include("types.jl")
include("utility/constants.jl")
include("utility/fields.jl")
include("utility/interpolation.jl")
include("diffusion.jl")
include("prepare.jl")
include("equations.jl")
include("problem.jl")
include("callbacks.jl")
include("splitting.jl")
include("diagnostics.jl")
include("precompile.jl")

end
