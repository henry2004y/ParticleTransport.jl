# Core abstract types and parameter containers.

"""
    AbstractField{itd} <: Function

Abstract supertype for analytical and interpolated vector fields, parameterized by
time-dependence `itd::Bool`.
"""
abstract type AbstractField{itd} <: Function end

"""
    AbstractFieldInterpolator <: Function

Abstract supertype for gridded field interpolators.
"""
abstract type AbstractFieldInterpolator <: Function end

"""
    AbstractSpatialDiffusion

Abstract supertype for spatial diffusion tensor models (`κ_∥`, `κ_⊥`).
"""
abstract type AbstractSpatialDiffusion end

"""
    AbstractMomentumDiffusion

Abstract supertype for momentum-space diffusion models (`D_pp`).
"""
abstract type AbstractMomentumDiffusion end

"""
    AbstractTransportModel

Abstract supertype for particle transport equations.
"""
abstract type AbstractTransportModel end

"""
    ParkerTransport{D} <: AbstractTransportModel

Parker's transport equation in `D ∈ (1, 2, 3)` Cartesian spatial dimensions.
"""
struct ParkerTransport{D} <: AbstractTransportModel end

"""
    TransportParams{D, DRIFT, U, B, K, M, T}

Immutable, `Adapt.jl`-compatible parameter container for Parker transport SDEs.
Construct with [`prepare`](@ref).
"""
struct TransportParams{D, DRIFT, U, B, K, M, T <: AbstractFloat}
    u::U
    B::B
    kappa::K
    dpp::M
    q::T
    m::T
    c::T
end

function Adapt.adapt_structure(to, p::TransportParams{D, DRIFT}) where {D, DRIFT}
    return TransportParams{
        D, DRIFT,
        typeof(Adapt.adapt(to, p.u)),
        typeof(Adapt.adapt(to, p.B)),
        typeof(Adapt.adapt(to, p.kappa)),
        typeof(Adapt.adapt(to, p.dpp)),
        typeof(p.q),
    }(
        Adapt.adapt(to, p.u),
        Adapt.adapt(to, p.B),
        Adapt.adapt(to, p.kappa),
        Adapt.adapt(to, p.dpp),
        p.q, p.m, p.c,
    )
end

spatial_dims(::TransportParams{D}) where {D} = D
has_drift(::TransportParams{D, DRIFT}) where {D, DRIFT} = DRIFT
