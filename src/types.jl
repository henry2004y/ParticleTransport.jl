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
    AbstractPitchAngleDiffusion

Abstract supertype for pitch-angle diffusion models (`D_μμ`) in focused transport.
"""
abstract type AbstractPitchAngleDiffusion end

"""
    AbstractTransportModel

Abstract supertype for particle transport equations.
"""
abstract type AbstractTransportModel end

"""
    ParkerTransport{D} <: AbstractTransportModel

Parker's transport equation in `D ∈ (1, 2, 3)` spatial dimensions.
"""
struct ParkerTransport{D} <: AbstractTransportModel end
ParkerTransport() = ParkerTransport{3}()

"""
    FocusedTransport{D} <: AbstractTransportModel

Focused transport equation in `D ∈ (1, 2, 3)` spatial dimensions plus momentum and
pitch-angle cosine `μ ∈ [-1, 1]`.
"""
struct FocusedTransport{D} <: AbstractTransportModel end
FocusedTransport() = FocusedTransport{3}()

"""
    AbstractCoordinateSystem

Abstract supertype for spatial coordinate systems.
"""
abstract type AbstractCoordinateSystem end

"""
    CartesianCoordinates <: AbstractCoordinateSystem

Cartesian spatial coordinates `(x, y, z)`.
"""
struct CartesianCoordinates <: AbstractCoordinateSystem end

"""
    SphericalCoordinates <: AbstractCoordinateSystem

Spherical spatial coordinates `(r, θ, φ)` with physical orthonormal field components
`(V_r, V_θ, V_φ)` and `(B_r, B_θ, B_φ)`.
"""
struct SphericalCoordinates <: AbstractCoordinateSystem end

"""
    AbstractAccelerationRegion

Abstract supertype for selective spatial acceleration-region masks.
"""
abstract type AbstractAccelerationRegion end

"""
    BoxAccelerationRegion{N, T} <: AbstractAccelerationRegion

Axis-aligned box mask specified by `N ∈ (1, 2, 3)` `(lo, hi)` coordinate bounds.
Momentum acceleration is active only when `lo <= x[i] <= hi` for all `i in 1:N`.
"""
struct BoxAccelerationRegion{N, T <: AbstractFloat} <: AbstractAccelerationRegion
    bounds::NTuple{N, Tuple{T, T}}
    function BoxAccelerationRegion(bounds::Tuple)
        n = length(bounds)
        n in (1, 2, 3) ||
            throw(ArgumentError("`bounds` must have 1, 2, or 3 (lo, hi) pairs."))
        T = promote_type(Float64, map(b -> promote_type(typeof(b[1]), typeof(b[2])), bounds)...)
        b_conv = ntuple(i -> (T(bounds[i][1]), T(bounds[i][2])), Val(n))
        return new{n, T}(b_conv)
    end
end

"""
    CustomAccelerationRegion(f) <: AbstractAccelerationRegion

Wrap a user predicate `f(x, t) -> Bool` controlling where momentum acceleration is active.
"""
struct CustomAccelerationRegion{F} <: AbstractAccelerationRegion
    f::F
end

Adapt.adapt_structure(to, r::BoxAccelerationRegion) = r
Adapt.adapt_structure(to, r::CustomAccelerationRegion) =
    CustomAccelerationRegion(Adapt.adapt(to, r.f))

"""
    in_acceleration_region(region, x, t)

Return `true` if momentum acceleration is active at `(x, t)`.
"""
@inline in_acceleration_region(::Nothing, x, t) = true

@inline function in_acceleration_region(r::BoxAccelerationRegion{N}, x, t) where {N}
    @inbounds for i in 1:N
        lo, hi = r.bounds[i]
        (x[i] < lo || x[i] > hi) && return false
    end
    return true
end

@inline in_acceleration_region(r::CustomAccelerationRegion, x, t) = Bool(r.f(x, t))

prepare_acc_region(::Nothing) = nothing
prepare_acc_region(r::AbstractAccelerationRegion) = r
prepare_acc_region(bounds::Tuple) = BoxAccelerationRegion(bounds)
prepare_acc_region(f) = CustomAccelerationRegion(f)

"""
    TransportParams{D, COORD, DRIFT, TRACK3D, U, B, K, M, ACC, T}

Immutable, `Adapt.jl`-compatible parameter container for Parker transport SDEs.
Construct with [`prepare`](@ref).
"""
struct TransportParams{
        D, COORD <: AbstractCoordinateSystem, DRIFT, TRACK3D,
        U, B, K, M, ACC, T <: AbstractFloat,
    }
    u::U
    B::B
    kappa::K
    dpp::M
    acc_region::ACC
    q::T
    m::T
    c::T
end

function Adapt.adapt_structure(
        to, p::TransportParams{D, COORD, DRIFT, TRACK3D}
    ) where {D, COORD, DRIFT, TRACK3D}
    u_a = Adapt.adapt(to, p.u)
    B_a = Adapt.adapt(to, p.B)
    k_a = Adapt.adapt(to, p.kappa)
    d_a = Adapt.adapt(to, p.dpp)
    a_a = Adapt.adapt(to, p.acc_region)
    return TransportParams{
        D, COORD, DRIFT, TRACK3D,
        typeof(u_a), typeof(B_a), typeof(k_a), typeof(d_a), typeof(a_a), typeof(p.q),
    }(u_a, B_a, k_a, d_a, a_a, p.q, p.m, p.c)
end

"""
    FocusedTransportParams{D, COORD, DRIFT, TRACK3D, U, B, K, M, MU, ACC, T}

Immutable, `Adapt.jl`-compatible parameter container for Focused Transport Equation SDEs
in `(x, y, z, ln p, μ)`. Construct with `prepare(; model = FocusedTransport(), ...)`.
"""
struct FocusedTransportParams{
        D, COORD <: AbstractCoordinateSystem, DRIFT, TRACK3D,
        U, B, K, M, MU, ACC, T <: AbstractFloat,
    }
    u::U
    B::B
    kappa::K
    dpp::M
    dmu::MU
    acc_region::ACC
    q::T
    m::T
    c::T
end

function Adapt.adapt_structure(
        to, p::FocusedTransportParams{D, COORD, DRIFT, TRACK3D}
    ) where {D, COORD, DRIFT, TRACK3D}
    u_a = Adapt.adapt(to, p.u)
    B_a = Adapt.adapt(to, p.B)
    k_a = Adapt.adapt(to, p.kappa)
    d_a = Adapt.adapt(to, p.dpp)
    mu_a = Adapt.adapt(to, p.dmu)
    a_a = Adapt.adapt(to, p.acc_region)
    return FocusedTransportParams{
        D, COORD, DRIFT, TRACK3D,
        typeof(u_a), typeof(B_a), typeof(k_a), typeof(d_a), typeof(mu_a), typeof(a_a),
        typeof(p.q),
    }(u_a, B_a, k_a, d_a, mu_a, a_a, p.q, p.m, p.c)
end

const AbstractTransportParams = Union{TransportParams, FocusedTransportParams}

spatial_dims(::TransportParams{D}) where {D} = D
spatial_dims(::FocusedTransportParams{D}) where {D} = D
coordinate_system(::TransportParams{D, COORD}) where {D, COORD} = COORD()
coordinate_system(::FocusedTransportParams{D, COORD}) where {D, COORD} = COORD()
has_drift(::TransportParams{D, COORD, DRIFT}) where {D, COORD, DRIFT} = DRIFT
has_drift(::FocusedTransportParams{D, COORD, DRIFT}) where {D, COORD, DRIFT} = DRIFT
tracks_3rd_dim(::TransportParams{D, COORD, DRIFT, TRACK3D}) where {D, COORD, DRIFT, TRACK3D} =
    TRACK3D
tracks_3rd_dim(
    ::FocusedTransportParams{D, COORD, DRIFT, TRACK3D}
) where {D, COORD, DRIFT, TRACK3D} = TRACK3D
