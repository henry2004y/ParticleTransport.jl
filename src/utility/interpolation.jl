# Gridded field interpolation powered by FastInterpolations.jl.

struct FieldInterpolator3D{T} <: AbstractFieldInterpolator
    itp::T
end

@inline @inbounds (fi::FieldInterpolator3D)(x) = fi.itp((x[1], x[2], x[3]))
@inline (fi::FieldInterpolator3D)(x, t) = fi(x)

@inline @inbounds function jacobian(fi::FieldInterpolator3D, x, t)
    grads = gradient(fi.itp, (x[1], x[2], x[3]))
    return hcat(grads[1], grads[2], grads[3])
end

Adapt.adapt_structure(to, fi::FieldInterpolator3D) = FieldInterpolator3D(Adapt.adapt(to, fi.itp))

struct FieldInterpolator2D{T} <: AbstractFieldInterpolator
    itp::T
end

@inline @inbounds (fi::FieldInterpolator2D)(x) = fi.itp((x[1], x[2]))
@inline (fi::FieldInterpolator2D)(x, t) = fi(x)

@inline @inbounds function jacobian(fi::FieldInterpolator2D, x, t)
    grads = gradient(fi.itp, (x[1], x[2]))
    z = zero(grads[1])
    return hcat(grads[1], grads[2], z)
end

Adapt.adapt_structure(to, fi::FieldInterpolator2D) = FieldInterpolator2D(Adapt.adapt(to, fi.itp))

struct FieldInterpolator1D{T} <: AbstractFieldInterpolator
    itp::T
    dir::Int
end

@inline @inbounds (fi::FieldInterpolator1D)(x) = fi.itp(x[fi.dir])
@inline (fi::FieldInterpolator1D)(x, t) = fi(x)

@inline @inbounds function jacobian(fi::FieldInterpolator1D, x, t)
    d = deriv1(fi.itp)(x[fi.dir])
    z = zero(d)
    if fi.dir == 1
        return hcat(d, z, z)
    elseif fi.dir == 2
        return hcat(z, d, z)
    else
        return hcat(z, z, d)
    end
end

Adapt.adapt_structure(to, fi::FieldInterpolator1D) =
    FieldInterpolator1D(Adapt.adapt(to, fi.itp), fi.dir)

"""
    LazyTimeInterpolator(times, fields)

Linearly interpolate between a sequence of static field snapshots `fields` sampled at
strictly increasing `times`.
"""
struct LazyTimeInterpolator{T <: AbstractVector, F <: AbstractVector} <: AbstractField{true}
    times::T
    fields::F
    function LazyTimeInterpolator(times::T, fields::F) where {T <: AbstractVector, F <: AbstractVector}
        length(times) == length(fields) ||
            throw(ArgumentError("`times` and `fields` must have the same length."))
        length(times) >= 2 ||
            throw(ArgumentError("At least two time snapshots are required."))
        all(d -> d > zero(d), diff(times)) ||
            throw(ArgumentError("`times` must be strictly increasing."))
        return new{T, F}(times, fields)
    end
end

@inline function _time_bracket(times, t)
    n = length(times)
    if t <= times[1]
        return 1, 1, zero(t)
    elseif t >= times[n]
        return n, n, zero(t)
    end
    idx = clamp(searchsortedlast(times, t), 1, n - 1)
    α = (t - times[idx]) / (times[idx + 1] - times[idx])
    return idx, idx + 1, α
end

@inline function (lti::LazyTimeInterpolator)(x, t)
    i1, i2, α = _time_bracket(lti.times, t)
    i1 == i2 && return SVector{3}(lti.fields[i1](x, t))
    f1 = SVector{3}(lti.fields[i1](x, t))
    f2 = SVector{3}(lti.fields[i2](x, t))
    return (one(α) - α) * f1 + α * f2
end

@inline function jacobian(lti::LazyTimeInterpolator, x, t)
    i1, i2, α = _time_bracket(lti.times, t)
    i1 == i2 && return jacobian(lti.fields[i1], x, t)
    J1 = jacobian(lti.fields[i1], x, t)
    J2 = jacobian(lti.fields[i2], x, t)
    return (one(α) - α) * J1 + α * J2
end

Adapt.adapt_structure(to, lti::LazyTimeInterpolator) =
    LazyTimeInterpolator(Adapt.adapt(to, lti.times), Adapt.adapt(to, lti.fields))

function _as_svector_array(A::AbstractArray{<:SVector{3}})
    return A
end

function _as_svector_array(A::AbstractArray{T}) where {T <: Real}
    size(A, 1) == 3 || throw(
        ArgumentError("Leading dimension of a real vector-field array must be 3; got $(size(A, 1)).")
    )
    return reinterpret(reshape, SVector{3, T}, Array(A))
end

function _build_fast_itp(grids, A_sv, order, bc_or_extrap)
    axes_t = grids isa Tuple ? grids : (grids,)
    for g in axes_t
        length(g) >= 2 && all(isfinite, g) && all(d -> d > zero(d), diff(g)) ||
            throw(ArgumentError("Grid nodes must be finite and strictly increasing."))
    end
    policies = bc_or_extrap isa Tuple ? bc_or_extrap : map(Returns(bc_or_extrap), axes_t)
    length(policies) == length(axes_t) ||
        throw(ArgumentError("Provide one boundary/extrapolation policy per grid axis."))
    policies = map(policies) do policy
        if policy isa FillExtrap && policy.fill_value isa Number
            ET = eltype(A_sv)
            val = ET(ntuple(_ -> policy.fill_value, 3))
            return FillExtrap(val)
        end
        return policy
    end
    bcs = map(p -> p isa AbstractExtrap ? NoBC() : p, policies)
    extrap = map(p -> p isa AbstractExtrap ? p : NoExtrap(), policies)
    bc_val = grids isa Tuple ? bcs : only(bcs)
    extrap_val = grids isa Tuple ? extrap : only(extrap)

    if order == 1
        return linear_interp(grids, A_sv; bc = bc_val, extrap = extrap_val)
    elseif order == 3
        return cardinal_interp(grids, A_sv; bc = bc_val, extrap = extrap_val)
    elseif order == 0
        return constant_interp(grids, A_sv; bc = bc_val, extrap = extrap_val)
    else
        throw(ArgumentError("Unsupported interpolation order $order; expected 0, 1, or 3."))
    end
end

"""
    build_interpolator(A, grids; order = 1, bc = ClampExtrap(), dir = 1)

Construct a 1D, 2D, or 3D vector-field interpolator from array `A` and grid(s) `grids`.
`A` may be an array of `SVector{3}` or a real array with leading dimension `3`.
"""
function build_interpolator(
        A::AbstractArray, grids;
        order::Int = 1, bc = ClampExtrap(), dir::Int = 1
    )
    A_sv = _as_svector_array(A)
    nd = grids isa Tuple ? length(grids) : 1
    ndims(A_sv) == nd || throw(
        ArgumentError("Field array has $(ndims(A_sv)) spatial dimensions but $nd grid axes were given.")
    )
    if nd == 1
        g = grids isa Tuple ? only(grids) : grids
        itp = _build_fast_itp(g, A_sv, order, bc)
        return FieldInterpolator1D(itp, dir)
    elseif nd == 2
        itp = _build_fast_itp(grids, A_sv, order, bc)
        return FieldInterpolator2D(itp)
    elseif nd == 3
        itp = _build_fast_itp(grids, A_sv, order, bc)
        return FieldInterpolator3D(itp)
    else
        throw(ArgumentError("Only 1D, 2D, and 3D grids are supported; got $nd."))
    end
end
