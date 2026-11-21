# Field wrappers and spatial differential operators.

"""
    ZeroField <: AbstractField{false}

A zero 3-vector field whose value and derivatives are identically zero.
"""
struct ZeroField <: AbstractField{false} end

@inline (::ZeroField)(x::AbstractVector{T}) where {T} = zero(SVector{3, T})
@inline (::ZeroField)(x::AbstractVector{T}, t) where {T} = zero(SVector{3, T})

Adapt.adapt_structure(to, f::ZeroField) = f

"""
    is_time_dependent(f)

Return `true` if field `f` accepts `(x, t)`, `false` if it is static `f(x)`.
"""
is_time_dependent(f) = applicable(f, SVector(0.0, 0.0, 0.0), 0.0)
is_time_dependent(::AbstractField{itd}) where {itd} = itd
is_time_dependent(::AbstractFieldInterpolator) = false

"""
    Field(f; time_dependent = is_time_dependent(f))

Wrap a callable `f(x)` or `f(x, t)` as an `AbstractField`.
"""
struct Field{itd, F} <: AbstractField{itd}
    f::F
    function Field{itd, F}(f::F) where {itd, F}
        itd isa Bool || throw(ArgumentError("itd must be a Bool."))
        return new{itd, F}(f)
    end
end

Field(f; time_dependent::Bool = is_time_dependent(f)) =
    Field{time_dependent, typeof(f)}(f)

@inline (f::Field{true})(x, t) = SVector{3}(f.f(x, t))
@inline function (f::Field{true})(x)
    throw(ArgumentError("Time-dependent field requires a time argument `t`."))
end
@inline (f::Field{false})(x, t) = SVector{3}(f.f(x))
@inline (f::Field{false})(x) = SVector{3}(f.f(x))

Adapt.adapt_structure(to, f::Field{itd}) where {itd} = Field{itd, typeof(Adapt.adapt(to, f.f))}(Adapt.adapt(to, f.f))

prepare_field(f; kwargs...) = Field(f; kwargs...)
prepare_field(f::Field; kwargs...) = f
prepare_field(f::AbstractFieldInterpolator; kwargs...) = f
prepare_field(f::ZeroField; kwargs...) = f
prepare_field(::Nothing; kwargs...) = ZeroField()

"""
    jacobian(f, x, t)

Compute the 3×3 spatial Jacobian `J[i, j] = ∂f_i / ∂x_j` of vector field `f` at `(x, t)`.
"""
@inline jacobian(::ZeroField, x::AbstractVector{T}, t) where {T} = zero(SMatrix{3, 3, T})

@inline function jacobian(f::Field{false}, x, t)
    return SMatrix{3, 3}(ForwardDiff.jacobian(r -> SVector{3}(f.f(r)), SVector{3}(x)))
end

@inline function jacobian(f::Field{true}, x, t)
    return SMatrix{3, 3}(ForwardDiff.jacobian(r -> SVector{3}(f.f(r, t)), SVector{3}(x)))
end

@inline function jacobian(f, x, t)
    return SMatrix{3, 3}(ForwardDiff.jacobian(r -> SVector{3}(f(r, t)), SVector{3}(x)))
end

"""
    divergence(f, x, t, ::Val{D} = Val(3))

Compute the spatial divergence `∑_{i=1}^D ∂f_i / ∂x_i` of field `f` at `(x, t)`.
"""
@inline divergence(::ZeroField, x::AbstractVector{T}, t, ::Val{D} = Val(3)) where {T, D} = zero(T)

@inline function divergence(f, x, t, ::Val{D} = Val(3)) where {D}
    J = jacobian(f, x, t)
    @inbounds if D == 1
        return J[1, 1]
    elseif D == 2
        return J[1, 1] + J[2, 2]
    else
        return J[1, 1] + J[2, 2] + J[3, 3]
    end
end

"""
    curl_b_over_b2(B_field, x, t)

Compute `∇ × (B / B²)` at `(x, t)` analytically from `B` and its spatial Jacobian `∂B_i/∂x_j`.
Returns the zero vector if `|B| == 0`.
"""
@inline function curl_b_over_b2(::ZeroField, x::AbstractVector{T}, t) where {T}
    return zero(SVector{3, T})
end

@inline function curl_b_over_b2(B_field, x, t)
    B = B_field(x, t)
    B2 = dot(B, B)
    T = eltype(B)
    B2 > zero(T) || return zero(SVector{3, T})
    J = jacobian(B_field, x, t)
    invB2 = inv(B2)
    two_invB4 = 2 * invB2 * invB2
    # ∂(B_i / B²)/∂x_j = J[i,j]/B² - 2 B_i (B ⋅ J[:,j]) / B⁴
    @inbounds begin
        bdot_j1 = B[1] * J[1, 1] + B[2] * J[2, 1] + B[3] * J[3, 1]
        bdot_j2 = B[1] * J[1, 2] + B[2] * J[2, 2] + B[3] * J[3, 2]
        bdot_j3 = B[1] * J[1, 3] + B[2] * J[2, 3] + B[3] * J[3, 3]

        dg3_dx2 = J[3, 2] * invB2 - two_invB4 * B[3] * bdot_j2
        dg2_dx3 = J[2, 3] * invB2 - two_invB4 * B[2] * bdot_j3

        dg1_dx3 = J[1, 3] * invB2 - two_invB4 * B[1] * bdot_j3
        dg3_dx1 = J[3, 1] * invB2 - two_invB4 * B[3] * bdot_j1

        dg2_dx1 = J[2, 1] * invB2 - two_invB4 * B[2] * bdot_j1
        dg1_dx2 = J[1, 2] * invB2 - two_invB4 * B[1] * bdot_j2

        return SVector{3}(dg3_dx2 - dg2_dx3, dg1_dx3 - dg3_dx1, dg2_dx1 - dg1_dx2)
    end
end
