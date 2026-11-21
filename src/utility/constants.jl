# Physical constants and relativistic momentum/energy conversions.

const c = C_SI
const eV = E_SI

"""
    make_state(x, p)

Construct a 4-component Parker SDE state `SVector{4}(x[1], x[2], x[3], log(p))`.
"""
@inline function make_state(x::AbstractVector, p::Real)
    p > zero(p) || throw(ArgumentError("Momentum p must be positive; got $p."))
    T = promote_type(eltype(x), typeof(p), Float64)
    return SVector{4, T}(x[1], x[2], x[3], log(p))
end

@inline function make_state(x::Real, p::Real)
    p > zero(p) || throw(ArgumentError("Momentum p must be positive; got $p."))
    T = promote_type(typeof(x), typeof(p), Float64)
    return SVector{4, T}(x, zero(T), zero(T), log(p))
end

"""
    get_position(u)

Extract the 3D spatial position `SVector{3}(u[1], u[2], u[3])` from a transport state.
"""
@inline get_position(u) = @inbounds SVector{3}(u[1], u[2], u[3])

"""
    get_momentum(u)

Extract the momentum `p = exp(u[4])` from a transport state.
"""
@inline get_momentum(u) = @inbounds exp(u[4])

"""
    momentum2velocity(p, m, c_light = c)

Convert momentum magnitude `p` to speed `v`. Set `c_light = Inf` for non-relativistic kinematics.
"""
@inline function momentum2velocity(p::Real, m::Real, c_light::Real = c)
    if isinf(c_light)
        return p / m
    end
    pm = p / m
    pmc = pm / c_light
    return pm / sqrt(one(pmc) + pmc * pmc)
end

"""
    momentum2energy(p, m, c_light = c)

Convert momentum magnitude `p` to kinetic energy `E_k`. Uses a cancellation-free formula
valid from non-relativistic (`p ≪ m*c`) to ultra-relativistic (`p ≫ m*c`) regimes.
"""
@inline function momentum2energy(p::Real, m::Real, c_light::Real = c)
    if isinf(c_light)
        return (p * p) / (2 * m)
    end
    pmc = p / (m * c_light)
    return (p * p) / (m * (sqrt(one(pmc) + pmc * pmc) + one(pmc)))
end

"""
    energy2momentum(E, m, c_light = c)

Convert kinetic energy `E` to momentum magnitude `p`.
"""
@inline function energy2momentum(E::Real, m::Real, c_light::Real = c)
    E >= zero(E) || throw(ArgumentError("Kinetic energy E must be non-negative; got $E."))
    if isinf(c_light)
        return sqrt(2 * m * E)
    end
    return sqrt(E * (E + 2 * m * c_light * c_light)) / c_light
end

"""
    get_energy(u, param::TransportParams)
    get_energy(u, m, c_light = c)

Extract kinetic energy from a transport state `u`.
"""
@inline get_energy(u, p::TransportParams) = momentum2energy(get_momentum(u), p.m, p.c)
@inline get_energy(u, m::Real, c_light::Real = c) = momentum2energy(get_momentum(u), m, c_light)
