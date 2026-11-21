# SciMLBase.SDEProblem constructors for Parker transport.

"""
    TransportProblem(u0, tspan, param::TransportParams; kwargs...)

Construct a standard `SciMLBase.SDEProblem` for Parker's transport equation.

- When `u0` is an `SVector{4}`, builds an out-of-place `SDEProblem{false}` using
  [`parker_drift`](@ref) and [`parker_diffusion`](@ref) with `SMatrix{4, 4}` noise prototype.
- When `u0` is a mutable `AbstractVector` of length 4, builds an in-place `SDEProblem{true}`
  using [`parker_drift!`](@ref) and [`parker_diffusion!`](@ref) with a `4×4` matrix noise prototype.

Additional keyword arguments are forwarded directly to `SciMLBase.SDEProblem`.
"""
function TransportProblem(
        u0::SVector{4, T}, tspan, param::TransportParams;
        noise_rate_prototype = zero(SMatrix{4, 4, float(T)}),
        kwargs...
    ) where {T <: Real}
    u0_f = float.(u0)
    return SDEProblem{false}(
        parker_drift, parker_diffusion, u0_f, tspan, param;
        noise_rate_prototype, kwargs...
    )
end

function TransportProblem(
        u0::AbstractVector{T}, tspan, param::TransportParams;
        noise_rate_prototype = zeros(float(T), 4, 4),
        kwargs...
    ) where {T <: Real}
    length(u0) == 4 || throw(
        ArgumentError("Initial state `u0` must have length 4: (x, y, z, ln p); got $(length(u0)).")
    )
    u0_vec = float.(collect(u0))
    return SDEProblem{true}(
        parker_drift!, parker_diffusion!, u0_vec, tspan, param;
        noise_rate_prototype, kwargs...
    )
end
