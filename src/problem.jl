# SciMLBase.SDEProblem constructors for Parker and Focused transport.

"""
    TransportProblem(u0, tspan, param::TransportParams; kwargs...)
    TransportProblem(u0, tspan, param::FocusedTransportParams; kwargs...)

Construct a standard `SciMLBase.SDEProblem` for Parker's transport equation (4-component
state `(x, y, z, ln p)`) or the Focused Transport Equation (5-component state
`(x, y, z, ln p, μ)`).

- When `u0` is an `SVector`, builds an out-of-place `SDEProblem{false}`.
- When `u0` is a mutable `AbstractVector`, builds an in-place `SDEProblem{true}`.

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
        ArgumentError(
            "Initial state `u0` must have length 4: (x, y, z, ln p); got $(length(u0))."
        )
    )
    u0_vec = float.(collect(u0))
    return SDEProblem{true}(
        parker_drift!, parker_diffusion!, u0_vec, tspan, param;
        noise_rate_prototype, kwargs...
    )
end

function TransportProblem(
        u0::SVector{5, T}, tspan, param::FocusedTransportParams;
        noise_rate_prototype = zero(SMatrix{5, 5, float(T)}),
        kwargs...
    ) where {T <: Real}
    u0_f = float.(u0)
    return SDEProblem{false}(
        focused_drift, focused_diffusion, u0_f, tspan, param;
        noise_rate_prototype, kwargs...
    )
end

function TransportProblem(
        u0::AbstractVector{T}, tspan, param::FocusedTransportParams;
        noise_rate_prototype = zeros(float(T), 5, 5),
        kwargs...
    ) where {T <: Real}
    length(u0) == 5 || throw(
        ArgumentError(
            "Initial state `u0` must have length 5: (x, y, z, ln p, μ); got $(length(u0))."
        )
    )
    u0_vec = float.(collect(u0))
    return SDEProblem{true}(
        focused_drift!, focused_diffusion!, u0_vec, tspan, param;
        noise_rate_prototype, kwargs...
    )
end
