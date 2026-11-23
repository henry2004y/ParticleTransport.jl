# Composable SciML callbacks for spatial, momentum, and pitch-angle boundary conditions.

@inline _set_component(u::SVector, i::Int, val) = Base.setindex(u, val, i)
@inline function _set_component(u::AbstractVector, i::Int, val)
    u[i] = val
    return u
end

"""
    TerminateOutside(bounds; save_positions = (false, false))

Return a `DiscreteCallback` that terminates a trajectory when any active spatial coordinate
leaves `bounds`, specified as a tuple of `(lo, hi)` pairs (length 1, 2, or 3).
"""
function TerminateOutside(bounds::Tuple; save_positions = (false, false))
    n = length(bounds)
    n in (1, 2, 3) || throw(ArgumentError("`bounds` must have 1, 2, or 3 (lo, hi) pairs."))
    condition = function (u, t, integrator)
        @inbounds for i in 1:n
            lo, hi = bounds[i]
            (u[i] < lo || u[i] > hi) && return true
        end
        return false
    end
    affect! = integrator -> terminate!(integrator)
    return DiscreteCallback(condition, affect!; save_positions)
end

"""
    PeriodicBoundary(bounds; save_positions = (false, true))

Return a `DiscreteCallback` that wraps spatial coordinates periodically into `bounds`
(a tuple of `(lo, hi)` pairs of length 1, 2, or 3).
"""
function PeriodicBoundary(bounds::Tuple; save_positions = (false, true))
    n = length(bounds)
    n in (1, 2, 3) || throw(ArgumentError("`bounds` must have 1, 2, or 3 (lo, hi) pairs."))
    condition = function (u, t, integrator)
        @inbounds for i in 1:n
            lo, hi = bounds[i]
            (u[i] < lo || u[i] >= hi) && return true
        end
        return false
    end
    affect! = function (integrator)
        u = integrator.u
        @inbounds for i in 1:n
            lo, hi = bounds[i]
            L = hi - lo
            if u[i] < lo || u[i] >= hi
                u = _set_component(u, i, lo + mod(u[i] - lo, L))
            end
        end
        integrator.u = u
        return nothing
    end
    return DiscreteCallback(condition, affect!; save_positions)
end

"""
    ReflectingBoundary(bounds; save_positions = (false, true))

Return a `DiscreteCallback` that reflects spatial coordinates across `bounds`
(a tuple of `(lo, hi)` pairs of length 1, 2, or 3).
"""
function ReflectingBoundary(bounds::Tuple; save_positions = (false, true))
    n = length(bounds)
    n in (1, 2, 3) || throw(ArgumentError("`bounds` must have 1, 2, or 3 (lo, hi) pairs."))
    condition = function (u, t, integrator)
        @inbounds for i in 1:n
            lo, hi = bounds[i]
            (u[i] < lo || u[i] > hi) && return true
        end
        return false
    end
    affect! = function (integrator)
        u = integrator.u
        @inbounds for i in 1:n
            lo, hi = bounds[i]
            x = u[i]
            if x < lo || x > hi
                L = hi - lo
                # Fold into [0, 2L) then reflect [L, 2L) back into [0, L]
                r = mod(x - lo, 2 * L)
                x_ref = clamp(r <= L ? lo + r : hi - (r - L), lo, hi)
                u = _set_component(u, i, x_ref)
            end
        end
        integrator.u = u
        return nothing
    end
    return DiscreteCallback(condition, affect!; save_positions)
end

"""
    MomentumBounds(; pmin = 0.0, pmax = Inf, action = :terminate, save_positions = (false, true))

Return a `DiscreteCallback` enforcing momentum bounds `pmin <= exp(u[4]) <= pmax`.
`action` may be `:terminate` (stop trajectory) or `:reflect` (reflect `ln p` at the boundary).
"""
function MomentumBounds(;
        pmin::Real = 0.0, pmax::Real = Inf, action::Symbol = :terminate,
        save_positions = (false, true)
    )
    pmin >= zero(pmin) || throw(ArgumentError("pmin must be non-negative."))
    pmax > pmin || throw(ArgumentError("pmax must exceed pmin."))
    action in (:terminate, :reflect) ||
        throw(ArgumentError("`action` must be `:terminate` or `:reflect`; got :$action."))

    ymin = pmin == zero(pmin) ? -Inf : log(float(pmin))
    ymax = isinf(pmax) ? Inf : log(float(pmax))

    condition = (u, t, integrator) -> @inbounds(u[4] < ymin || u[4] > ymax)
    affect! = if action === :terminate
        integrator -> terminate!(integrator)
    else
        function (integrator)
            u = integrator.u
            y = @inbounds u[4]
            if y < ymin
                y = 2 * ymin - y
            elseif y > ymax
                y = 2 * ymax - y
            end
            integrator.u = _set_component(u, 4, clamp(y, ymin, ymax))
            return nothing
        end
    end
    return DiscreteCallback(condition, affect!; save_positions)
end

"""
    PitchAngleBounds(; save_positions = (false, false))

Return a `DiscreteCallback` reflecting pitch-angle cosine `u[5] = μ` back into `[-1, 1]`
when a finite SDE step overshoots `|μ| > 1` in focused transport simulations.
"""
function PitchAngleBounds(; save_positions = (false, false))
    condition = (u, t, integrator) -> @inbounds(u[5] < -one(u[5]) || u[5] > one(u[5]))
    affect! = function (integrator)
        u = integrator.u
        mu = @inbounds u[5]
        T = typeof(mu)
        # Fold into [0, 4) relative to -1, then reflect [2, 4) back into [-1, 1]
        r = mod(mu + one(T), T(4))
        mu_ref = clamp(r <= T(2) ? -one(T) + r : T(3) - r, -one(T), one(T))
        integrator.u = _set_component(u, 5, mu_ref)
        return nothing
    end
    return DiscreteCallback(condition, affect!; save_positions)
end
