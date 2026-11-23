# Initial and continuous particle injection samplers for EnsembleProblem workflows.

"""
    AbstractMomentumSampler

Abstract supertype for initial momentum distribution samplers.
"""
abstract type AbstractMomentumSampler end

"""
    AbstractSpatialSampler

Abstract supertype for initial spatial position samplers.
"""
abstract type AbstractSpatialSampler end

"""
    DeltaMomentum(p0) <: AbstractMomentumSampler

Mono-energetic (delta-function) momentum sampler returning `p0`.
"""
struct DeltaMomentum{T <: AbstractFloat} <: AbstractMomentumSampler
    p0::T
    function DeltaMomentum(p0::Real)
        p0 > zero(p0) || throw(ArgumentError("`p0` must be positive; got $p0."))
        pf = float(p0)
        return new{typeof(pf)}(pf)
    end
end

"""
    MaxwellianMomentum(; p_th, pmin = 0.0, pmax = Inf) <: AbstractMomentumSampler

3D Maxwellian momentum magnitude sampler with `f(p) ∝ p² exp(-p² / (2 p_th²))`
(i.e. `p = p_th * √(Z₁² + Z₂² + Z₃²)`), optionally truncated to `[pmin, pmax]`.
"""
struct MaxwellianMomentum{T <: AbstractFloat} <: AbstractMomentumSampler
    p_th::T
    pmin::T
    pmax::T
    function MaxwellianMomentum(; p_th::Real, pmin::Real = 0.0, pmax::Real = Inf)
        p_th > zero(p_th) || throw(ArgumentError("`p_th` must be positive; got $p_th."))
        pmin >= zero(pmin) || throw(ArgumentError("`pmin` must be non-negative; got $pmin."))
        pmax > pmin || throw(ArgumentError("`pmax` must exceed `pmin`."))
        vals = promote(float(p_th), float(pmin), float(pmax))
        return new{typeof(vals[1])}(vals...)
    end
end

"""
    PowerLawMomentum(; pmin, pmax, index) <: AbstractMomentumSampler

Power-law momentum sampler with `dN / dp ∝ p^(-index)` on `[pmin, pmax]`.
"""
struct PowerLawMomentum{T <: AbstractFloat} <: AbstractMomentumSampler
    pmin::T
    pmax::T
    index::T
    function PowerLawMomentum(; pmin::Real, pmax::Real, index::Real)
        pmin > zero(pmin) || throw(ArgumentError("`pmin` must be positive; got $pmin."))
        pmax > pmin || throw(ArgumentError("`pmax` must exceed `pmin`."))
        vals = promote(float(pmin), float(pmax), float(index))
        return new{typeof(vals[1])}(vals...)
    end
end

@inline sample_momentum(rng::AbstractRNG, s::DeltaMomentum) = s.p0

function sample_momentum(rng::AbstractRNG, s::MaxwellianMomentum{T}) where {T}
    while true
        z1 = randn(rng, T)
        z2 = randn(rng, T)
        z3 = randn(rng, T)
        p = s.p_th * sqrt(z1 * z1 + z2 * z2 + z3 * z3)
        if p >= s.pmin && p <= s.pmax && p > zero(T)
            return p
        end
    end
end

function sample_momentum(rng::AbstractRNG, s::PowerLawMomentum{T}) where {T}
    u = rand(rng, T)
    if abs(s.index - one(T)) < T(1.0e-12)
        return s.pmin * (s.pmax / s.pmin)^u
    end
    a = one(T) - s.index
    pmin_a = s.pmin^a
    pmax_a = s.pmax^a
    return (pmin_a + u * (pmax_a - pmin_a))^inv(a)
end

"""
    PointInjection(x0) <: AbstractSpatialSampler

Inject all particles at a fixed 3D position `x0`.
"""
struct PointInjection{T <: AbstractFloat} <: AbstractSpatialSampler
    x0::SVector{3, T}
    function PointInjection(x0::AbstractVector{<:Real})
        length(x0) == 3 || throw(ArgumentError("`x0` must have length 3; got $(length(x0))."))
        xv = SVector{3}(float(x0[1]), float(x0[2]), float(x0[3]))
        return new{eltype(xv)}(xv)
    end
    function PointInjection(x0::Real)
        xf = float(x0)
        return new{typeof(xf)}(SVector{3}(xf, zero(xf), zero(xf)))
    end
end

"""
    UniformBoxInjection(bounds) <: AbstractSpatialSampler

Inject particles uniformly inside a 1D, 2D, or 3D box specified as a tuple of `(lo, hi)`
pairs. Unspecified trailing coordinates are set to zero.
"""
struct UniformBoxInjection{N, T <: AbstractFloat} <: AbstractSpatialSampler
    bounds::NTuple{N, Tuple{T, T}}
    function UniformBoxInjection(bounds::Tuple)
        n = length(bounds)
        n in (1, 2, 3) ||
            throw(ArgumentError("`bounds` must have 1, 2, or 3 (lo, hi) pairs."))
        T = promote_type(Float64, map(b -> promote_type(typeof(b[1]), typeof(b[2])), bounds)...)
        b_conv = ntuple(val_i -> begin
            lo, hi = T(bounds[val_i][1]), T(bounds[val_i][2])
            hi >= lo || throw(ArgumentError("Upper bound must be >= lower bound."))
            (lo, hi)
        end, Val(n))
        return new{n, T}(b_conv)
    end
end

"""
    ConditionedInjection(bounds, predicate; max_tries = 10_000) <: AbstractSpatialSampler

Rejection-sample positions uniformly in `bounds` until `predicate(x, t)` returns `true`
(e.g., to inject where current density `|J|`, compression `-∇ ⋅ u`, or density `ρ` exceeds
a threshold).
"""
struct ConditionedInjection{N, F, T <: AbstractFloat} <: AbstractSpatialSampler
    box::UniformBoxInjection{N, T}
    predicate::F
    max_tries::Int
    function ConditionedInjection(bounds::Tuple, predicate; max_tries::Int = 10_000)
        max_tries >= 1 || throw(ArgumentError("`max_tries` must be >= 1."))
        box = UniformBoxInjection(bounds)
        return new{length(bounds), typeof(predicate), typeof(box.bounds[1][1])}(
            box, predicate, max_tries
        )
    end
end

@inline sample_position(rng::AbstractRNG, s::PointInjection, t) = s.x0

@inline function sample_position(rng::AbstractRNG, s::UniformBoxInjection{N, T}, t) where {N, T}
    @inbounds begin
        x1 = s.bounds[1][1] + rand(rng, T) * (s.bounds[1][2] - s.bounds[1][1])
        x2 = N >= 2 ? s.bounds[2][1] + rand(rng, T) * (s.bounds[2][2] - s.bounds[2][1]) : zero(T)
        x3 = N == 3 ? s.bounds[3][1] + rand(rng, T) * (s.bounds[3][2] - s.bounds[3][1]) : zero(T)
        return SVector{3, T}(x1, x2, x3)
    end
end

function sample_position(rng::AbstractRNG, s::ConditionedInjection, t)
    for _ in 1:s.max_tries
        x = sample_position(rng, s.box, t)
        s.predicate(x, t) && return x
    end
    throw(ErrorException(
        "ConditionedInjection failed to find a valid point after $(s.max_tries) attempts."
    ))
end

"""
    InjectionSource(; space, momentum, tspan = nothing, pitch_angle = nothing)

Bundle spatial, momentum, optional continuous-injection time window `tspan = (t_start, t_max)`,
and optional pitch-angle range `pitch_angle = (mu_min, mu_max)` (or `:isotropic`) for
5D focused transport states.
"""
struct InjectionSource{S <: AbstractSpatialSampler, M <: AbstractMomentumSampler, TS, PA}
    space::S
    momentum::M
    tspan::TS
    pitch_angle::PA
    function InjectionSource(;
            space::AbstractSpatialSampler,
            momentum::AbstractMomentumSampler,
            tspan::Union{Nothing, Tuple{Real, Real}} = nothing,
            pitch_angle::Union{Nothing, Symbol, Tuple{Real, Real}} = nothing
        )
        ts = if isnothing(tspan)
            nothing
        else
            t0, t1 = promote(float(tspan[1]), float(tspan[2]))
            t1 >= t0 || throw(ArgumentError("`tspan[2]` must be >= `tspan[1]`."))
            (t0, t1)
        end
        pa = if isnothing(pitch_angle)
            nothing
        elseif pitch_angle === :isotropic
            (-1.0, 1.0)
        elseif pitch_angle isa Tuple
            m0, m1 = promote(float(pitch_angle[1]), float(pitch_angle[2]))
            (-1.0 <= m0 <= m1 <= 1.0) ||
                throw(ArgumentError("`pitch_angle` bounds must satisfy -1 <= lo <= hi <= 1."))
            (m0, m1)
        else
            throw(ArgumentError("`pitch_angle` must be `nothing`, `:isotropic`, or `(lo, hi)`."))
        end
        return new{typeof(space), typeof(momentum), typeof(ts), typeof(pa)}(
            space, momentum, ts, pa
        )
    end
end

"""
    sample_state(rng::AbstractRNG, source::InjectionSource; t = 0.0)

Sample `(u0, t_inj)` from `source`. Returns a 4-component `SVector{4}` when
`source.pitch_angle === nothing`, or a 5-component `SVector{5}` when `pitch_angle` is set.
"""
function sample_state(rng::AbstractRNG, source::InjectionSource; t::Real = 0.0)
    t_inj = if isnothing(source.tspan)
        float(t)
    else
        t0, t1 = source.tspan
        t0 + rand(rng, typeof(t0)) * (t1 - t0)
    end
    x = sample_position(rng, source.space, t_inj)
    p = sample_momentum(rng, source.momentum)
    if isnothing(source.pitch_angle)
        return (make_state(x, p), t_inj)
    end
    m0, m1 = source.pitch_angle
    mu = m0 + rand(rng, typeof(m0)) * (m1 - m0)
    return (make_state(x, p, mu), t_inj)
end

"""
    injection_prob_func(source::InjectionSource; rng = nothing)

Return a SciML `prob_func(prob, ctx)` that samples each trajectory's initial state `u0`
(and injection start time `t_inj` when `source.tspan !== nothing`) from `source`.
"""
function injection_prob_func(source::InjectionSource; rng::Union{Nothing, AbstractRNG} = nothing)
    return function (prob, ctx)
        r = isnothing(rng) ? default_rng() : rng
        u0_s, t_inj = sample_state(r, source; t = prob.tspan[1])
        u0_new = prob.u0 isa SVector ? u0_s : collect(u0_s)
        t_end = prob.tspan[2]
        t0_new = isnothing(source.tspan) ? prob.tspan[1] : min(t_inj, t_end)
        return remake(prob; u0 = u0_new, tspan = (t0_new, t_end))
    end
end
