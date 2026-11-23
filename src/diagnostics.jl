# Diagnostics: weighted momentum spectra, spatial density, escaped-particle spectra,
# multi-dimensional local phase-space distributions, and O(1)-memory streaming reductions.

"""
    MomentumSpectrum{T}

Binned momentum distribution with bin `edges`, geometric bin `centers`, weighted counts
`counts` (``\\Delta N``), `dN_dlnp`, and omni-directional phase-space density
`f` (``f(p) = \\Delta N / (4\\pi p^2 \\Delta p)``).
"""
struct MomentumSpectrum{T <: AbstractFloat}
    edges::Vector{T}
    centers::Vector{T}
    counts::Vector{T}
    dN_dlnp::Vector{T}
    f::Vector{T}
end

function _bin_momenta(momenta::AbstractVector, weights::AbstractVector, edges::AbstractVector)
    T = float(promote_type(eltype(edges), eltype(weights)))
    ed = collect(T, edges)
    length(ed) >= 2 && all(d -> d > zero(d), diff(ed)) ||
        throw(ArgumentError("`edges` must have at least 2 strictly increasing values."))
    nb = length(ed) - 1
    counts = zeros(T, nb)
    for (p, w) in zip(momenta, weights)
        if p >= ed[1] && p < ed[end]
            idx = searchsortedlast(ed, p)
            if 1 <= idx <= nb
                counts[idx] += T(w)
            end
        end
    end
    return spectrum_from_counts(counts, ed)
end

"""
    momentum_spectrum(sol, edges; weights = nothing)

Compute a [`MomentumSpectrum`](@ref) from an `EnsembleSolution`, a [`SplittingResult`](@ref),
or a vector of 4- or 5-component states.
"""
function momentum_spectrum(
        states::AbstractVector, edges::AbstractVector;
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    momenta = [get_momentum(s) for s in states]
    w = isnothing(weights) ? ones(length(momenta)) : weights
    return _bin_momenta(momenta, w, edges)
end

function momentum_spectrum(res::SplittingResult, edges::AbstractVector)
    return momentum_spectrum(res.states, edges; weights = res.weights)
end

function momentum_spectrum(
        sim::EnsembleSolution, edges::AbstractVector;
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    states = [s.u[end] for s in sim.u]
    return momentum_spectrum(states, edges; weights)
end

"""
    spatial_density(sol, x_edges; dim = 1, weights = nothing)

Compute a 1D weighted spatial number-density histogram `dN / dx` along coordinate `dim ∈ 1:3`.
Returns `(; edges, centers, counts, density)`.
"""
function spatial_density(
        states::AbstractVector, x_edges::AbstractVector;
        dim::Int = 1, weights::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    dim in (1, 2, 3) || throw(ArgumentError("`dim` must be 1, 2, or 3; got $dim."))
    T = float(eltype(x_edges))
    ed = collect(T, x_edges)
    nb = length(ed) - 1
    counts = zeros(T, nb)
    w_vec = isnothing(weights) ? ones(T, length(states)) : weights
    for (s, w) in zip(states, w_vec)
        x = @inbounds s[dim]
        if x >= ed[1] && x < ed[end]
            idx = searchsortedlast(ed, x)
            if 1 <= idx <= nb
                counts[idx] += T(w)
            end
        end
    end
    centers = [(ed[i] + ed[i + 1]) / 2 for i in 1:nb]
    dx = diff(ed)
    return (; edges = ed, centers, counts, density = counts ./ dx)
end

function spatial_density(res::SplittingResult, x_edges::AbstractVector; dim::Int = 1)
    return spatial_density(res.states, x_edges; dim, weights = res.weights)
end

function spatial_density(
        sim::EnsembleSolution, x_edges::AbstractVector;
        dim::Int = 1, weights::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    states = [s.u[end] for s in sim.u]
    return spatial_density(states, x_edges; dim, weights)
end

"""
    streaming_spectrum_ensemble(prob, edges; prob_func = (prob, ctx) -> prob, weight_func = (sol, ctx) -> 1.0)

Construct a `SciMLBase.EnsembleProblem` with a streaming `output_func` and `reduction`
that accumulates weighted momentum bin counts in O(1) memory per trajectory.
Convert the resulting `sim.u` vector to a [`MomentumSpectrum`](@ref) with
`spectrum_from_counts(sim.u, edges)`.
"""
function streaming_spectrum_ensemble(
        prob::SDEProblem, edges::AbstractVector;
        prob_func = (prob, ctx) -> prob,
        weight_func = (sol, ctx) -> 1.0
    )
    T = float(eltype(edges))
    ed = collect(T, edges)
    nb = length(ed) - 1

    output_func = function (sol, ctx)
        p = get_momentum(sol.u[end])
        w = T(weight_func(sol, ctx))
        idx = (p >= ed[1] && p < ed[end]) ? searchsortedlast(ed, p) : 0
        return ((idx, w), false)
    end

    reduction = function (u, data, I)
        acc = isempty(u) ? zeros(T, nb) : u
        for (idx, w) in data
            if 1 <= idx <= nb
                @inbounds acc[idx] += w
            end
        end
        return (acc, false)
    end

    return EnsembleProblem(prob; prob_func, output_func, reduction, u_init = zeros(T, nb))
end

"""
    spectrum_from_counts(counts, edges)

Build a [`MomentumSpectrum`](@ref) from accumulated bin counts (e.g., from
[`streaming_spectrum_ensemble`](@ref)).
"""
function spectrum_from_counts(counts::AbstractVector, edges::AbstractVector)
    T = float(promote_type(eltype(counts), eltype(edges)))
    ed = collect(T, edges)
    c_vec = collect(T, counts)
    nb = length(ed) - 1
    length(c_vec) == nb || throw(
        ArgumentError("`counts` length ($(length(c_vec))) must equal length(edges) - 1 ($nb).")
    )
    centers = [sqrt(ed[i] * ed[i + 1]) for i in 1:nb]
    dlnp = [log(ed[i + 1] / ed[i]) for i in 1:nb]
    dp = [ed[i + 1] - ed[i] for i in 1:nb]
    dN_dlnp = c_vec ./ dlnp
    f = [c_vec[i] / (4 * T(pi) * centers[i]^2 * dp[i]) for i in 1:nb]
    return MomentumSpectrum(ed, centers, c_vec, dN_dlnp, f)
end

# --- Escaped-particle diagnostics ---

const _FACE_NAMES = ((:xmin, :xmax), (:ymin, :ymax), (:zmin, :zmax))

"""
    escape_face(u, bounds) -> Symbol

Classify which boundary face of `bounds` (a 1-, 2-, or 3-tuple of `(lo, hi)` pairs) state
`u` exited through. Returns `:xmin`, `:xmax`, `:ymin`, `:ymax`, `:zmin`, `:zmax`, or
`:interior` if `u` lies inside all bounds.
"""
function escape_face(u, bounds::Tuple)
    n = length(bounds)
    @inbounds for i in 1:n
        lo, hi = bounds[i]
        x = u[i]
        if x < lo
            return _FACE_NAMES[i][1]
        elseif x > hi
            return _FACE_NAMES[i][2]
        end
    end
    return :interior
end

function _face_index(face::Symbol, n::Int)
    @inbounds for i in 1:n
        face === _FACE_NAMES[i][1] && return 2 * i - 1
        face === _FACE_NAMES[i][2] && return 2 * i
    end
    return 0
end

"""
    escaped_spectra(sol, bounds, edges; weights = nothing) -> Dict{Symbol, MomentumSpectrum}

Compute per-face [`MomentumSpectrum`](@ref) histograms for all particles that exited
`bounds` (e.g., via [`TerminateOutside`](@ref)). Keys are the active face symbols
(`:xmin`, `:xmax`, ...).
"""
function escaped_spectra(
        states::AbstractVector, bounds::Tuple, edges::AbstractVector;
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    n = length(bounds)
    n in (1, 2, 3) || throw(ArgumentError("`bounds` must have 1, 2, or 3 (lo, hi) pairs."))
    T = float(promote_type(eltype(edges), isnothing(weights) ? Float64 : eltype(weights)))
    ed = collect(T, edges)
    length(ed) >= 2 && all(d -> d > zero(d), diff(ed)) ||
        throw(ArgumentError("`edges` must have at least 2 strictly increasing values."))
    nb = length(ed) - 1
    face_counts = Dict{Symbol, Vector{T}}()
    for i in 1:n
        face_counts[_FACE_NAMES[i][1]] = zeros(T, nb)
        face_counts[_FACE_NAMES[i][2]] = zeros(T, nb)
    end
    w_vec = isnothing(weights) ? ones(T, length(states)) : weights
    for (s, w) in zip(states, w_vec)
        face = escape_face(s, bounds)
        if face !== :interior
            p = get_momentum(s)
            if p >= ed[1] && p < ed[end]
                idx = searchsortedlast(ed, p)
                if 1 <= idx <= nb
                    @inbounds face_counts[face][idx] += T(w)
                end
            end
        end
    end
    return Dict{Symbol, MomentumSpectrum{T}}(
        k => spectrum_from_counts(v, ed) for (k, v) in face_counts
    )
end

function escaped_spectra(res::SplittingResult, bounds::Tuple, edges::AbstractVector)
    return escaped_spectra(res.states, bounds, edges; weights = res.weights)
end

function escaped_spectra(
        sim::EnsembleSolution, bounds::Tuple, edges::AbstractVector;
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    states = [s.u[end] for s in sim.u]
    return escaped_spectra(states, bounds, edges; weights)
end

"""
    streaming_escaped_ensemble(prob, bounds, edges; prob_func = (prob, ctx) -> prob, weight_func = (sol, ctx) -> 1.0)

Construct a `SciMLBase.EnsembleProblem` that accumulates escaped-particle momentum counts
per boundary face in O(1) memory per trajectory. The resulting `sim.u` is a `(nbins, 2*n)`
matrix whose columns correspond to `(:xmin, :xmax, ...)` in order.
"""
function streaming_escaped_ensemble(
        prob::SDEProblem, bounds::Tuple, edges::AbstractVector;
        prob_func = (prob, ctx) -> prob,
        weight_func = (sol, ctx) -> 1.0
    )
    n = length(bounds)
    n in (1, 2, 3) || throw(ArgumentError("`bounds` must have 1, 2, or 3 (lo, hi) pairs."))
    T = float(eltype(edges))
    ed = collect(T, edges)
    nb = length(ed) - 1
    nfaces = 2 * n

    output_func = function (sol, ctx)
        u_end = sol.u[end]
        face = escape_face(u_end, bounds)
        fidx = _face_index(face, n)
        if fidx == 0
            return ((0, 0, zero(T)), false)
        end
        p = get_momentum(u_end)
        pidx = (p >= ed[1] && p < ed[end]) ? searchsortedlast(ed, p) : 0
        w = T(weight_func(sol, ctx))
        return ((pidx, fidx, w), false)
    end

    reduction = function (u, data, I)
        acc = isempty(u) ? zeros(T, nb, nfaces) : u
        for (pidx, fidx, w) in data
            if 1 <= pidx <= nb && 1 <= fidx <= nfaces
                @inbounds acc[pidx, fidx] += w
            end
        end
        return (acc, false)
    end

    return EnsembleProblem(
        prob; prob_func, output_func, reduction, u_init = zeros(T, nb, nfaces)
    )
end

# --- Multi-dimensional local phase-space distributions ---

"""
    LocalDistribution{N, T}

Multi-dimensional weighted particle distribution over 1D/2D/3D spatial bins, momentum bins,
and optional pitch-angle `μ` bins (for focused transport).

`counts` has axes `(spatial..., p)` when `mu_edges === nothing`, or `(spatial..., p, mu)`
when `mu_edges` is provided.
"""
struct LocalDistribution{N, T <: AbstractFloat, SE, ME}
    spatial_edges::SE
    p_edges::Vector{T}
    mu_edges::ME
    counts::Array{T, N}
end

@inline function _find_bin(ed::Vector{T}, val) where {T}
    nb = length(ed) - 1
    if val >= ed[1] && val < ed[end]
        idx = searchsortedlast(ed, val)
        return (1 <= idx <= nb) ? idx : 0
    elseif val == ed[end]
        return nb
    end
    return 0
end

function _normalize_spatial_edges(spatial_edges, ::Type{T}) where {T}
    se_tup = spatial_edges isa Tuple ? spatial_edges : (spatial_edges,)
    n = length(se_tup)
    n in (1, 2, 3) ||
        throw(ArgumentError("`spatial_edges` must provide 1, 2, or 3 edge vectors."))
    eds = ntuple(i -> begin
        v = collect(T, se_tup[i])
        length(v) >= 2 && all(d -> d > zero(d), diff(v)) ||
            throw(ArgumentError("Spatial edges must have >= 2 strictly increasing values."))
        v
    end, Val(n))
    return eds
end

function _accumulate_state!(
        counts::Array{T}, s, w::T, se::NTuple{NS, Vector{T}},
        pe::Vector{T}, me::Union{Nothing, Vector{T}}
    ) where {NS, T}
    pidx = _find_bin(pe, get_momentum(s))
    pidx == 0 && return nothing
    ix = _find_bin(se[1], @inbounds s[1])
    ix == 0 && return nothing
    iy = NS >= 2 ? _find_bin(se[2], @inbounds s[2]) : 1
    iy == 0 && return nothing
    iz = NS == 3 ? _find_bin(se[3], @inbounds s[3]) : 1
    iz == 0 && return nothing

    if isnothing(me)
        @inbounds if NS == 1
            counts[ix, pidx] += w
        elseif NS == 2
            counts[ix, iy, pidx] += w
        else
            counts[ix, iy, iz, pidx] += w
        end
    else
        muidx = _find_bin(me, get_pitch_angle(s))
        muidx == 0 && return nothing
        @inbounds if NS == 1
            counts[ix, pidx, muidx] += w
        elseif NS == 2
            counts[ix, iy, pidx, muidx] += w
        else
            counts[ix, iy, iz, pidx, muidx] += w
        end
    end
    return nothing
end

"""
    local_distribution(sol, spatial_edges, p_edges; mu_edges = nothing, weights = nothing)

Bin states into a [`LocalDistribution`](@ref) over 1D/2D/3D `spatial_edges` (a single edge
vector or tuple of 1–3 edge vectors), momentum `p_edges`, and optional pitch-angle
`mu_edges`.
"""
function local_distribution(
        states::AbstractVector, spatial_edges, p_edges::AbstractVector;
        mu_edges::Union{Nothing, AbstractVector} = nothing,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    T = float(promote_type(eltype(p_edges), isnothing(weights) ? Float64 : eltype(weights)))
    se = _normalize_spatial_edges(spatial_edges, T)
    pe = collect(T, p_edges)
    length(pe) >= 2 && all(d -> d > zero(d), diff(pe)) ||
        throw(ArgumentError("`p_edges` must have >= 2 strictly increasing values."))
    me = if isnothing(mu_edges)
        nothing
    else
        v = collect(T, mu_edges)
        length(v) >= 2 && all(d -> d > zero(d), diff(v)) ||
            throw(ArgumentError("`mu_edges` must have >= 2 strictly increasing values."))
        v
    end

    s_dims = map(v -> length(v) - 1, se)
    dims = isnothing(me) ? (s_dims..., length(pe) - 1) : (s_dims..., length(pe) - 1, length(me) - 1)
    counts = zeros(T, dims...)
    w_vec = isnothing(weights) ? ones(T, length(states)) : weights
    for (s, w) in zip(states, w_vec)
        _accumulate_state!(counts, s, T(w), se, pe, me)
    end
    return LocalDistribution(se, pe, me, counts)
end

function local_distribution(
        res::SplittingResult, spatial_edges, p_edges::AbstractVector;
        mu_edges::Union{Nothing, AbstractVector} = nothing
    )
    return local_distribution(
        res.states, spatial_edges, p_edges; mu_edges, weights = res.weights
    )
end

function local_distribution(
        sim::EnsembleSolution, spatial_edges, p_edges::AbstractVector;
        mu_edges::Union{Nothing, AbstractVector} = nothing,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing
    )
    states = [s.u[end] for s in sim.u]
    return local_distribution(states, spatial_edges, p_edges; mu_edges, weights)
end

"""
    streaming_distribution_ensemble(
        prob, spatial_edges, p_edges;
        mu_edges = nothing, saveat = nothing,
        prob_func = (prob, ctx) -> prob,
        weight_func = (sol, ctx) -> 1.0
    )

Construct an O(1)-memory streaming `EnsembleProblem` accumulating weighted counts into
`(spatial..., p[, mu])` (when `saveat === nothing`) or `(spatial..., p[, mu], n_snapshots)`
when `saveat` is a vector of snapshot times.
"""
function streaming_distribution_ensemble(
        prob::SDEProblem, spatial_edges, p_edges::AbstractVector;
        mu_edges::Union{Nothing, AbstractVector} = nothing,
        saveat::Union{Nothing, AbstractVector} = nothing,
        prob_func = (prob, ctx) -> prob,
        weight_func = (sol, ctx) -> 1.0
    )
    T = float(eltype(p_edges))
    se = _normalize_spatial_edges(spatial_edges, T)
    pe = collect(T, p_edges)
    me = isnothing(mu_edges) ? nothing : collect(T, mu_edges)
    s_dims = map(v -> length(v) - 1, se)
    base_dims = isnothing(me) ?
        (s_dims..., length(pe) - 1) : (s_dims..., length(pe) - 1, length(me) - 1)

    if isnothing(saveat)
        output_func = (sol, ctx) -> ((sol.u[end], T(weight_func(sol, ctx))), false)
        reduction = function (u, data, I)
            acc = isempty(u) ? zeros(T, base_dims...) : u
            for (s, w) in data
                _accumulate_state!(acc, s, w, se, pe, me)
            end
            return (acc, false)
        end
        return EnsembleProblem(
            prob; prob_func, output_func, reduction, u_init = zeros(T, base_dims...)
        )
    end

    ts = collect(T, saveat)
    nt = length(ts)
    full_dims = (base_dims..., nt)

    output_func = function (sol, ctx)
        w = T(weight_func(sol, ctx))
        snaps = [sol(t_s) for t_s in ts]
        return ((snaps, w), false)
    end

    reduction = function (u, data, I)
        acc = isempty(u) ? zeros(T, full_dims...) : u
        for (snaps, w) in data
            for k in 1:nt
                s = snaps[k]
                pidx = _find_bin(pe, get_momentum(s))
                pidx == 0 && continue
                ix = _find_bin(se[1], @inbounds s[1])
                ix == 0 && continue
                NS = length(se)
                iy = NS >= 2 ? _find_bin(se[2], @inbounds s[2]) : 1
                iy == 0 && continue
                iz = NS == 3 ? _find_bin(se[3], @inbounds s[3]) : 1
                iz == 0 && continue
                if isnothing(me)
                    @inbounds if NS == 1
                        acc[ix, pidx, k] += w
                    elseif NS == 2
                        acc[ix, iy, pidx, k] += w
                    else
                        acc[ix, iy, iz, pidx, k] += w
                    end
                else
                    muidx = _find_bin(me, get_pitch_angle(s))
                    muidx == 0 && continue
                    @inbounds if NS == 1
                        acc[ix, pidx, muidx, k] += w
                    elseif NS == 2
                        acc[ix, iy, pidx, muidx, k] += w
                    else
                        acc[ix, iy, iz, pidx, muidx, k] += w
                    end
                end
            end
        end
        return (acc, false)
    end

    return EnsembleProblem(
        prob; prob_func, output_func, reduction, u_init = zeros(T, full_dims...)
    )
end
