# Diagnostics: weighted momentum spectra, spatial density, and O(1)-memory streaming reductions.

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
    centers = [sqrt(ed[i] * ed[i + 1]) for i in 1:nb]
    dlnp = [log(ed[i + 1] / ed[i]) for i in 1:nb]
    dp = [ed[i + 1] - ed[i] for i in 1:nb]
    dN_dlnp = counts ./ dlnp
    f = [counts[i] / (4 * T(pi) * centers[i]^2 * dp[i]) for i in 1:nb]
    return MomentumSpectrum(ed, centers, counts, dN_dlnp, f)
end

"""
    momentum_spectrum(sol, edges; weights = nothing)

Compute a [`MomentumSpectrum`](@ref) from an `EnsembleSolution`, a [`SplittingResult`](@ref),
or a vector of 4-component states.
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
