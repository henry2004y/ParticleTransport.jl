# Generational EnsembleProblem runner for particle splitting.

"""
    SplitRule(; ratio = 2.0, nsplit = 2, max_generations = 8, max_particles = 1_000_000)

Configuration for momentum-based particle splitting. Whenever a particle's momentum increases
by a factor `ratio` above its current split threshold, its weight is divided by `nsplit`,
its threshold advances by `ratio`, and `nsplit - 1` child particles are queued for the next
ensemble generation.
"""
struct SplitRule{T <: AbstractFloat}
    log_ratio::T
    nsplit::Int
    max_generations::Int
    max_particles::Int
    function SplitRule(;
            ratio::Real = 2.0, nsplit::Int = 2,
            max_generations::Int = 8, max_particles::Int = 1_000_000
        )
        ratio > one(ratio) || throw(ArgumentError("`ratio` must be > 1; got $ratio."))
        nsplit >= 2 || throw(ArgumentError("`nsplit` must be >= 2; got $nsplit."))
        max_generations >= 0 || throw(ArgumentError("`max_generations` must be >= 0."))
        max_particles >= 1 || throw(ArgumentError("`max_particles` must be >= 1."))
        lr = log(float(ratio))
        return new{typeof(lr)}(lr, nsplit, max_generations, max_particles)
    end
end

struct _SpawnRecord{U, T <: AbstractFloat}
    u0::U
    t0::T
    weight::T
    next_y::T
end

"""
    SplittingResult{U, T}

Container returned by [`solve_with_splitting`](@ref) holding final states, final times,
conserved particle weights, and generation count.
"""
struct SplittingResult{U, T <: AbstractFloat}
    states::Vector{U}
    times::Vector{T}
    weights::Vector{T}
    generations::Int
end

"""
    solve_with_splitting(
        prob, alg, ensemblealg = EnsembleThreads();
        rule::SplitRule = SplitRule(),
        trajectories::Int = 1,
        prob_func = (prob, ctx) -> prob,
        weights = nothing,
        callback = nothing,
        kwargs...
    )

Run a generational `SciMLBase.EnsembleProblem` with momentum-based particle splitting.
Total particle weight is strictly conserved across all split events.
"""
function solve_with_splitting(
        prob::SDEProblem, alg, ensemblealg = EnsembleThreads();
        rule::SplitRule = SplitRule(),
        trajectories::Int = 1,
        prob_func = (prob, ctx) -> prob,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
        callback = nothing,
        kwargs...
    )
    T = float(eltype(prob.u0))
    t_end = T(prob.tspan[2])

    init_records = Vector{_SpawnRecord{typeof(prob.u0), T}}(undef, trajectories)
    for i in 1:trajectories
        ctx = EnsembleContext(i, 1, 1, nothing, default_rng(), nothing)
        pi = prob_func(prob, ctx)
        w0 = isnothing(weights) ? one(T) : T(weights[i])
        y0 = @inbounds T(pi.u0[4])
        init_records[i] = _SpawnRecord(pi.u0, T(pi.tspan[1]), w0, y0 + T(rule.log_ratio))
    end

    all_states = typeof(prob.u0)[]
    all_times = T[]
    all_weights = T[]

    current_gen = init_records
    gen_count = 0
    total_particles = length(current_gen)

    while !isempty(current_gen) && gen_count <= rule.max_generations
        gen_count += 1
        n_cur = length(current_gen)
        allow_split = gen_count <= rule.max_generations && total_particles < rule.max_particles

        child_buckets = [Vector{_SpawnRecord{typeof(prob.u0), T}}() for _ in 1:n_cur]
        final_weights = Vector{T}(undef, n_cur)

        gen_prob_func = function (p_base, ctx)
            i = ctx.sim_id
            rec = current_gen[i]
            final_weights[i] = rec.weight
            if !allow_split
                return remake(p_base; u0 = rec.u0, tspan = (rec.t0, t_end), callback)
            end
            next_y_ref = Ref(rec.next_y)
            w_ref = Ref(rec.weight)
            bucket = child_buckets[i]
            split_cb = DiscreteCallback(
                (u, t, integrator) -> @inbounds(u[4] >= next_y_ref[]),
                function (integrator)
                    y = @inbounds integrator.u[4]
                    while y >= next_y_ref[]
                        w_new = w_ref[] / rule.nsplit
                        w_ref[] = w_new
                        next_y_ref[] += T(rule.log_ratio)
                        final_weights[i] = w_new
                        u_copy = integrator.u isa SVector ? integrator.u : copy(integrator.u)
                        for _ in 1:(rule.nsplit - 1)
                            push!(bucket, _SpawnRecord(u_copy, T(integrator.t), w_new, next_y_ref[]))
                        end
                    end
                    return nothing
                end;
                save_positions = (false, false),
            )
            cb = isnothing(callback) ? split_cb : CallbackSet(callback, split_cb)
            return remake(p_base; u0 = rec.u0, tspan = (rec.t0, t_end), callback = cb)
        end

        ep = EnsembleProblem(prob; prob_func = gen_prob_func, safetycopy = false)
        sim = solve(ep, alg, ensemblealg; trajectories = n_cur, save_everystep = false, kwargs...)

        for i in 1:n_cur
            push!(all_states, sim.u[i].u[end])
            push!(all_times, T(sim.u[i].t[end]))
            push!(all_weights, final_weights[i])
        end

        next_gen = _SpawnRecord{typeof(prob.u0), T}[]
        if allow_split
            for b in child_buckets
                for rec in b
                    if total_particles < rule.max_particles
                        push!(next_gen, rec)
                        total_particles += 1
                    end
                end
            end
        end
        current_gen = next_gen
    end

    return SplittingResult(all_states, all_times, all_weights, gen_count)
end
