using BenchmarkTools
using ParticleTransport
using Random: MersenneTwister
using StaticArrays: SA, SMatrix, SVector
using StochasticDiffEq: EM

const SUITE = BenchmarkGroup()

# Batch evaluators keep micro-benchmarks well above timer resolution without
# introducing allocations or allowing the compiler to constant-fold inputs.
function _bench_drift_oop(states, param, t)
    acc = zero(SVector{4, Float64})
    @inbounds for u in states
        acc += parker_drift(u, param, t)
    end
    return acc
end

function _bench_diffusion_oop(states, param, t)
    acc = zero(SMatrix{4, 4, Float64})
    @inbounds for u in states
        acc += parker_diffusion(u, param, t)
    end
    return acc
end

function _bench_drift_iip(du, states, param, t)
    acc = 0.0
    @inbounds for u in states
        parker_drift!(du, u, param, t)
        acc += du[1] + du[4]
    end
    return acc
end

function _bench_diffusion_iip(du, states, param, t)
    acc = 0.0
    @inbounds for u in states
        parker_diffusion!(du, u, param, t)
        acc += du[1, 1] + du[4, 4]
    end
    return acc
end

function _bench_field_eval(field, xs, t)
    acc = zero(SVector{3, Float64})
    @inbounds for x in xs
        acc += field(x, t)
    end
    return acc
end

function _bench_field_div(field, xs, t, ::Val{D}) where {D}
    acc = 0.0
    @inbounds for x in xs
        acc += ParticleTransport.divergence(field, x, t, Val(D))
    end
    return acc
end

# ---------------------------------------------------------------------------
# 1. Hot-path SDE drift and diffusion kernels (1024 evaluations per sample)
# ---------------------------------------------------------------------------
let
    kernels = SUITE["kernels"] = BenchmarkGroup()
    rng = MersenneTwister(1234)
    n_eval = 1024

    states_1d = [make_state(0.05 * randn(rng), 0.5 + 2.0 * rand(rng)) for _ in 1:n_eval]
    states_3d = [
        make_state(
            SA[0.5 + 3.0 * rand(rng), -1.5 + 3.0 * rand(rng), 0.5 + 3.0 * rand(rng)],
            0.5 + 2.0 * rand(rng),
        )
        for _ in 1:n_eval
    ]
    states_3d_vec = [collect(u) for u in states_3d]

    u_shock(x) = SA[0.625 - 0.375 * tanh(x[1] / 0.02), 0.0, 0.0]
    param_1d = prepare(; u = u_shock, kappa = IsotropicDiffusion(0.2), dims = 1)

    kernels["parker_drift_1d_analytical"] =
        @benchmarkable _bench_drift_oop($states_1d, $param_1d, 0.0)
    kernels["parker_diffusion_1d"] =
        @benchmarkable _bench_diffusion_oop($states_1d, $param_1d, 0.0)

    b_hat = SA[1.0, 2.0, 2.0] / 3.0
    B_tilted(x) = 2.0 * b_hat
    u_3d(x) = SA[1.0 + 0.1 * x[1], -0.05 * x[2], 0.02 * x[3]]
    param_3d = prepare(;
        u = u_3d,
        B = B_tilted,
        kappa = ConstantDiffusion(1.0, 0.1),
        dpp = PowerLawMomentumDiffusion(; D0 = 0.2, p0 = 1.0, gamma = 2.0),
        dims = 3,
    )

    kernels["parker_drift_3d_anisotropic"] =
        @benchmarkable _bench_drift_oop($states_3d, $param_3d, 0.0)
    kernels["parker_diffusion_3d_anisotropic"] =
        @benchmarkable _bench_diffusion_oop($states_3d, $param_3d, 0.0)

    B_grad(x) = SA[0.0, 0.0, 1.0 + 0.2 * x[1]]
    param_drift = prepare(;
        u = u_3d,
        B = B_grad,
        kappa = ConstantDiffusion(1.0, 0.1),
        q = 1.0,
        m = 1.0,
        c = Inf,
        dims = 3,
        drift = true,
    )
    kernels["parker_drift_3d_magnetic_drift"] =
        @benchmarkable _bench_drift_oop($states_3d, $param_drift, 0.0)

    xs = range(0.0, 4.0; length = 17)
    ys = range(-2.0, 2.0; length = 17)
    zs = range(0.0, 4.0; length = 17)
    Au = [SA[1.0 + 0.1 * x, -0.05 * y, 0.02 * z] for x in xs, y in ys, z in zs]
    AB = [SA[0.2, 0.4, 1.0 + 0.1 * x] for x in xs, y in ys, z in zs]
    u_itp = build_interpolator(Au, (xs, ys, zs); order = 1)
    B_itp = build_interpolator(AB, (xs, ys, zs); order = 1)
    param_itp = prepare(;
        u = u_itp,
        B = B_itp,
        kappa = ConstantDiffusion(1.0, 0.1),
        dims = 3,
    )

    kernels["parker_drift_3d_interpolated"] =
        @benchmarkable _bench_drift_oop($states_3d, $param_itp, 0.0)

    du_vec = zeros(4)
    du_mat = zeros(4, 4)
    kernels["parker_drift_inplace"] =
        @benchmarkable _bench_drift_iip($du_vec, $states_3d_vec, $param_3d, 0.0)
    kernels["parker_diffusion_inplace"] =
        @benchmarkable _bench_diffusion_iip($du_mat, $states_3d_vec, $param_3d, 0.0)
end

# ---------------------------------------------------------------------------
# 2. Field evaluation and interpolation (1024 evaluations per sample)
# ---------------------------------------------------------------------------
let
    fields = SUITE["fields"] = BenchmarkGroup()
    rng = MersenneTwister(5678)
    n_eval = 1024

    xs = range(0.0, 10.0; length = 65)
    ys = range(-2.0, 2.0; length = 33)
    zs = range(0.0, 4.0; length = 33)

    A1 = [SA[2.0 * x + 1.0, 0.0, 0.0] for x in xs]
    A1_t1 = [SA[4.0 * x + 1.0, 0.0, 0.0] for x in xs]
    itp1 = build_interpolator(A1, xs; order = 1)
    itp1_t1 = build_interpolator(A1_t1, xs; order = 1)
    lti = LazyTimeInterpolator([0.0, 2.0], [itp1, itp1_t1])

    A3 = [SA[x, 2.0 * y, -0.5 * z] for x in xs, y in ys, z in zs]
    itp3 = build_interpolator(A3, (xs, ys, zs); order = 1)

    pts_1d = [SA[0.5 + 9.0 * rand(rng), 0.0, 0.0] for _ in 1:n_eval]
    pts_3d = [
        SA[0.5 + 9.0 * rand(rng), -1.8 + 3.6 * rand(rng), 0.2 + 3.6 * rand(rng)]
        for _ in 1:n_eval
    ]

    fields["interp_1d_eval"] = @benchmarkable _bench_field_eval($itp1, $pts_1d, 0.0)
    fields["interp_3d_eval"] = @benchmarkable _bench_field_eval($itp3, $pts_3d, 0.0)
    fields["interp_3d_div"] = @benchmarkable _bench_field_div($itp3, $pts_3d, 0.0, Val(3))
    fields["lazy_time_interp_eval"] = @benchmarkable _bench_field_eval($lti, $pts_1d, 1.0)
end

# ---------------------------------------------------------------------------
# 3. End-to-end SDE integration workloads
# ---------------------------------------------------------------------------
let
    solves = SUITE["solve"] = BenchmarkGroup()

    u_shock(x) = SA[0.625 - 0.375 * tanh(x[1] / 0.02), 0.0, 0.0]
    param_1d = prepare(; u = u_shock, kappa = IsotropicDiffusion(0.2), dims = 1)
    prob_1d = TransportProblem(make_state(0.0, 1.0), (0.0, 2.0), param_1d)

    solves["single_1d_dsa"] =
        @benchmarkable solve($prob_1d, EM(); dt = 1.0e-2, save_everystep = false)

    B_grad(x) = SA[0.0, 0.0, 1.0 + 0.2 * x[1]]
    param_3d = prepare(;
        B = B_grad,
        kappa = ConstantDiffusion(1.0, 0.1),
        q = 1.0,
        m = 1.0,
        c = Inf,
        dims = 3,
        drift = true,
    )
    prob_3d = TransportProblem(make_state(SA[0.0, 0.0, 0.0], 3.0), (0.0, 2.0), param_3d)

    solves["single_3d_drift"] =
        @benchmarkable solve($prob_3d, EM(); dt = 1.0e-2, save_everystep = false)

    ep_1d = EnsembleProblem(prob_1d)
    solves["ensemble_1d_serial"] = @benchmarkable solve(
        $ep_1d, EM(), EnsembleSerial();
        dt = 2.0e-2, trajectories = 50, save_everystep = false,
    )

    u_comp(x) = SA[-3.0 * x[1], 0.0, 0.0]
    param_split = prepare(; u = u_comp, kappa = ZeroDiffusion(), dims = 1)
    prob_split = TransportProblem(make_state(0.0, 1.0), (0.0, 2.5), param_split)
    rule = SplitRule(; ratio = 2.0, nsplit = 2, max_generations = 4)

    solves["splitting_1d"] = @benchmarkable solve_with_splitting(
        $prob_split, EM(), EnsembleSerial();
        rule = $rule, trajectories = 10, dt = 1.0e-2,
    )
end

# ---------------------------------------------------------------------------
# 4. Diagnostics
# ---------------------------------------------------------------------------
let
    diag = SUITE["diagnostics"] = BenchmarkGroup()

    rng = MersenneTwister(42)
    states = [make_state(SA[randn(rng), 0.0, 0.0], exp(abs(randn(rng)))) for _ in 1:10_000]
    p_edges = 10.0 .^ range(0.0, 2.0; length = 21)
    x_edges = collect(range(-4.0, 4.0; length = 41))

    diag["momentum_spectrum"] = @benchmarkable momentum_spectrum($states, $p_edges)
    diag["spatial_density"] = @benchmarkable spatial_density($states, $x_edges; dim = 1)
end
