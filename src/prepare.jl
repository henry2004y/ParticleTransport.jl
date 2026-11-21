# Parameter construction for Parker transport simulations.

"""
    prepare(;
        u = nothing, B = nothing,
        kappa = ZeroDiffusion(), dpp = ZeroMomentumDiffusion(),
        species = Proton, q = nothing, m = nothing, c = C_SI,
        dims::Int = 3, drift::Bool = false, type = nothing
    )

Build a [`TransportParams`](@ref) struct for Parker's transport equation in `dims ∈ (1, 2, 3)`
Cartesian dimensions.

- `u`, `B`: Analytical callables `f(x)` or `f(x, t)`, interpolators from
  [`build_interpolator`](@ref), [`LazyTimeInterpolator`](@ref), or `nothing` (zero field).
- `kappa`: Spatial diffusion model (`ZeroDiffusion`, `IsotropicDiffusion`,
  `ConstantDiffusion`, `PowerLawDiffusion`, `CustomDiffusion`, or a callable `(x, p, t) -> (kpara, kperp)`).
- `dpp`: Momentum diffusion model (`ZeroMomentumDiffusion`, `PowerLawMomentumDiffusion`,
  `CustomMomentumDiffusion`, or a callable `(x, p, t) -> Dpp`).
- `species`, `q`, `m`, `c`: Particle species (default `Proton` in SI units). Pass `q`, `m`, `c`
  explicitly (or `c = Inf` for non-relativistic kinematics) for dimensionless/normalized runs.
- `drift`: Include guiding-center magnetic drift `v_d = (p v / 3q) ∇ × (B / B²)`.
"""
function prepare(;
        u = nothing, B = nothing,
        kappa = ZeroDiffusion(), dpp = ZeroMomentumDiffusion(),
        species = Proton, q = nothing, m = nothing, c = C_SI,
        dims::Int = 3, drift::Bool = false, type = nothing
    )
    dims in (1, 2, 3) || throw(ArgumentError("`dims` must be 1, 2, or 3; got $dims."))
    any(f -> f isa AbstractArray, (u, B)) && throw(
        ArgumentError("Array fields require `build_interpolator(A, grids; ...)` before `prepare`.")
    )
    u_field = prepare_field(u)
    B_field = prepare_field(B)
    k_model = kappa isa AbstractSpatialDiffusion ? kappa : CustomDiffusion(kappa)
    d_model = dpp isa AbstractMomentumDiffusion ? dpp : CustomMomentumDiffusion(dpp)

    q_val = isnothing(q) ? species.q : q
    m_val = isnothing(m) ? species.m : m
    T = isnothing(type) ? promote_type(typeof(q_val), typeof(m_val), typeof(c), Float64) : type
    q_t, m_t, c_t = T(q_val), T(m_val), T(c)

    return TransportParams{
        dims, drift,
        typeof(u_field), typeof(B_field), typeof(k_model), typeof(d_model), T,
    }(u_field, B_field, k_model, d_model, q_t, m_t, c_t)
end
