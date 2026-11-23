# Parameter construction for Parker and Focused transport simulations.

"""
    prepare(;
        u = nothing, B = nothing,
        kappa = ZeroDiffusion(), dpp = ZeroMomentumDiffusion(),
        dmu = ZeroPitchAngleDiffusion(),
        model = ParkerTransport(), coords = CartesianCoordinates(),
        species = Proton, q = nothing, m = nothing, c = C_SI,
        dims::Int = 3, drift::Bool = false,
        track_3rd_dim::Bool = false, acc_region = nothing, type = nothing
    )

Build a [`TransportParams`](@ref) (for `ParkerTransport`) or
[`FocusedTransportParams`](@ref) (for `FocusedTransport`) struct in `dims ∈ (1, 2, 3)`
dimensions.

- `u`, `B`: Analytical callables `f(x)` or `f(x, t)`, interpolators from
  [`build_interpolator`](@ref), [`LazyTimeInterpolator`](@ref), or `nothing` (zero field).
- `kappa`: Spatial diffusion model (`ZeroDiffusion`, `IsotropicDiffusion`,
  `ConstantDiffusion`, `PowerLawDiffusion`, `QLTDiffusion`, `NLGCDiffusion`,
  `CustomDiffusion`, or a callable `(x, p, t) -> (kpara, kperp)`).
- `dpp`: Momentum diffusion model (`ZeroMomentumDiffusion`, `PowerLawMomentumDiffusion`,
  `AlfvenWaveMomentumDiffusion`, `FlowShearMomentumDiffusion`, `CustomMomentumDiffusion`,
  or a callable `(x, p, t) -> Dpp`).
- `dmu`: Pitch-angle diffusion model (used when `model isa FocusedTransport`).
- `model`: `ParkerTransport()` (default) or `FocusedTransport()`.
- `coords`: `CartesianCoordinates()` (default) or `SphericalCoordinates()`.
- `species`, `q`, `m`, `c`: Particle species (default `Proton` in SI units).
- `drift`: Include guiding-center magnetic drift.
- `track_3rd_dim`: When `dims == 2`, evolve the 3rd spatial coordinate (`z`) while keeping
  spatial field derivatives 2D (`∂/∂z = 0`).
- `acc_region`: Optional acceleration-region mask (`nothing`, a box tuple `((xmin, xmax), ...)`,
  `BoxAccelerationRegion`, or predicate `f(x, t) -> Bool`).
"""
function prepare(;
        u = nothing, B = nothing,
        kappa = ZeroDiffusion(), dpp = ZeroMomentumDiffusion(),
        dmu = ZeroPitchAngleDiffusion(),
        model::AbstractTransportModel = ParkerTransport(),
        coords::AbstractCoordinateSystem = CartesianCoordinates(),
        species::Particle = Proton, q = nothing, m = nothing, c::Real = C_SI,
        dims::Int = 3, drift::Bool = false,
        track_3rd_dim::Bool = false, acc_region = nothing, type = nothing
    )
    dims in (1, 2, 3) || throw(ArgumentError("`dims` must be 1, 2, or 3; got $dims."))
    any(f -> f isa AbstractArray, (u, B)) && throw(
        ArgumentError(
            "Array fields require `build_interpolator(A, grids; ...)` before `prepare`."
        )
    )
    u_field = prepare_field(u)
    B_field = prepare_field(B)
    k_model = kappa isa AbstractSpatialDiffusion ? kappa : CustomDiffusion(kappa)
    d_model = dpp isa AbstractMomentumDiffusion ? dpp : CustomMomentumDiffusion(dpp)
    acc_reg = prepare_acc_region(acc_region)
    t3d = dims == 2 ? track_3rd_dim : false

    q_val = isnothing(q) ? species.q : q
    m_val = isnothing(m) ? species.m : m
    T = isnothing(type) ?
        promote_type(typeof(q_val), typeof(m_val), typeof(c), Float64) : type
    q_t, m_t, c_t = T(q_val), T(m_val), T(c)
    COORD = typeof(coords)

    if model isa FocusedTransport
        mu_model = dmu isa AbstractPitchAngleDiffusion ?
            dmu : CustomPitchAngleDiffusion(dmu)
        return FocusedTransportParams{
            dims, COORD, drift, t3d,
            typeof(u_field), typeof(B_field), typeof(k_model), typeof(d_model),
            typeof(mu_model), typeof(acc_reg), T,
        }(u_field, B_field, k_model, d_model, mu_model, acc_reg, q_t, m_t, c_t)
    end

    return TransportParams{
        dims, COORD, drift, t3d,
        typeof(u_field), typeof(B_field), typeof(k_model), typeof(d_model),
        typeof(acc_reg), T,
    }(u_field, B_field, k_model, d_model, acc_reg, q_t, m_t, c_t)
end
