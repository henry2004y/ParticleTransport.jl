module test_physics_diffusion

using Test
using ParticleTransport
using StochasticDiffEq: EM
using StaticArrays: SA
using LinearAlgebra: norm
const ForwardDiff = ParticleTransport.ForwardDiff
const Adapt = ParticleTransport.Adapt

@testset "QLT Spatial Diffusion & Spatially Varying Turbulence" begin
    let
        # Giacalone & Jokipii (1999) asymptotic scaling at Ω L_c / v >> 1:
        # κ_∥ ∝ p^(4/3) B^(-1/3) when γ = 5/3 and non-relativistic (v = p / m)
        qlt = QLTDiffusion(; sigma2 = 0.1, Lc = 1.0e4, gamma = 5 / 3, eta = 0.04, q = 1.0, m = 1.0, c = Inf)
        B1 = SA[1.0, 0.0, 0.0]
        B2 = SA[8.0, 0.0, 0.0]
        x = SA[0.0, 0.0, 0.0]
        kp1, kr1 = ParticleTransport.diffusion_coefficients(qlt, B1, x, 1.0, 0.0)
        kp2, kr2 = ParticleTransport.diffusion_coefficients(qlt, B1, x, 8.0, 0.0)
        kpB, _ = ParticleTransport.diffusion_coefficients(qlt, B2, x, 1.0, 0.0)
        @test kr1 / kp1 ≈ 0.04 rtol = 1.0e-12
        # 8^(4/3) = 16
        @test kp2 / kp1 ≈ 16.0 rtol = 1.0e-3
        # 8^(-1/3) = 0.5
        @test kpB / kp1 ≈ 0.5 rtol = 1.0e-3

        # Finite-difference verification of div_diffusion_tensor with spatially varying B, sigma2, Lc
        B_field = ParticleTransport.Field(r -> SA[1.0 + 0.2 * r[1], 0.3 * r[2], 0.5 + 0.1 * r[3]])
        s2_fn(r, t) = 0.1 * (1.0 + 0.3 * r[1] + 0.1 * r[2])
        lc_fn(r, t) = 2.0 * (1.0 + 0.2 * r[3])
        qlt_var = QLTDiffusion(; sigma2 = s2_fn, Lc = lc_fn, gamma = 5 / 3, eta = 0.05, q = 1.0, m = 1.0, c = Inf)
        x0 = SA[0.4, -0.3, 0.7]
        p0 = 1.2
        div_ana = ParticleTransport.div_diffusion_tensor(qlt_var, B_field, x0, p0, 0.0, Val(3))
        # Numerical divergence of κ_ij via ForwardDiff on each column
        div_fd = [
            sum(
                ForwardDiff.jacobian(
                    r -> ParticleTransport.diffusion_tensor(qlt_var, B_field, r, p0, 0.0)[i, :], x0
                )[j, j] for j in 1:3
            )
            for i in 1:3
        ]
        @test div_ana ≈ div_fd rtol = 1.0e-10
        @test Adapt.adapt(identity, qlt) == qlt
    end
end

@testset "NLGC Perpendicular Diffusion & Gradients" begin
    let
        qlt = QLTDiffusion(; sigma2 = 0.1, Lc = 1.0e4, gamma = 5 / 3, q = 1.0, m = 1.0, c = Inf)
        nlgc = NLGCDiffusion(; kappa_para = qlt, sigma2_2d = 0.4, L_2d = 1.0e3, gamma = 5 / 3, m = 1.0, c = Inf)
        B_field = ParticleTransport.Field(r -> SA[1.0 + 0.2 * r[1], 0.15 * r[2], 0.8])
        x0 = SA[0.5, 0.2, -0.1]
        p0 = 2.0

        # Check NLGC velocity/B scaling: κ_⊥ ∝ v^(5-γ)/3 B^(γ-2)/3 = p^(10/9) B^(-1/9)
        _, kr1 = ParticleTransport.diffusion_coefficients(nlgc, SA[1.0, 0.0, 0.0], x0, 1.0, 0.0)
        _, kr2 = ParticleTransport.diffusion_coefficients(nlgc, SA[1.0, 0.0, 0.0], x0, 2.0, 0.0)
        @test kr2 / kr1 ≈ 2.0^(10 / 9) rtol = 2.0e-3

        # Check div_diffusion_tensor against ForwardDiff
        div_ana = ParticleTransport.div_diffusion_tensor(nlgc, B_field, x0, p0, 0.0, Val(3))
        div_fd = [
            sum(
                ForwardDiff.jacobian(
                    r -> ParticleTransport.diffusion_tensor(nlgc, B_field, r, p0, 0.0)[i, :], x0
                )[j, j] for j in 1:3
            )
            for i in 1:3
        ]
        @test div_ana ≈ div_fd rtol = 1.0e-10
        @test Adapt.adapt(identity, nlgc) == nlgc
    end
end

@testset "Alfvén-Wave & Flow-Shear Momentum Diffusion" begin
    # 1. Alfvén-wave D_pp = p^2 v_A^2 / (9 κ_∥)
    let
        vA = 0.3
        κ0 = 0.2
        # Constant κ_∥ => D_pp = D0 p^2 => drift_y = 3 D0, sigma_y = sqrt(2 D0)
        D0 = vA^2 / (9 * κ0)
        param = prepare(;
            B = x -> SA[1.0, 0.0, 0.0],
            kappa = ConstantDiffusion(κ0, 0.0),
            dpp = AlfvenWaveMomentumDiffusion(; vA),
        )
        drift_y, sigma_y = ParticleTransport.momentum_diffusion_terms(
            param.dpp, SA[0.0, 0.0, 0.0], 2.5, 0.0, param
        )
        @test drift_y ≈ 3 * D0 rtol = 1.0e-12
        @test sigma_y ≈ sqrt(2 * D0) rtol = 1.0e-12

        # PowerLaw κ_∥ ∝ p^(4/3) => D_pp ∝ p^(2/3) => (1/p^2) ∂(p D_pp)/∂p = (5/3) D_pp / p^2
        # Wait: in y = ln p, drift_y = (γ + 1) D_pp / p^2 with γ = 2/3 => (5/3) * vA^2 / (9 κ_∥)
        param_pl = prepare(;
            B = x -> SA[1.0, 0.0, 0.0],
            kappa = PowerLawDiffusion(; kappa0 = κ0, p0 = 1.0, alpha = 4 / 3),
            dpp = AlfvenWaveMomentumDiffusion(; vA),
        )
        p_test = 2.0
        kp_test = κ0 * p_test^(4 / 3)
        d_pl, s_pl = ParticleTransport.momentum_diffusion_terms(
            param_pl.dpp, SA[0.0, 0.0, 0.0], p_test, 0.0, param_pl
        )
        @test d_pl ≈ (5 / 3) * (vA^2 / (9 * kp_test)) rtol = 1.0e-12
        @test s_pl ≈ sqrt(2 * vA^2 / (9 * kp_test)) rtol = 1.0e-12
    end

    # 2. Flow-shear D_pp = Γ τ p^2
    let
        # Pure linear shear u_x(y) = S * y => div_u = 0, Γ = S^2 / 15
        S = 3.0
        τ0 = 0.2
        param_shear = prepare(;
            u = x -> SA[S * x[2], 0.0, 0.0],
            B = x -> SA[1.0, 0.0, 0.0],
            dpp = FlowShearMomentumDiffusion(; tau0 = τ0, p0 = 1.0, alpha = 0.0),
            dims = 2,
        )
        Γ_exact = S^2 / 15
        d_sh, s_sh = ParticleTransport.momentum_diffusion_terms(
            param_shear.dpp, SA[0.0, 0.0, 0.0], 2.0, 0.0, param_shear
        )
        @test d_sh ≈ 3 * Γ_exact * τ0 rtol = 1.0e-12
        @test s_sh ≈ sqrt(2 * Γ_exact * τ0) rtol = 1.0e-12

        # Pure isotropic expansion u = a * x => shear tensor σ_ij = 0 => Γ = 0
        param_exp = prepare(;
            u = x -> SA[2.0 * x[1], 2.0 * x[2], 2.0 * x[3]],
            B = x -> SA[1.0, 0.0, 0.0],
            dpp = FlowShearMomentumDiffusion(; tau0 = τ0),
            dims = 3,
        )
        d_exp, s_exp = ParticleTransport.momentum_diffusion_terms(
            param_exp.dpp, SA[1.0, 1.0, 1.0], 2.0, 0.0, param_exp
        )
        @test abs(d_exp) < 1.0e-14
        @test abs(s_exp) < 1.0e-14
    end
end

@testset "2D Out-of-Plane Tracking & Selective Acceleration Region" begin
    # 1. track_3rd_dim = true in 2D evolves z while keeping ∂/∂z = 0
    let
        u(x) = SA[1.0, 0.0, 2.5]
        B(x) = SA[0.0, 1.0, 0.0]
        p_no3d = prepare(; u, B, dims = 2, track_3rd_dim = false)
        p_3d = prepare(; u, B, dims = 2, track_3rd_dim = true)
        u0 = make_state(SA[0.0, 0.0, 0.0], 1.0)
        d_no3d = parker_drift(u0, p_no3d, 0.0)
        d_3d = parker_drift(u0, p_3d, 0.0)
        @test d_no3d[3] == 0.0
        @test d_3d[3] ≈ 2.5
        @test ParticleTransport.tracks_3rd_dim(p_3d)
        @test !ParticleTransport.tracks_3rd_dim(p_no3d)
    end

    # 2. Selective acceleration region masks momentum acceleration outside box
    let
        u(x) = SA[-3.0 * x[1], 0.0, 0.0] # compressing everywhere: div_u = -3 => dy = +1
        B(x) = SA[1.0, 0.0, 0.0]
        param = prepare(;
            u, B, dims = 1,
            dpp = PowerLawMomentumDiffusion(; D0 = 0.5),
            acc_region = ((-1.0, 1.0),),
        )
        u_in = make_state(SA[0.5, 0.0, 0.0], 1.0)
        u_out = make_state(SA[2.0, 0.0, 0.0], 1.0)
        @test parker_drift(u_in, param, 0.0)[4] > 0.0
        @test parker_diffusion(u_in, param, 0.0)[4, 4] > 0.0
        @test parker_drift(u_out, param, 0.0)[4] == 0.0
        @test parker_diffusion(u_out, param, 0.0)[4, 4] == 0.0
        # Spatial convection still active outside acceleration region
        @test parker_drift(u_out, param, 0.0)[1] ≈ -6.0
        @test Adapt.adapt(identity, param) == param
    end
end

end
