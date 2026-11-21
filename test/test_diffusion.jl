module test_diffusion

using Test
using LinearAlgebra: norm
using ParticleTransport
using StaticArrays: SA, SMatrix

@testset "Diffusion Tensor & Symmetric Square Root" begin
    let
        x = SA[0.5, -0.3, 0.8]
        for Bvec in (
                SA[1.0, 0.0, 0.0],
                SA[0.0, 0.0, 2.5], # z-axis pole check
                SA[1.0, -2.0, 3.0],
                SA[0.0, 0.0, 0.0], # zero-B fallback
            )
            Bf = ParticleTransport.Field(_ -> Bvec)
            for model in (
                    IsotropicDiffusion(0.4),
                    ConstantDiffusion(1.2, 0.05),
                    ConstantDiffusion(1.2, 0.0), # purely field-aligned
                    PowerLawDiffusion(; kappa0 = 0.8, p0 = 2.0, alpha = 1 / 3, B0 = 1.0, beta = 0.5, eta = 0.1),
                )
                K = ParticleTransport.diffusion_tensor(model, Bf, x, 3.0, 0.0)
                σ = ParticleTransport.diffusion_sqrt(model, Bf, x, 3.0, 0.0)
                @test σ * σ' ≈ 2.0 * K atol = 1.0e-12
                @test σ ≈ σ' atol = 1.0e-12
            end
        end
    end
end

@testset "div_diffusion_tensor vs Finite Difference" begin
    let
        # Curved, non-uniform 3D magnetic field
        B_fn(x) = SA[1.0 + 0.2 * x[2], 0.3 * x[1] - 0.1 * x[3], 2.0 + 0.15 * x[1]^2]
        Bf = ParticleTransport.Field(B_fn)
        x = SA[0.4, -0.6, 0.9]
        p = 2.5
        h = 1.0e-6

        for model in (
                ConstantDiffusion(1.5, 0.2),
                PowerLawDiffusion(; kappa0 = 1.2, p0 = 1.0, alpha = 0.5, B0 = 2.0, beta = 1.0, eta = 0.15),
                CustomDiffusion((r, pm, t) -> (0.5 + 0.2 * r[1]^2, 0.1 + 0.05 * r[2])),
            )
            div_k = ParticleTransport.div_diffusion_tensor(model, Bf, x, p, 0.0, Val(3))
            # Numerical (∇⋅κ)_i = ∑_j ∂κ_ij/∂x_j
            fd = zeros(3)
            for j in 1:3
                ej = SA[j == 1 ? h : 0.0, j == 2 ? h : 0.0, j == 3 ? h : 0.0]
                Kp = ParticleTransport.diffusion_tensor(model, Bf, x + ej, p, 0.0)
                Km = ParticleTransport.diffusion_tensor(model, Bf, x - ej, p, 0.0)
                dK_dxj = (Kp - Km) / (2 * h)
                for i in 1:3
                    fd[i] += dK_dxj[i, j]
                end
            end
            @test div_k ≈ SA[fd[1], fd[2], fd[3]] rtol = 1.0e-6
        end
    end
end

@testset "Momentum Diffusion D_pp" begin
    let
        dpp_pl = PowerLawMomentumDiffusion(; D0 = 0.25, p0 = 2.0, gamma = 1.5)
        dpp_cu = CustomMomentumDiffusion((x, p, t) -> 0.25 * (p / 2.0)^1.5)
        x = SA[0.0, 0.0, 0.0]
        p = 3.5
        drift1, sig1 = ParticleTransport.momentum_diffusion_terms(dpp_pl, x, p, 0.0)
        drift2, sig2 = ParticleTransport.momentum_diffusion_terms(dpp_cu, x, p, 0.0)
        @test drift1 ≈ drift2 rtol = 1.0e-12
        @test sig1 ≈ sig2 rtol = 1.0e-12
    end
end

end
