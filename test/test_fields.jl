module test_fields

using Test
using ParticleTransport
using StaticArrays: SA, SVector

@testset "Analytical Fields & Derivatives" begin
    let
        u_lin(x) = SA[2.0 * x[1], -3.0 * x[2], 5.0 * x[3]]
        f = ParticleTransport.Field(u_lin)
        @test !ParticleTransport.is_time_dependent(f)
        x = SA[1.0, 2.0, 3.0]
        @test f(x, 0.0) == SA[2.0, -6.0, 15.0]
        @test ParticleTransport.divergence(f, x, 0.0, Val(1)) ≈ 2.0
        @test ParticleTransport.divergence(f, x, 0.0, Val(2)) ≈ -1.0
        @test ParticleTransport.divergence(f, x, 0.0, Val(3)) ≈ 4.0

        u_td(x, t) = SA[t * x[1], 0.0, 0.0]
        ft = ParticleTransport.Field(u_td)
        @test ParticleTransport.is_time_dependent(ft)
        @test ft(x, 2.5) == SA[2.5, 0.0, 0.0]
        @test ParticleTransport.divergence(ft, x, 2.5, Val(3)) ≈ 2.5
    end
end

@testset "1D / 2D / 3D Interpolators & LazyTimeInterpolator" begin
    let
        # 1D linear field
        xs = range(0.0, 10.0; length = 21)
        A1 = [SA[2.0 * x + 1.0, 0.0, 0.0] for x in xs]
        itp1 = build_interpolator(A1, xs; order = 1)
        @test itp1(SA[3.25, 0.0, 0.0], 0.0) ≈ SA[7.5, 0.0, 0.0]
        @test ParticleTransport.divergence(itp1, SA[3.25, 0.0, 0.0], 0.0, Val(1)) ≈ 2.0

        # 2D linear field passed as 3 × nx × ny array
        ys = range(-2.0, 2.0; length = 17)
        A2 = Array{Float64, 3}(undef, 3, length(xs), length(ys))
        for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
            A2[1, i, j] = 3.0 * x
            A2[2, i, j] = -1.5 * y
            A2[3, i, j] = 0.0
        end
        itp2 = build_interpolator(A2, (xs, ys); order = 1)
        @test itp2(SA[2.5, 0.5, 0.0], 0.0) ≈ SA[7.5, -0.75, 0.0]
        @test ParticleTransport.divergence(itp2, SA[2.5, 0.5, 0.0], 0.0, Val(2)) ≈ 1.5

        # 3D linear field
        zs = range(0.0, 4.0; length = 9)
        A3 = [SA[x, 2.0 * y, -0.5 * z] for x in xs, y in ys, z in zs]
        itp3 = build_interpolator(A3, (xs, ys, zs); order = 1)
        @test itp3(SA[1.5, 0.25, 2.0], 0.0) ≈ SA[1.5, 0.5, -1.0]
        @test ParticleTransport.divergence(itp3, SA[1.5, 0.25, 2.0], 0.0, Val(3)) ≈ 2.5

        # LazyTimeInterpolator
        A1_t1 = [SA[4.0 * x + 1.0, 0.0, 0.0] for x in xs]
        itp1_t1 = build_interpolator(A1_t1, xs; order = 1)
        lti = LazyTimeInterpolator([0.0, 2.0], [itp1, itp1_t1])
        @test ParticleTransport.is_time_dependent(lti)
        @test lti(SA[3.0, 0.0, 0.0], 1.0) ≈ SA[10.0, 0.0, 0.0]
        @test ParticleTransport.divergence(lti, SA[3.0, 0.0, 0.0], 1.0, Val(1)) ≈ 3.0
    end
end

@testset "curl_b_over_b2" begin
    let
        # B(x) = (0, 0, B0 * (1 + α x)) => B/B² = (0, 0, 1 / (B0(1+αx)))
        # ∇ × (B/B²) = (0, -∂/∂x (1/(B0(1+αx))), 0) = (0, α / (B0(1+αx)²), 0)
        B0 = 2.0
        α = 0.5
        Bf = ParticleTransport.Field(x -> SA[0.0, 0.0, B0 * (1.0 + α * x[1])])
        x = SA[1.0, 0.0, 0.0]
        expected = SA[0.0, α / (B0 * (1.0 + α * x[1])^2), 0.0]
        @test ParticleTransport.curl_b_over_b2(Bf, x, 0.0) ≈ expected rtol = 1.0e-12
    end
end

end
