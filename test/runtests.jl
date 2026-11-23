using Test

@testset "ParticleTransport.jl" begin
    @testset "Code Quality" begin
        include("test_quality.jl")
    end
    @testset "API Surface" begin
        include("test_api_surface.jl")
    end
    @testset "Fields & Interpolation" begin
        include("test_fields.jl")
    end
    @testset "Diffusion & Drifts" begin
        include("test_diffusion.jl")
    end
    @testset "Physics Diffusion, 2D 3rd-Dim & Acc Region" begin
        include("test_physics_diffusion.jl")
    end
    @testset "1D Parker Transport & DSA" begin
        include("test_parker_1d.jl")
    end
    @testset "3D Anisotropic Diffusion, D_pp & Drifts" begin
        include("test_parker_3d.jl")
    end
    @testset "Focused Transport Equation" begin
        include("test_focused.jl")
    end
    @testset "Spherical Coordinates" begin
        include("test_spherical.jl")
    end
    @testset "Injection Samplers" begin
        include("test_injection.jl")
    end
    @testset "Callbacks & Particle Splitting" begin
        include("test_callbacks_splitting.jl")
    end
    @testset "Diagnostics & Streaming Reduction" begin
        include("test_diagnostics.jl")
    end
end
