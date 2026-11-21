module test_quality

using Test
using Aqua
using ExplicitImports
using ParticleTransport

@testset "Aqua" begin
    Aqua.test_all(ParticleTransport; ambiguities = false)
end

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(ParticleTransport) === nothing
    @test check_no_stale_explicit_imports(ParticleTransport) === nothing
end

end
