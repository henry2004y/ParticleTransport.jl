using Documenter
using DocumenterVitepress
using ParticleTransport

DocMeta.setdocmeta!(
    ParticleTransport,
    :DocTestSetup,
    :(using ParticleTransport, StochasticDiffEq, StaticArrays);
    recursive = true,
)

const DOCTEST_ONLY = "doctest" in ARGS

makedocs(;
    modules = [ParticleTransport],
    authors = "Hongyang Zhou <hyzhou@umich.edu> and contributors",
    sitename = "ParticleTransport.jl",
    doctest = DOCTEST_ONLY ? :only : true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "https://github.com/JuliaSpacePhysics/ParticleTransport.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Physics & SDE Formulation" => "physics.md",
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs, :linkcheck],
)

if !DOCTEST_ONLY
    DocumenterVitepress.deploydocs(;
        repo = "github.com/JuliaSpacePhysics/ParticleTransport.jl",
        target = "build",
        devbranch = "main",
        branch = "gh-pages",
        push_preview = true,
    )
end
