using Documenter
using DocumenterVitepress
using Literate
using SHA: sha256, bytes2hex
using ParticleTransport

DocMeta.setdocmeta!(
    ParticleTransport,
    :DocTestSetup,
    :(using ParticleTransport, StochasticDiffEq, StaticArrays);
    recursive = true,
)

const DOCTEST_ONLY = "doctest" in ARGS
const FORCE_REBUILD = get(ENV, "DOCS_FORCE_REBUILD", "false") == "true"

const EXAMPLES_DIR = joinpath(@__DIR__, "examples")
const GENERATED_DIR = joinpath(@__DIR__, "src", "generated")

const ANALYTICAL_DEMOS = [
    "demo_convection_diffusion.jl",
    "demo_dsa_1d.jl",
    "demo_momentum_diffusion.jl",
    "demo_anisotropic_diffusion.jl",
]

const COMPARISON_DEMOS = [
    "demo_compare_magnetic_drift.jl",
    "demo_compare_parallel_shock.jl",
]

function process_examples(
        subdir::String, order::Vector{String};
        force_rebuild::Bool = FORCE_REBUILD
    )
    out_dir = joinpath(GENERATED_DIR, subdir)
    mkpath(out_dir)
    pages = String[]
    for filename in order
        src_path = joinpath(EXAMPLES_DIR, subdir, filename)
        name = replace(filename, ".jl" => ".md")
        md_path = joinpath(out_dir, name)
        hash_path = joinpath(out_dir, filename * ".sha256")
        current_hash = bytes2hex(sha256(read(src_path)))

        is_cached = !force_rebuild &&
            isfile(md_path) &&
            isfile(hash_path) &&
            read(hash_path, String) == current_hash

        if !is_cached
            @info "Generating $subdir/$filename"
            Literate.markdown(
                src_path, out_dir;
                flavor = Literate.DocumenterFlavor(),
                execute = true,
                credit = false,
            )
            write(hash_path, current_hash)
        end
        push!(pages, joinpath("generated", subdir, name))
    end
    return pages
end

base_pages = [
    "Home" => "index.md",
    "Physics & SDE Formulation" => "physics.md",
    "Numerical Methods" => "numerics.md",
    "API Reference" => "api.md",
]

pages = if DOCTEST_ONLY
    base_pages
else
    analytical_pages = process_examples("analytical", ANALYTICAL_DEMOS)
    comparison_pages = process_examples("comparison", COMPARISON_DEMOS)
    [
        "Home" => "index.md",
        "Guide" => [
            "Physics & SDE Formulation" => "physics.md",
            "Numerical Methods" => "numerics.md",
        ],
        "Analytical Benchmarks" => analytical_pages,
        "Comparison with TestParticle.jl" => comparison_pages,
        "API Reference" => "api.md",
    ]
end

makedocs(;
    modules = [ParticleTransport],
    authors = "Hongyang Zhou <hyzhou@umich.edu> and contributors",
    sitename = "ParticleTransport.jl",
    doctest = DOCTEST_ONLY ? :only : true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "https://github.com/henry2004y/ParticleTransport.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages,
    warnonly = [:missing_docs, :linkcheck],
)

if !DOCTEST_ONLY
    DocumenterVitepress.deploydocs(;
        repo = "github.com/henry2004y/ParticleTransport.jl",
        target = "build",
        devbranch = "main",
        branch = "gh-pages",
        push_preview = true,
    )
end
