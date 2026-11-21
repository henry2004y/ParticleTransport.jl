# ParticleTransport.jl

## Code and commits
- Follow Runic formatting conventions (92-char line limit).
- Avoid `Any` in function signatures, and use an explicit `;` before keyword arguments in calls.
- Keep SDE drift/diffusion kernels allocation-free and `Adapt.jl`-compatible.
- Add comments or docstrings only when behavior is not obvious from names and types.
- Commit titles follow `component: Brief summary` with a brief prose body.

## Dependencies
- Manage dependencies with `Pkg.add` and `Pkg.rm`; edit `Project.toml` directly only for `[compat]`.

## Testing and doctests
- Single test file (fast loop): `julia --project=test -e 'include("test/test_parker_1d.jl")'`
- Full test suite: `julia --project=test test/runtests.jl`
- Doctests: `julia --project=docs docs/make.jl doctest`
- Write new tests as standalone `module test_<name>` files included from [`test/runtests.jl`](./test/runtests.jl), placing imports and `struct` definitions at module top level and isolating cases in `let` blocks.
- Update [`test/test_api_surface.jl`](./test/test_api_surface.jl) whenever `export` or `public` names change.
