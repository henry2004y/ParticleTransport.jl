# ParticleTransport.jl

`ParticleTransport.jl` solves energetic particle transport and acceleration equations in
magnetized plasmas using forward-in-time stochastic differential equations (SDEs) built on
`SciMLBase` and `StochasticDiffEq.jl`.

Inspired by [`xiaocanli/stochastic-parker`](https://github.com/xiaocanli/stochastic-parker)
(GPAT) and structured alongside
[`TestParticle.jl`](https://github.com/JuliaSpacePhysics/TestParticle.jl), it provides an
alternative macroscopic transport solver to test-particle orbit tracing.

## Quick Start

```julia
using ParticleTransport, StochasticDiffEq, StaticArrays

u(x) = SA[1.5 - 0.5 * tanh(x[1] / 0.1), 0.0, 0.0]
B(x) = SA[1.0, 0.0, 0.0]

param = prepare(; u, B, kappa = ConstantDiffusion(0.1, 0.0), dims = 1)
u0 = make_state(0.0, 1.0)
prob = TransportProblem(u0, (0.0, 10.0), param)
sol = solve(prob, EM(); dt = 1.0e-3)
```
