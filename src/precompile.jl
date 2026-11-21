@setup_workload begin
    u_const(x) = SA[1.0, 0.0, 0.0]
    B_const(x) = SA[1.0, 0.0, 0.0]
    u0 = make_state(SA[0.0, 0.0, 0.0], 1.0)
    @compile_workload begin
        p = prepare(; u = u_const, B = B_const, kappa = ConstantDiffusion(0.1, 0.01))
        parker_drift(u0, p, 0.0)
        parker_diffusion(u0, p, 0.0)
        TransportProblem(u0, (0.0, 1.0), p)
    end
end
