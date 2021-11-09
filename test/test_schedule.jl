import MultiScaleOT as MOT
import MultiScaleOT: scaling_schedule, template_schedule, constant_schedule, make_schedule
using Test

@testset ExtendedTestSet "schedule" begin
    Ns = [1, 2, 4]
    depth = length(Ns)
    template = [1,1]

    N_schedule = template_schedule(depth, template, Ns)
    @test N_schedule == repeat(Ns, inner = 2)

    # This should be the same as calling 
    Nsteps = 2
    N_schedule = template_schedule(depth, Nsteps, Ns)
    @test N_schedule == repeat(Ns, inner = 2)

    # If we want to perform further iterations on the final level we can
    # add some steps to the schedule
    N_schedule2 = template_schedule(depth, Nsteps, Ns; last_iter = [4])
    @test N_schedule2 == push!(copy(N_schedule), 4)

    ## Similar tests for `scaling_schedule`, which is appropriate for `ε`
    eps_target = 1.
    factor = 2.
    eps_schedule = scaling_schedule(depth, eps_target, Nsteps, factor);
    @test eps_schedule == [8., 4, 4, 2, 2, 1]

    # Can add some final steps
    eps_schedule2 = scaling_schedule(depth, eps_target, Nsteps, factor; last_iter = [0.5])
    @test eps_schedule2 == [8., 4, 4, 2, 2, 1, 0.5]

    ## We can group several schedules together, together with some parameters that may remain constant
    ## accross the algorithm, with `make_schedule`

    θ = 1e-20
    max_error = 1e-6

    sinkhorn_schedule = make_schedule(
        ε = eps_schedule, 
        N = N_schedule,
        θ = θ,
        max_error = max_error
    )

    @test sinkhorn_schedule.ε == eps_schedule
    @test sinkhorn_schedule.N == N_schedule
    @test sinkhorn_schedule.θ == fill(θ, length(eps_schedule))
    @test sinkhorn_schedule.max_error == fill(max_error, length(eps_schedule))

    # If non-trivial schedules have different length, an error is thrown
    @test_throws ErrorException sinkhorn_schedule = make_schedule(
        ε = eps_schedule2, 
        N = N_schedule,
        θ = θ,
        max_error = max_error
    )

    # If there is no non-trivial schedule another error is thrown
    @test_throws ErrorException sinkhorn_schedule = make_schedule(
        θ = θ,
        max_error = max_error
    )
end