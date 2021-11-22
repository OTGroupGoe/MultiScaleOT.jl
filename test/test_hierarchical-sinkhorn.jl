@testset ExtendedTestSet "hierarchical_sinkhorn" begin
    
    # Compare results of `sinkhorn!` with hierarchical sinkhorn
    Random.seed!(0)
    M = 100
    N = 101

    c(x,y) = MOT.l22(x,y)

    x = collect(range(-1, 1, length = M))
    y = collect(range(-1, 1, length = N))
    X = MOT.flat_grid(x)
    Y = MOT.flat_grid(y)

    μ = rand(M) .+ 1e-2
    ν = rand(N) .+ 1e-2

    MOT.normalize!(μ)
    MOT.normalize!(ν)

    C = [c(xi,yj) for xi in x, yj in y]
    ε = 1e-2

    u = ones(M)
    v = ones(N)

    K = exp.(-C./ε) .* (μ .* ν')
    KT = K'

    # Run the sinkhorn algorithm for enough iterations
    MOT.sinkhorn!(u, v, μ, ν, K, KT, 100000)

    # Get primal and duals
    P = u .* K .* v'
    a = ε.*log.(u)
    b = ε.*log.(v)

    # Test PD gap is small
    score_1_primal = MOT.primal_score_dense(P, c, X, Y, μ, ν, ε)
    score_1_dual = MOT.dual_score_dense(a, b, c, X, Y, μ, ν, ε)

    ###########################################

    # Now do the same for hierarchical sinkhorn
    shapeX = (M,)
    shapeY = (N,)
    mu = MOT.GridMeasure(X, μ, shapeX)
    nu = MOT.GridMeasure(Y, ν, shapeY)
    depth = MOT.compute_multiscale_depth(mu)

    # Epsilon schedule
    Nsteps = 3
    factor = 2.
    eps_target = ε

    layer_schedule, eps_schedule = MOT.default_eps_schedule(depth, eps_target; Nsteps, factor)

    θ0 = 1e-15
    θ_schedule = MOT.template_schedule(depth, [0, fill(θ0,Nsteps-1)...], 1.)

    params_schedule = MOT.make_multiscale_schedule(
                    layer = layer_schedule,
                    epsilon = eps_schedule, 
                    solver_truncation = θ_schedule,
                    solver_max_error = 1e-10,
                    solver_verbose = false,
                    solver_max_iter = 100000
            );
    layer0 = 3
    K, a, b, status = MOT.hierarchical_sinkhorn(mu, nu, c, params_schedule, layer0)
    score_2_primal = MOT.primal_score_dense(K, c, X, Y, μ, ν, ε)
    score_2_dual = MOT.dual_score_dense(a, b, c, X, Y, μ, ν, ε)

    # Compare them
    @test score_1_primal ≈ score_2_primal
    @test score_1_dual ≈ score_2_dual
end