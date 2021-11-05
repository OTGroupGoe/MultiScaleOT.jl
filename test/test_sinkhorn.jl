import MultiScaleOT as MOT
using SparseArrays
import Random

@testset ExtendedTestSet "sinkhorn!" begin
    # We run the `sinkhorn!` routine for enough iterations and 
    # check that the primal-dual gap is approximately zero.

    # We start with some setup
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
    MOT.sinkhorn!(u, v, μ, ν, K, KT, 10000)

    # Get primal and duals
    P = u .* K .* v'
    a = ε.*log.(u)
    b = ε.*log.(v)

    # Test PD gap is small
    score1 = MOT.primal_score_dense(P, c, X, Y, μ, ν, ε)
    score2 = MOT.dual_score_dense(a, b, c, X, Y, μ, ν, ε)
    @test MOT.PD_gap_dense(a, b, P, c, X, Y, μ, ν, ε) < 1e-6

    # Test also PD gap for sparse versions 
    P2 = sparse(P)
    score1 = MOT.primal_score_sparse(P2, c, X, Y, μ, ν, ε)
    score2 = MOT.dual_score_sparse(a, b, c, X, Y, μ, ν, ε, P2)
    @test MOT.PD_gap_sparse(a, b, P2, c, X, Y, μ, ν, ε) < 1e-6
end

@testset ExtendedTestSet "get_stabilized_kernel, sinkhorn_stabilized!" begin

    # We start with the same setup# We start with some setup
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

    a = zeros(M)
    b = zeros(N)

    # Now we see that we can get the whole kernel function with
    # `get_stabilized_kernel` 
    K0 = exp.(-C./ε) .* (μ .* ν')
    rowval0 = repeat(1:M, N)
    colptr0 = 1:M:N*M+1
    θ = 0
    K = MOT.get_stabilized_kernel(c, a, b, ε, X, Y, μ, ν, θ, rowval0, colptr0)
    @test all(K .≈ K0)

    # But we actually would want a higher θ
    θ = 1e-20
    K = MOT.get_stabilized_kernel(c, a, b, ε, X, Y, μ, ν, θ, rowval0, colptr0)
    KT = K'

    status = MOT.sinkhorn_stabilized!(a, b, μ, ν, K, ε; max_error = 1e-6)
    @test status == 0
    @test MOT.PD_gap_sparse(a, b, K, c, X, Y, μ, ν, ε) < 1e-8

    # Try to get a sparser kernel
    K0 = deepcopy(K)
    droptol!(K0, 1e-20) # Remove entries below 1e-20
    
    # Take the support of `K0` as initial support
    rowval0 = K0.rowval
    colptr0 = K0.colptr;
    
    θ = 1e-15
    K = MOT.get_stabilized_kernel(c, a, b, ε, X, Y, μ, ν, θ, rowval0, colptr0)
    # We used the same epsilon so each entry of `K` must be either
    # * Smaller than θ (before multiplication by μ[i]ν[j]), or
    # * equal to that of K0
    mask = K0 .≥ θ.*(μ.*ν')
    @test all(K[mask] .≈ K0[mask])
    @test all(K[.!mask] .== 0)

    # With a stabilized kernel we can solve entropic OT with 
    # smaller epsilon.
    for _ in 1:5
        ε /= 2
        rowval0 = K.rowval
        colptr0 = K.colptr;
        K = MOT.get_stabilized_kernel(c, a, b, ε, X, Y, μ, ν, θ, rowval0, colptr0)
        status = MOT.sinkhorn_stabilized!(a, b, μ, ν, K, ε; max_error = 1e-6, max_iter = 100000)
        @test status != 2
        @test MOT.PD_gap_sparse(a, b, K, c, X, Y, μ, ν, ε) < 1e-6
    end
end
