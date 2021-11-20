import MultiScaleOT as MOT
import LinearAlgebra: dot
using SparseArrays

@testset ExtendedTestSet "scores" begin
    # We will test the `primal-score` and `dual-score` 
    # functions agains an exact solution of the entropic OT problem.
    # For the following setup:
    N = 100
    c(x,y) = -dot(x,y)
    x = collect(range(-1, 1, length = N))
    y = [-0.5, 0.5]
    X = MOT.flat_grid(x)
    Y = MOT.flat_grid(y)
    μ = ones(N)./N
    ν = [0.5, 0.5]

    # Use two different types so that it is shown that 
    # it should work for all measures
    mu = MOT.GridMeasure(X, μ, (N,))
    nu = MOT.CloudMeasure(Y, ν)

    C = [c(xi,yj) for xi in x, yj in y]
    for ε in [0.1, 0.01, 0.001, 0.0]
        # The optimal entropic potentials are given by:
        b = [0, 0]  
        if ε == 0
            a = -0.5 .*abs.(x)
            P = [Float64(xi*yj > 0)/N for xi in x, yj in y]
            @test MOT.l1(sum(P, dims=2)[:], μ) < 1e-10
            @test MOT.l1(sum(P, dims=1)[:], ν) < 1e-10
        else
            a = -ε*log.(cosh.(x./(2ε)))
            P = exp.((a .+ b' .- C)/ε).* μ .* ν'
        end
        
        if ε > 0
            score1 = MOT.primal_score_dense(P, c, X, Y, μ, ν, ε)
            score2 = MOT.dual_score_dense(a, b, c, X, Y, μ, ν, ε)
            @test MOT.PD_gap_dense(a, b, P, c, X, Y, μ, ν, ε) < 1e-8
            
            # Test measure versions
            @test MOT.PD_gap_dense(a, b, P, c, mu, nu, ε) < 1e-8
        end

    
        # Test also sparse versions 
        P2 = sparse(P)
        score1 = MOT.primal_score_sparse(P2, c, X, Y, μ, ν, ε)
        score2 = MOT.dual_score_sparse(a, b, c, X, Y, μ, ν, ε, P2)
        @test score1 ≈ score2
        @test MOT.PD_gap_sparse(a, b, P2, c, X, Y, μ, ν, ε) < 1e-8
        
        # Test measure versions
        @test MOT.PD_gap_sparse(a, b, P2, c, mu, nu, ε) < 1e-8
    end
end
