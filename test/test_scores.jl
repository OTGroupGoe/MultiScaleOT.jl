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

    C = [c(xi,yj) for xi in x, yj in y]
    for ε in [0.1, 0.01, 0.001]
        # The optimal entropic potentials are given by:
        a = -ε*log.(cosh.(x./(2ε)))
        b = [0, 0]
        # So the optimal plan should be:
        P = exp.((a .+ b' .- C)/ε).* μ .* ν'
        
        score1 = MOT.primal_score_dense(P, c, X, Y, μ, ν, ε)
        score2 = MOT.dual_score_dense(a, b, c, X, Y, μ, ν, ε)
        @test score1 ≈ score2
    
        # Test also sparse versions 
        P2 = sparse(P)
        score1 = MOT.primal_score_sparse(P2, c, X, Y, μ, ν, ε)
        score2 = MOT.dual_score_sparse(a, b, c, X, Y, μ, ν, ε, P2)
        @test score1 ≈ score2
    end
end
