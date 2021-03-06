
"""
    primal_score_dense(P, c, X, Y, μ, ν, ε)

Compute the primal score of `P` for the entropic problem 
with cost function `c` and regularization `ε`, using all
entries of `P`. Works for `P` both dense and sparse.
"""
function primal_score_dense(P, c, X, Y, μ, ν, ε)
    score = 0.0
    for j in eachindex(ν)
        for i in eachindex(μ)
            # Transport part
            score += @views P[i,j]*c(X[:,i], Y[:,j])
            # Entropic part
            score += ε*KL(P[i,j], μ[i]*ν[j])
        end
    end
    return score
end

"""
    dual_score_dense(a, b, c, X, Y, μ, ν, ε)

Compute the dual score of `(a,b)` for the entropic problem 
with cost function `c` and regularization `ε`, using all
points in the product space. 
"""
function dual_score_dense(a, b, c, X, Y, μ, ν, ε)
    # Transport part
    score = dot(a, μ) + dot(b, ν)
    # Entropic part
    for j in eachindex(ν)
        for i in eachindex(μ)
            cij = @views c(X[:,i], Y[:,j])
            score += ε*(1 - exp((a[i]+b[j]-cij)/ε))*μ[i]*ν[j]
        end
    end
    return score
end

"""
    primal_score_sparse(P::SparseMatrixCSC, c, X, Y, μ, ν, ε)

Compute the primal score of `P` for the entropic problem 
with cost function `c` and regularization `ε`, using only
the stored entries of `P`.
"""
function primal_score_sparse(P, c, X, Y, μ, ν, ε)
    if ε == 0
        return primal_score_sparse_eps0(P, c, X, Y, μ, ν, ε)
    end
    score = 0.0
    rows = rowvals(P) 
    vals = nonzeros(P)
    for j in 1:size(P, 2)
        for r in nzrange(P, j)
            i = rows[r]
            Pij = vals[r] # equivalent to getting P[i,j]
            # Transport part
            score += @views Pij*c(X[:,i], Y[:,j])
            # Entropic part
            score += ε*KL(Pij, μ[i]*ν[j])
        end
    end
    return score
end

"""
    primal_score_sparse_eps0(P, c, X, Y, μ, ν, ε)

Unregularized primal cost. To be called by `primal_score_sparse`
"""
function primal_score_sparse_eps0(P, c, X, Y, μ, ν, ε)
    score = 0.0
    for j in 1:size(P, 2)
        for r in P.colptr[j]:P.colptr[j+1]-1
            i = P.rowval[r]
            Pij = P.nzval[r]
            score += @views c(X[:,i], Y[:,j])*Pij
        end
    end
    return score
end


"""
    dual_score_sparse(a, b, c, X, Y, μ, ν, ε, P::SparseMatrixCSC)

Compute the dual score of `(a,b)` for the entropic problem 
with cost function `c` and regularization `ε`, using only the 
indices `(i,j)` where `P` is non-zero. 
"""
function dual_score_sparse(a, b, c, X, Y, μ, ν, ε, P::SparseMatrixCSC)
    if ε == 0
        return dual_score_sparse_eps0(a, b, c, X, Y, μ, ν, ε, P; threshold = 1e-8)
    end
    # Transport part
    score = dot(a, μ) + dot(b, ν)
    # Entropic part
    rows = rowvals(P) 
    vals = nonzeros(P)
    for j in 1:size(P, 2)
        for r in nzrange(P, j)
            i = rows[r]
            cij = @views c(X[:,i], Y[:,j])
            score += ε*(1 - exp((a[i] + b[j] - cij)/ε))*μ[i]*ν[j]
        end
    end
    return score
end

"""
    dual_score_sparse_eps0(P, c, X, Y, μ, ν, ε, P; threshold = 1e-8)

Unregularized dual cost. Yields infinite if some entry of the form 
`a[i]+b[j]-C[i,j]` in the support of `P` is bigger than `threshold`.
"""
function dual_score_sparse_eps0(a, b, c, X, Y, μ, ν, ε, P; threshold = 1e-8)
    # Transport part
    score = dot(a, μ) + dot(b, ν)
    # Entropic part
    rows = rowvals(P) 
    vals = nonzeros(P)
    for j in 1:size(P, 2)
        for r in nzrange(P, j)
            i = rows[r]
            if @views (a[i] + b[j] - c(X[:,i], Y[:,j])) > threshold
                return Inf
            end
        end
    end
    return score
end


"""
    PD_gap_dense(a, b, P, c, X, Y, μ, ν, ε)

Compute `primal_score - dual_score` using the dense versions of these functions.
"""
function PD_gap_dense(a, b, P, c, X, Y, μ, ν, ε)
    primal_score_dense(P, c, X, Y, μ, ν, ε) - dual_score_dense(a, b, c, X, Y, μ, ν, ε)
end

"""
    PD_gap_dense(a, b, P, c, X, Y, μ, ν, ε)

Compute `primal_score - dual_score` using the dense versions of these functions.
"""
function PD_gap_sparse(a, b, P, c, X, Y, μ, ν, ε)
    primal_score_sparse(P, c, X, Y, μ, ν, ε) - dual_score_sparse(a, b, c, X, Y, μ, ν, ε, P)
end

###################################################
# Versions for measures
###################################################

primal_score_dense(P, c, mu, nu, ε) = primal_score_dense(P, c, mu.points, nu.points, mu.weights, nu.weights, ε)

dual_score_dense(a, b, c, mu, nu, ε) = dual_score_dense(a, b, c, mu.points, nu.points, mu.weights, nu.weights, ε)

PD_gap_dense(a, b, P, c, mu, nu, ε) = PD_gap_dense(a, b, P, c, mu.points, nu.points, mu.weights, nu.weights, ε)

primal_score_sparse(P, c, mu, nu, ε) = primal_score_sparse(P, c, mu.points, nu.points, mu.weights, nu.weights, ε)

dual_score_sparse(a, b, c, mu, nu, ε, P) = dual_score_sparse(a, b, c, mu.points, nu.points, mu.weights, nu.weights, ε, P)

PD_gap_sparse(a, b, P, c, mu, nu, ε) = PD_gap_sparse(a, b, P, c, mu.points, nu.points, mu.weights, nu.weights, ε)
