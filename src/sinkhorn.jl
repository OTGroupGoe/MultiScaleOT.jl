# CODE STATUS: MOSTLY REVISED, tESTED
# A lot of cleaning up needed

import LinearAlgebra: mul!

# TODO: MEDIUM, PERFORMANCE
# Inbounds everywhere where it applies
# TODO: LOW, PERFORMANCE
# Benchmark `using MKLSparse` and see if it improves performance

# TODO: maybe this to aux?
function isthere_nan_or_inf(v)
    for vi in v
        if (isnan(vi) | isinf(vi))
            return true
        end
    end
    return false
end

"""
    sinkhorn!(u, v, μ, ν, K, KT, Niter)

Run `Niter` iterations of the Sinkhorn algorithm inplace on the scaling factors
`(u,v)`, with marginals `μ` and `ν` and Gibbs entropy given by 
`K` and its transpose `KT`.
At the end of the iterates the coupling `u.*K.*v'`` has exact first marginal, 
i.e., `sum(u .* K .* v', dims = 2)[:] == μ`.
"""
function sinkhorn!(u, v, μ, ν, K, KT, Niter)
    for _ in 1:Niter
        mul!(v, KT, u)
        v .= ν ./ v
        mul!(u, K, v)
        u .= μ ./ u
    end
end

# TODO, MIDDLE, PERFORMANCE
# code `get_stabilized_kernel` when a matrix is passed. When handling relatively
# big problems in DomDec it could be useful to have a sparse Sinkhorn.

"""
    get_stabilized_kernel(c::Function, a, b, ε, X, Y, μ, ν, θ[, colptr0, roval0])

Compute a truncated, stabilized kernel, following Section 3.3 of
https://arxiv.org/abs/1610.06519, using the dual variables `(a, b)`, the threshold `θ`
and a previous estimate on the neighborhood given by `(colptr0, rowval0)`.
These must be interpreted as the representation of a CSC sparse matrix, on a subset of
whose entries the truncated kernel will lie.

If `colptr0, rowval0` are not given, it is assumed that the all entries must be inspected.
"""
function get_stabilized_kernel(c::Function, a, b, ε, X, Y, μ, ν, θ, colptr0, rowval0)
    if θ == 0
        return get_stabilized_kernel_zero_threshold(c, a, b, ε, X, Y, μ, ν, colptr0, rowval0)
    end
    m = length(μ)
    n = length(ν)
    threshold = ε*log(θ)
    rowval = Int[]
    colptr = Vector{Int}(undef, n+1)
    nzval = Float64[]
    # TODO: provide sizehints?
    colptr[1] = 1
    next_ptr = 1
    for j in 1:n
        for r in colptr0[j]:colptr0[j+1]-1 # Indices of non-zero values in column j
            i = rowval0[r]
            log_entry = @views a[i] + b[j] - c(X[:,i], Y[:,j])
            if log_entry ≥ threshold
                push!(rowval, i)
                next_ptr += 1
                push!(nzval, μ[i]*exp(log_entry/ε)*ν[j])
            end
        end
        colptr[j+1] = next_ptr
    end
    return SparseMatrixCSC{Float64, Int}(m, n, colptr, rowval, nzval)
end

function get_stabilized_kernel(c::Function, a, b, ε, X, Y, μ, ν, θ)
    # Iterator over the dense matrix
    m = length(μ)
    n = length(ν)
    # TODO, LOW, PERFORMANCE
    # The following `rowval0` is a vector of size m×n. 
    # An iterator running over `1:m` `n` times would more efficient (for large problems),
    # but it is not implemented in default julia, and it is unclear
    # if it is worth the effort. Consider at a later state.
    colptr0 = collect(1:m:n*m+1)
    rowval0 = repeat(1:m, n)
    get_stabilized_kernel(c::Function, a, b, ε, X, Y, μ, ν, θ, colptr0, rowval0)
end

# Special case: for θ = 0 we can recycle rowval0, colptr0
function get_stabilized_kernel_zero_threshold(c, a, b, ε, X, Y, μ, ν, colptr0, rowval0)
    m = length(μ)
    n = length(ν)
    nzval = Vector{Float64}(undef, length(rowval0))
    for j in 1:n
        for r in colptr0[j]:colptr0[j+1]-1 # Indices of non-zero values in column j
            i = rowval0[r]
            log_entry = @views a[i] + b[j] - c(X[:,i], Y[:,j])
            nzval[r] = μ[i]*exp(log_entry/ε)*ν[j]
        end
    end
    return SparseMatrixCSC{Float64, Int}(m, n, colptr0, rowval0, nzval)
end

# TODO: provide parallel version of `get_stabilized_kernel`

"""
    update_current_Y_marginal!(νK, v, KT, u, ν)

Update the current Y marginal of the coupling and 
return the error with respect to the true marginal `ν`.
"""
function update_current_Y_marginal!(νK, v, KT, u, ν)
    mul!(νK, KT, u)
    νK .*= v
    return l1(ν, νK)
end

"""
    sinkhorn_stabilized!(a, b, μ, ν, K, ε; kwargs...)

Implement the stabilized Sinkhorn algorithm of https://arxiv.org/abs/1610.06519.

"""
function sinkhorn_stabilized!(a, b, μ, ν, K, ε; 
            max_iter = 1000, max_error = 1e-8, 
            max_error_rel=true, verbose = true)

    if max_error_rel
        max_error *= sum(μ)
    end
    # TODO, MEDIUM, DESIGN
    # Decide if the following arguments are important enough
    # to need to go on the optional argument list.

    Ninner = 20 # number of inner iterates # TODO: change this?
    τ = 1e20 # absorption parameter
    # Compute transport; in this case lazy transport
    KT = K'
    u = ones(length(a))
    v = ones(length(b))
    n = 0 # Number of iterations

    # Start with an X-iteration.
    u = K * v
    u .= μ ./ u

    # Compute init error
    νK = similar(ν)
    current_error = update_current_Y_marginal!(νK, v, KT, u, ν)
    status = -1
    while status == -1
        sinkhorn!(u, v, μ, ν, K, KT, Ninner)
        current_error = update_current_Y_marginal!(νK, v, KT, u, ν)
        n += Ninner

        if n ≥ max_iter 
            status = 1
        end
        if isthere_nan_or_inf(u)
            #println("Inf found")
            # Reduce `Ninner` so that absorption works
            Ninner = Ninner÷2
            if Ninner == 0
                # `Ninner` was already 1 and it still failed, so no hope to do better
                status = 2
            end
            # In any case re-initialize u and v and go back to main loop
            v .= 1
            u .= 1
        elseif current_error < max_error
            # Converged
            status = 0
        elseif (maximum(u) > τ) | (maximum(v) > τ)
            # Perform absorption
            K .*= u
            K .*= v'
            # KT is automatically updated since its entries refer to K
            a .+= ε .* log.(u)
            b .+= ε .* log.(v)
            u .= 1
            v .= 1
        end
    end
    if verbose & (status == 1)
        println("warning: Sinkhorn did not converge to accuracy ", current_error)
    end
    # Absorve scalings and turn them into dual potentials
    K .*= u
    K .*= v'
    a .+= ε.*log.(u)
    b .+= ε.*log.(v)
    return status
end
