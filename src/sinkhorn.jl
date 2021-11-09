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


# TODO, LOW-MEDIUM, PARALLELIZATION
# Provide threaded version of the `get_stabilized_kernel` function using the comented code.
# function get_stabilized_kernel_colrange(c::Function, α, β, ε, X, Y, μ, ν, θ, P0, J::UnitRange)
#     threshold = ε*log(θ)
#     rowval = Int[]
#     colptr = Vector{Int}(undef, length(J))
#     offset_j = J[1] - 1
#     nzval = Float64[]
#     rows = rowvals(P0) 
#     next_colptr = 0
#     for j in J
#         for r in nzrange(P0, j)
#             i = rows[r]
#             stab_cost = @views α[i] + β[j] - c(X[:,i], Y[:,j])
#             if stab_cost ≥ threshold
#                 push!(rowval, i)
#                 next_colptr += 1
#                 push!(nzval, μ[i]*exp(stab_cost/ε)*ν[j])
#             end
#         end
#         colptr[j-offset_j] = next_colptr
#     end
#     return rowval, colptr, nzval
# end

# function get_stabilized_kernel(c::Function, α, β, ε, X, Y, μ, ν, θ, P0)
#     m, n = size(P0)
#     rowval, colptr, nzval = get_stabilized_kernel_colrange(c, α, β, ε, X, Y, μ, ν, θ, P0,1:n)
#     pushfirst!(colptr, 0)
#     colptr .+= 1
#     K = SparseMatrixCSC{Float64, Int}(m, n, colptr, rowval, nzval)
# end

# function get_stabilized_kernel_threaded(c::Function, α, β, ε, X, Y, μ, ν, θ, P0)
#     m, n = size(P0)
    
#     Nchunks = Threads.nthreads()
#     #Nchunks = 1
#     Rowvals = [Int[] for i in 1:Nchunks]
#     Colptrs = [Int[] for i in 1:Nchunks]
#     Nzvals = [Float64[] for i in 1:Nchunks]
    
#     chunk_size = n÷Nchunks
#     j_chunks = [(k-1)*chunk_size+1:k*chunk_size for k in 1:Nchunks]
#     j_chunks[end] = (Nchunks-1)*chunk_size+1:n
    
#     #for k in 1:Nchunks
#     Threads.@threads for k in 1:Nchunks
#         Rowvals[k], Colptrs[k], Nzvals[k] = get_stabilized_kernel_colrange(c, α, β, ε, X, Y, μ, ν, θ, P0, j_chunks[k])
#     end
    
#     # These can be be concatted right away
#     # TODO: do in different threads
#     rowval = vcat(Rowvals...)::Vector{Int}
#     nzval = vcat(Nzvals...)::Vector{Float64}
    
#     # Colptrs must be adjusted by the offset of the previous colptr
#     # TODO: do in a different thread
#     offset = 1
#     for k in 1:Nchunks
#         Colptrs[k] .+= offset
#         offset = Colptrs[k][end]
#     end
    
#     colptr = vcat([1], Colptrs...)::Vector{Int}
    
#     #return colptr, rowval, nzval
#     SparseMatrixCSC{Float64, Int}(m, n, colptr, rowval, nzval)
# end

# TODO, HIGH, DECISION: Is the following version needed?
# function sparse_sinkhorn(α, β, μ, ν, C, ε, I, J; kwargs...)
    
#     if max_error_rel
#         max_error *= sum(μ)
#     end
    
#     #α, β assumed to be already non-too-bad potentials
#     K = get_stabilized_kernel(C, α, β, ε, I, J, μ, ν)
#     sparse_sinkhorn_stabilized_kernel(α, β, μ, ν, K, ε; kwargs...)
# end

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

# TODO: docstring
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
    error = update_current_Y_marginal!(νK, v, KT, u, ν)
    status = -1
    while status == -1
        sinkhorn!(u, v, μ, ν, K, KT, Ninner)
        error = update_current_Y_marginal!(νK, v, KT, u, ν)
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
        elseif error < max_error
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
        println("warning: Sinkhorn did not converge to accuracy ",error)
    end
    # Absorve scalings and turn them into dual potentials
    K .*= u
    K .*= v'
    a .+= ε.*log.(u)
    b .+= ε.*log.(v)
    return status
end

# function sparse_sinkhorn_stabilized_kernel_threaded(α, β, μ, ν, K, ε; 
#     max_iter = 1000, max_error = 1e-8, 
#     max_error_rel=true, verbose = true)

#     #println(isthere_nan_or_inf([α; β]))
#     Ninner = 20 # number of inner iterates # TODO: change this?
#     τ = 1e20 # absorption parameter
#     u = ones(length(α))
#     v = ones(length(β))
#     n = 0
#     Kv = K * v
#     u .= μ ./ Kv
#     KT = transpose(K)
#     Ku = (KT * u)
#     #println(extrema([α; β]))

#     KTT = transpose(sparse(KT))

#     # Compute init error
#     νK = Ku .* v
#     error = l1(ν, νK)
#     status = -1
#     while status == -1
#         sinkhorn!(u, v, μ, ν, Ku, Kv, KTT, KT, Ninner) #KTT is a tranpose(sparse), so MKL is efficient
#         νK = Ku .* v
#         error = l1(ν, νK)
#         n += Ninner
#         if n ≥ max_iter 
#             status = 1
#         end
#         if isthere_nan_or_inf(u) 
#             #println("Inf found")
#             # Reduce `Ninner` so that absorption works
#             Ninner = Ninner÷2
#             if Ninner == 0
#                 # `Ninner` was already 1 and it still failed, so no hope to do better
#                 verbose && println("Inf found.")
#                 status = 2
#                 return K, α, β, status
#             end
#             # In any case re-initialize u and v and go back to main loop
#             v .= 1
#             u .= 1
#         elseif error < max_error
#             # Converged
#             status = 0
#         elseif (maximum(u) > τ) | (maximum(v) > τ)
#             # Perform absorption
#             K .*= u
#             K .*= v'
#             # KT is automatically updated, KTT must be updated manually
#             KTT .*= u
#             KTT .*= v'
#             α .+= ε .* log.(u)
#             β .+= ε .* log.(v)
#             u .= 1
#             v .= 1
#         end
#         # Safety check
#         mul!(Ku, KT, u)
#     end
#     if verbose & (status == 1)
#         println("warning: Sinkhorn did not converge to accuracy ",error)
#     end
#     # Absorve scalings and turn them into dual potentials
#     K .*= u
#     K .*= v'
#     # No need to update KT since we are done with it
#     α .+= ε.*log.(u)
#     β .+= ε.*log.(v)
#     #println(isthere_nan_or_inf([α; β    ]))
#     return K, α, β, status
# end

# TODO, MEDIUM, DOMDEC
# `sinkhorn_autofix` needs the matrix `C` instead of `K`, 
# (or `c, X, Y`, which might seem excesive. To not lose the 
# consistency within this package it might be better to 
# just define it in DomDec.


# function sparse_sinkhorn(α, β, μ, ν, C, ε; kwargs...)
#     # Initialize dual scalings
#     # β is assumed to be alright
#     # TODO: is this sufficiently efficient?
#     α .= get_missing_dual_potential(C, β, ν, ε, 2)
#     θ = 1e-14 # truncation parameter
#     I, J = get_neigh_indices(C, α, β, ε, θ)
#     sparse_sinkhorn(α, β, μ, ν, C, ε, I, J; kwargs...)
# end

# function sparse_sinkhorn_autofix(α, β, μ, ν, C, ε; kwargs...)
#     α .= get_missing_dual_potential(C, β, ν, ε, 2)
#     α0 = copy(α)
#     β0 = copy(β)
#     #println("\t", μsub)
#     eps_factor = 2
#     #eps_factor = 4/3
#     θ = 1e-14
#     I, J = get_neigh_indices(C, α, β, ε, θ)
#     K, α, β, status = sparse_sinkhorn(α, β, μ, ν, C, ε, I, J; kwargs...)
    
#     if status != 2
#         return K, α, β, status
#     else
#         # Upward branch
#         eps_doubling_steps = 0
#         print("Increasing epsilon x")
#         while status == 2
#             # Reset dual potentials
#             α .= α0
#             β .= β0
#             ε *= eps_factor
#             eps_doubling_steps += 1
#             print(" ", eps_doubling_steps)
#             I, J = get_neigh_indices(C, α, β, ε, θ)
#             K, α, β, status = sparse_sinkhorn(α, β, μ, ν, C, ε, I, J; kwargs...)
#             if eps_doubling_steps == 100
#                 status = 1
#             end
#         end
#         # Downward branch
#         print("Decreasing epsilon x")
#         for i in 1:eps_doubling_steps
#             # Save dual potentials
#             α0 .= copy(α)
#             β0 .= copy(β)
#             # println(a0)
#             ε /= eps_factor
#             print(" ", i)
#             # Update neighborhood with the info from prev. iteration
#             I, J = get_neigh_indices(C, α, β, ε, θ, I, J)
#             K, α, β, status = sparse_sinkhorn(α, β, μ, ν, C, ε, I, J; kwargs...)
#             # println(status)
#             if status == 2
#                 # Give up at this point
#                 print(" not successful.\n")
#                 α .= α0
#                 β .= β0
#                 ε *= eps_factor
#                 return K, α, β, status
#             end
#         end
#         # If the downward branch went well, return status
#         print(" successful!\n")
#         return K, α, β, status
#     end
# end