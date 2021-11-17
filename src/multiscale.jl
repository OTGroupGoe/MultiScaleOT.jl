# CODE STATUS: REVISED, TESTED
import Interpolations

# TODO: Test
"""
    compute_multiscale_depth(mu::GridMeasure{D})    

Compute the minimum multiscale depth to refine a measure
consisting of a single point to `mu` by dividing each point
iteratively into (at most) 2^D points
"""
compute_multiscale_depth(mu) = maximum(Int.(ceil.(log2.(mu.gridshape))))+1

# TODO: How to handle this for arbitrary measures not living on a grid?

"""
    MultiScaleMeasure{<:AbstractMeasure}

Type encoding a measure at different resolutions, and how to refine a coarse
version to a finer version. Inner representation is
```julia
mutable struct MultiScaleMeasure{M<:AbstractMeasure{D} where D} 
    depth::Int # Number of resolutions
    measures::Vector{M} # `measures[i]` gives resolution `i`
    refinements::Vector{Vector{Vector{Int}}} # `refinements[i][j]` yields the indices of
                                             # `mesures[i+1]` into which the atom
                                             # `j` of `measures[i]` is splitted
end
```
"""
mutable struct MultiScaleMeasure{M<:AbstractMeasure{D} where D} 
    depth::Int
    measures::Vector{M}
    refinements::Vector{Vector{Vector{Int}}}

    function MultiScaleMeasure(depth, measures, refinements)
        M = eltype(measures)
        # TODO: perform consistency check
        new{M}(depth, measures, refinements)
    end
end

# TODO, LOW, ECONOMY
# Do we need to do one version of the function MultiScaleMeasure(Mu)
# for each subtype of AbstractMeasure or one could fit all?
"""
    MultiScaleMeasure(mu[; depth])

Compute the MultiScaleMeasure with `depth` levels and finest 
resolution given by `mu`.
"""
function MultiScaleMeasure(mu::GridMeasure{D}; depth = -1) where D
    isa(depth, Int) || error("depth must be of type Int")
    measures = [mu]
    if depth < 0 # Desired depth not handled
        depth = compute_multiscale_depth(mu)
    else
        depth = min(compute_multiscale_depth(mu), depth)
    end

    # Refine original measure until final layer
    refinements = Vector{Vector{Int}}[]
    cellsize = 2 # Each point splitted into (pontetially) cellsize^D points.
    # TODO: Make `cellsize` optional?
    for i in 1:depth-1
        new_mu, cells = fine_to_coarse(mu, cellsize)
        pushfirst!(measures, new_mu)
        pushfirst!(refinements, cells)
        mu = new_mu
    end
    return MultiScaleMeasure(depth, measures, refinements)
end

"""
    refine_support(colptr0, rowval0, refinementX, refinementY)

Seeing `colptr0, rowval0` as the support of a sparse matrix `K0`, 
obtain a new support `colptr, rowval` that is the support of the
matrix `K` resulting from the refinements `refinementX, refinementY`
"""
function refine_support(m0, n0, colptr0, rowval0, m, n, refinementX, refinementY)
    rowval_scattered = [Int[] for _ in 1:n] # Entry `i` is the support of column `i`
    colptr = Vector{Int}(undef, n+1) # Entry `i+1` is the length of col `i`; a cumsum then yields the correct colptr.
    colptr[1] = 1 
    for j in 1:n0
        newcol = Int[]
        for r in colptr0[j]:colptr0[j+1]-1 # Indices of non-zero values in column j
            i = rowval0[r]
            for k in refinementX[i]
                push!(newcol, k)
            end
        end
        sort!(newcol)
        # Copy same column to all the refined Ys
        for k in refinementY[j]
            rowval_scattered[k] = newcol
            colptr[k+1] = length(newcol)
        end
    end
    
    # Concatenate all columns
    rowval = vcat(rowval_scattered...)
    # Do a cumsum inplace to get the correct colptr
    cumsum!(colptr, colptr)
    
    return colptr, rowval     
end

# TODO: Add tests!
function refine_dual(Z, nodes, newX, prev_shape::NTuple{N, Int}) where N
    Z2 = reshape(Z, prev_shape)
    ext_Z = pad_extrapolate(Z2)
    ext_nodes = pad_extrapolate.(nodes)
    int = Interpolations.interpolate(Tuple(ext_nodes), ext_Z, Interpolations.Gridded(Interpolations.Linear()))
    new_Z = [int(x...) for x in eachcol(newX)]
    return new_Z
end

