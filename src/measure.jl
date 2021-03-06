# CODE STATUS: REVISED, TESTED

"""
    AbstractMeasure{D}

Type from which `GridMeasure{D}, CloudMeasure{D}` inherit. 
It is left unexported to avoid potential conflicts with 
`MeasureTheory.jl`, which has another type with the same name.
"""
abstract type AbstractMeasure{D} end

"""
    GridMeasure{D} <: AbstractMeasure{D}

A measure supported on a `D`-dimensional grid. Its attributes are:
* `points::Matrix{Float64}`: `points[:,j]` are the coordinates of point `j`.
* `weights::Vector{Float64}`: `weights[j]` is the mass of point `j`.
* `gridshape::NTuple{D,Int}`: shape of the grid, i.e. (length of x1, length of x2,...)
"""
mutable struct GridMeasure{D} <: AbstractMeasure{D}
    points::Matrix{Float64}   # coordinates of the points
    weights::Vector{Float64}  # masses
    gridshape::NTuple{D, Int} # shape

    function GridMeasure(points, weights, gridshape::NTuple{D, Int}) where D
        (size(points, 1) == D) || error("Number of rows of points must equal D")
        size(points, 2) == length(weights) || error("Number of columns of points must equal length of weights")
        prod(gridshape) == size(points, 2) || error("Number of columns of points must equal prod(gridshape)")
        new{D}(points, weights, gridshape)
    end
end

# TODO: test
function copy(mu::GridMeasure{D}) where D
    GridMeasure(mu.points, mu.weights, mu.gridshape)
end

# TODO: test
function npoints(mu::GridMeasure{D}) where D
    length(mu.weights)
end

# TODO: test
function mass(mu::GridMeasure{D}) where D
    sum(mu.weights)
end

# TODO: test
function ==(mu::GridMeasure, nu::GridMeasure)
    (mu.points == nu.points) & (mu.weights == nu.weights) & (mu.gridshape == nu.gridshape)
end

"""
    fine_to_coarse(mu::GridMeasure{D}, cellsize)

Compute a coarse approximation of `mu` by combining 
`cellsize × ... × cellsize` points together. 
Return the coarser Measure `new_mu` and the partition `cells`,
such that `sum(mu.weights[cells[i]]) == new_mu.weights[i]`.
"""
function fine_to_coarse(mu::GridMeasure{D}, cellsize) where D
    cells, new_gridshape = get_cells(mu.gridshape, cellsize)
    new_weights = [sum(mu.weights[cell]) for cell in cells]
    # New points
    xs = get_grid_nodes(mu.points, mu.gridshape)
    new_xs = Vector{Float64}[]
    for i in 1:D
        cells_xi, _ = get_cells_1D(length(xs[i]), cellsize)
        new_xi = [mean(xs[i], cell) for cell in cells_xi]
        push!(new_xs, new_xi)
    end
    new_points = flat_grid(new_xs...)
    new_mu = GridMeasure(new_points, new_weights, new_gridshape)
    return new_mu, cells
end

"""
    CloudMeasure{D} <: AbstractMeasure{D}

A measure supported on an unstructured `D`-dimensional point cloud. Its attributes are:
* `points::Matrix{Float64}`: `points[:,j]` are the coordinates of point `j`.
* `weights::Vector{Float64}`: `weights[j]` is the mass of point `j`.
* `extent::NTuple{D, Tuple{Float64, Float64}}`: `extent[i]` are lower and upper bounds
    on the `i`-th row of `points`.
"""
mutable struct CloudMeasure{D} <: AbstractMeasure{D}
    points::Matrix{Float64}
    weights::Vector{Float64}
    extents::NTuple{D, Tuple{Float64, Float64}}

    function CloudMeasure(points, weights, extents::NTuple{D, Tuple{Float64, Float64}}) where D
        (size(points, 1) == D) || error("Number of rows of points must equal D")
        size(points, 2) == length(weights) || error("Number of columns of points must equal length of weights")
        for d in 1:D 
            a, b = extents[d]
            a2, b2 = extrema(@views points[d,:])
            (a <= a2 <= b2 <= b) || error("some column of points is outside given extent")
        end

        new{D}(points, weights, extents)
    end

    function CloudMeasure(points, weights, extents::NTuple{D, Tuple{Number, Number}}) where D
        extents_float = convert.(NTuple{2, Float64}, extents)
        new{D}(points, weights, extents_float)
    end

    function CloudMeasure(points, weights)
        D = size(points, 1)
        extents = Tuple(extrema(r) for r in eachrow(points))::NTuple{D, Tuple{Float64, Float64}}
        new{D}(points, weights, extents)
    end
end

# TODO: test
function copy(mu::CloudMeasure{D}) where D
    CloudMeasure(mu.points, mu.weights, mu.extents)
end

# TODO: test
function npoints(mu::CloudMeasure{D}) where D
    length(mu.weights)
end

# TODO: test
function mass(mu::CloudMeasure{D}) where D
    sum(mu.weights)
end

# TODO: test
function ==(mu::CloudMeasure, nu::CloudMeasure)
    (mu.points == nu.points) & (mu.weights == nu.weights) & (mu.extents == nu.extents)
end

# TODO, MEDIUM, ENHANCEMENT
# Support CloudMeasure to the same extent as GridMeasure, with refinement of the alhpas included.

function fine_to_coarse(mu::CloudMeasure{D}, nbins) where D
    # Just some ideas here: CloudMeasure{D} could also have a 
    # `shape` attribute signaling the number of squares in which
    # we have divide the extent of a finer CloudMeasure to get some
    # discretization. Then, if the CloudMeasure was not the result
    # of this operation, `shape` would be `(-1, -1).`

    # Cover the case of empty measure
    if npoints(mu) == 0
        return deepcopy(mu)
    end        
    # Get lower-left and top-right corner
    a = [ex[1] for ex in mu.extents]
    b = [ex[2] for ex in mu.extents]
    a .-= (b.-a)*1e-10 # enlarge box slightly to avoid introducing another box
    # For each column of X, get the linear index of its box in the 
    # coarse measure (up to some constant offset)
    X = mu.points
    e = nbins.^(0.:D-1)
    I = [Int(dot(e, ceil.(nbins.*(x.-a)./(b.-a)))) for x in eachcol(X)]
    # Get permutation that would sort I
    perm = sortperm(I)
    # Going in this order, start adding points to the new measure. The weights of a cluster
    # of points will be their sum, their position will be their euclidean barycenter
    i = perm[1]
    current_ind = I[i]
    new_weights = [mu.weights[i]]
    new_X = [mu.weights[i].*X[:,i]]
    cells = [[i]]
    for k in 2:length(perm)
        i = perm[k]
        if I[i] == current_ind
            new_weights[end] += mu.weights[i]
            @inbounds new_X[end] .+= @views mu.weights[i].*X[:,i]
            push!(cells[end], i)
        else
            current_ind = I[i]
            push!(new_weights, mu.weights[i])
            push!(new_X, @inbounds mu.weights[i]*X[:,i])
            push!(cells, [i])
        end
    end
    # Build new points matrix
    new_points = hcat(new_X...)
    new_points ./= new_weights'
    # Extents theoretically remain unaltered, however our way to compute the barycenter 
    # leads to a small error. So we let the constructor of CloudMeasure compute 
    # the nex extents; if this is not good enough we can revisit it at a later moment.
    return CloudMeasure(new_points, new_weights), cells
end