# CODE STATUS: REVISED, TESTED

abstract type AbstractMeasure{D} end

"""
    GridMeasure{D} <: AbstractMeasure{D}

A measure supported on a `D`-dimensional grid. Its attributes are:
* `points::Matrix{Float64}`: `points[:,j]` are the coordinates of point `j`.
* `weights::Vector{Float64}`: `weights[j]` is the mass of point `j`.
* `gridshape::NTuple{D,Int}`: shape of the grid, i.e. (length of x1, length of x2,...)
"""
mutable struct GridMeasure{D} <: AbstractMeasure{D}
    points::Matrix{Float64}   # X, coordinates
    weights::Vector{Float64}  # masses, μ
    gridshape::NTuple{D, Int} # shapeX

    function GridMeasure(points, weights, gridshape::NTuple{D, Int}) where D
        (size(points, 1) == D) || error("Number of rows of points must equal D")
        size(points, 2) == length(weights) || error("Number of columns of points must equal length of weights")
        prod(gridshape) == size(points, 2) || error("Number of columns of points must equal prod(gridshape)")
        new{D}(points, weights, gridshape)
    end
end

# TODO, MEDIUM, ENHANCEMENT
# Support CloudMeasure to the same extent as GridMeasure, with refinement of the alhpas included.
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

    function CloudMeasure(points, weights)
        D = size(points, 1)
        extents = Tuple(extrema(r) for r in eachrow(points))::NTuple{D, Tuple{Float64, Float64}}
        new{D}(points, weights, extents)
    end
end

# TODO: Here also MultiScaleMeasure?