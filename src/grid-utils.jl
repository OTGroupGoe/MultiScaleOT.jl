
"""
    flat_grid(xs...)

Create a grid with the arguments `xs...` and flatten its points
to a matrix. Each column of the matrix represents one point on the grid, 
and the first coordinate runs faster. 

# Examples

```julia-repl
julia> x = [1, 2]; y = [2, 3];
julia> flat_grid(x, y)

2×4 Matrix{Int64}:
 1  2  1  2
 2  2  3  3
```
"""
function flat_grid(xs::Vector{T}...) where T
    n = length(xs)
    stride1 = 1
    stride2 = prod(length.(xs))
    X = Matrix{T}(undef, n, stride2)
    for i in 1:n
        stride2 ÷= length(xs[i])
        X[i,:] .= repeat(repeat(xs[i], stride2), inner = stride1)
        stride1 *= length(xs[i])
    end
    return X
end

"""
    get_grid_nodes(X, shapeX)

For `X` of size `(N, prod(shapeX))` on a grid, get its 1-dimensional factors.

# Examples
```julia-repl
julia> X = [1 2 1 2 1 2
            1 1 2 2 3 3];
julia> get_grid_nodes(X, (2, 3))
([1, 2], [1, 2, 3])
```
"""
function get_grid_nodes(X::Matrix{T}, shapeX::NTuple{N, Int}) where {T,N}
    grid_nodes = Vector{T}[]
    stride::Int = 1
    for (i, s) in enumerate(shapeX)
        x = X[i, 1:stride:stride*s]
        push!(grid_nodes, x)
        stride *= s
    end
    
    return Tuple(grid_nodes)::NTuple{N, Vector{T}}
end

# Interpolation utils

"""
    pad_extrapolate(x::AbstractVector{T})

Extrapolate a vector of length `N` at both ends. 
The result is a vector of length `N+2`. 

# Examples
```julia-repl
julia> pad_extrapolate([1,2,3])
5-element Vector{Int64}:
 0
 1
 2
 3
 4
```
"""
function pad_extrapolate(x::AbstractVector{T}) where T
    y = Vector{T}(undef, length(x)+2)
    pad_extrapolate!(y, x)
    return y
end

# TODO, LOW, DESIGN
# Do we really need to have a mutating version?
function pad_extrapolate!(y, x)
    if length(x) == 1
        y .= x[1]
    else
        @views begin
            y[2:end-1] .= x
            y[1] = 2*y[2] - y[3]
            y[end] = 2*y[end-1] - y[end-2]
        end
    end
end

"""
    pad_extrapolate(A::Matrix)

Extrapolate a `(N,M)` matrix along its boundary. The result is a 
`(N+2, M+2)` matrix. 

# Examples
```julia-repl
julia> pad_extrapolate([1 2; 3 4])
4×4 Matrix{Float64}:
 -2.0  -1.0  0.0  1.0
  0.0   1.0  2.0  3.0
  2.0   3.0  4.0  5.0
  4.0   5.0  6.0  7.0
```
"""
function pad_extrapolate(A::Matrix{T}) where T
    B = zeros(T, size(A) .+ 2)
    # Probably better to cycle over cols and rows?_
    @views begin 
        B[2:end-1, 2:end-1] .= A
        B[2:end-1, 1] .= 2 .* B[2:end-1, 2] .- B[2:end-1, 3]
        B[2:end-1, end] .= 2 .*B[2:end-1, end-1] .- B[2:end-1, end-2]
        B[1, 1:end] .= 2 .*B[2,1:end] .- B[3, 1:end]
        B[end, 1:end] .= 2 .*B[end-1,1:end] .- B[end-2, 1:end]
    end
    return B
end