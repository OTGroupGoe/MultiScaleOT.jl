# CODE STATUS: REVISED, TESTED

# TODO, LOW: faster converting iterators to arrays?
"""
    get_cells_1D(N, cellSize, offset)

Divide `[1,...,N]` into cells of size `cellSize`, with the first cell
having `offset` indices. Return the list of cells and its shape. 

# Examples
```julia-repl
julia> get_cells_1D(4, 2, 0)
([[1, 2], [3, 4]], (2,))

julia> get_cells_1D(4, 2, 1)
([[1], [2, 3], [4]], (3,))
```
"""
function get_cells_1D(N, cellSize, offset = 0)
    cells = Vector{Int}[]
    if offset > 0
        push!(cells, 1:offset)
    end
    for i in offset:cellSize:N-cellSize
        push!(cells, i+1:i+cellSize)
    end
    remainder = (N-offset)%cellSize
    if remainder > 0
        push!(cells, N-remainder+1:N)
    end
    return cells, (length(cells),)
end

"""
    get_product_list(it1, it2, N1)

Obtain the list of flattened indices of the cartesian indices `(i,j)`,
for `j` in `it2` and `i` in `it1`, assuming that the total dimension
of the first axis is `N1`.
"""
function get_product_list(it1, it2, N1)
    return [i + (j-1)*N1 for j in it2 for i in it1]
end

"""
    get_product_list(it1, it2, it3, N1, N2)

Obtain the list of flattened indices of the cartesian indices `(i,j,k)`,
for `k` in `it3`, `j` in `it2` and `i` in `it1`, assuming that the total dimension
of the first and second axes are respectively `N1` and `N2`. 
"""
function get_product_list(it1, it2, it3, N1, N2)
    return [i + (j-1)*N1 + (k-1)*N1*N2 for k in it3 for j in it2 for i in it1]
end

"""
    get_cells_2D(shape, cellSize, offset)

Divide `[1,...,shape[1]]×[1,...,shape[2]]` into cells of size
`(cellSize, cellSize)`, with the given `offset`. The order  of the basic cells
is lexicographic, with the first index running faster. 
Return the list of cells and its shape. 

# Examples
```julia-repl
julia> get_cells_2D((4,4), 2, 0)
([[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]], (2, 2))
```
"""
function get_cells_2D(shape, cellSize, offset)
    l1, _ = get_cells_1D(shape[1], cellSize, offset)
    l2, _ = get_cells_1D(shape[2], cellSize, offset)

    return [get_product_list(i1,i2,shape[1]) for i2 in l2 for i1 in l1], (length(l1), length(l2))
end

"""
    get_cells_3D(shape, cellSize, offset)

Divide `[1,...,shape[1]]×[1,...,shape[2]]×[1,...,shape[3]]` into cells of size
`(cellSize, cellSize, cellSize)`, with the given `offset`. The order  of the basic cells
is lexicographic, with the first index running faster. 
Return the list of cells and its shape. 
"""
function get_cells_3D(shape, cellSize, offset)
    l1, _ = get_cells_1D(shape[1], cellSize, offset)
    l2, _ = get_cells_1D(shape[2], cellSize, offset)
    l3, _ = get_cells_1D(shape[3], cellSize, offset)

    return [get_product_list(i1, i2, i3, shape[1], shape[2]) for i3 in l3 for i2 in l2 for i1 in l1], (length(l1), length(l2), length(l3))
end

"""
    get_cells(shape, cellSize, offset=0)

Call either `get_cells_1D`, `get_cells_2D` or `get_cells_3D` depending on the
size of `shape`. Higher dimensional not implemented.
"""
function get_cells(shape::NTuple{N, Int}, cellSize, offset=0) where N
    if N == 1
        return get_cells_1D(shape[1], cellSize, offset)
    elseif N == 2
        return get_cells_2D(shape, cellSize, offset)
    elseif N == 3
        return get_cells_3D(shape, cellSize, offset)
    else
        error("not implemented for dimension > 3.")
    end
end

