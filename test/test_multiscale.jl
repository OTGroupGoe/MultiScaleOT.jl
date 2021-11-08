import MultiScaleOT: GridMeasure, CloudMeasure, flat_grid
using SparseArrays

@testset ExtendedTestSet "MultiScaleMeasure" begin
    # Same setup as in test_measure
    x1 = [1,2,3,4]
    X = MOT.flat_grid(x1)
    weights = [1,2,3,4]
    shape = (4,)
    mu = MOT.GridMeasure(X, weights, shape)

    # Compute corresponding MultiScaleMeasure
    MS_mu = MOT.MultiScaleMeasure(mu)

    # Precomputed solution:
    mu1 = MOT.GridMeasure(reshape([2.5], 1, 1), [10.], (1,))
    mu2 = MOT.GridMeasure([1.5 3.5], [3., 7.], (2,))
    mu3 = mu

    for (i, mui) in enumerate([mu1, mu2, mu3])
        MS_mui = MS_mu.measures[i]
        @test (MS_mui.points == mui.points) & (MS_mui.weights == mui.weights) & (MS_mui.gridshape == mui.gridshape)
    end

    # In 2D
    x2 = [1, 2, 3]
    X = MOT.flat_grid(x1, x2)
    gridshape = (length(x1), length(x2))
    weights = collect(1. : prod(gridshape))
    mu = MOT.GridMeasure(X, weights, gridshape)

    MS_mu = MOT.MultiScaleMeasure(mu)

    mu1 = MOT.GridMeasure(reshape([2.5; 2.25], :, 1), [78], (1,1))
    mu2 = MOT.GridMeasure([1.5 3.5 1.5 3.5; 1.5 1.5 3 3], [14, 22, 19, 23], (2,2))
    mu3 = mu

    for (i, mui) in enumerate([mu1, mu2, mu3])
        MS_mui = MS_mu.measures[i]
        @test (MS_mui.points == mui.points) & (MS_mui.weights == mui.weights) & (MS_mui.gridshape == mui.gridshape)
    end
end


@testset ExtendedTestSet "refine_support" begin
    A = sparse([
        1 0 0 0 
        0 1 0 0
        0 0 0 1
        0 0 1 0
        ]
    )

    m0 = 4
    n0 = 4
    refinementX = [[1], [2,3], [4], [5,6]]
    refinementY = [[1], [2], [3,4,5], [5,6]]
    m = 6
    n = 6
    
    # Do with the library function
    colptr, rowval = MOT.refine_support(m0, n0, A.colptr, A.rowval, m, n, refinementX, refinementY)
    A2 = SparseMatrixCSC{Int, Int}(m, n, colptr, rowval, ones(Int, length(rowval)))

    # Do the operation on dense matrix
    A2_test = zeros(m, n)
    for j in 1:n0
        for i in 1:m0
            A2_test[refinementX[i], refinementY[j]] .= A[i,j]
        end
    end
    
    @test sparse(A2_test) == sparse(A2)
    
    # With some randomly generated A
    m = 60
    n = 93
    m0 = m÷2
    n0 = n÷3

    A = sprand(m0, n0,  0.1)
    A[A.>0] .= 1.
    refinementX, _ = MOT.get_cells_1D(m, 2)
    refinementY, _ = MOT.get_cells_1D(n, 3)


    # Do with the library function
    colptr, rowval = MOT.refine_support(m0, n0, A.colptr, A.rowval, m, n, refinementX, refinementY)
    A2 = SparseMatrixCSC{Float64, Int}(m, n, colptr, rowval, ones(length(rowval)))

    # Do the operation on dense matrix
    A2_test = zeros(m, n)
    for j in 1:n0
        for i in 1:m0
            A2_test[refinementX[i], refinementY[j]] .= A[i,j]
        end
    end

    @test sparse(A2_test) == sparse(A2)

end