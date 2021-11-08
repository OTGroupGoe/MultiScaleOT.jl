import MultiScaleOT: GridMeasure, CloudMeasure, flat_grid

@testset ExtendedTestSet "GridMeasure" begin
    N1 = 10
    x1 = collect(1.:N1)
    X = flat_grid(x1)
    w = rand(N1)
    shape = (N1,)

    @test typeof(GridMeasure(X, w, shape)) == GridMeasure{1}
    
    # Test giving the wrong numer of dimensions
    @test_throws ErrorException GridMeasure(X, w, (N1, 1))

    # Test giving wrong dimension
    @test_throws ErrorException GridMeasure(X, w, (N1+1,))

    # Test giving wrong length of weights
    push!(w, 1)
    @test_throws ErrorException GridMeasure(X, w, shape)

    # 2D 
    N2 = 5
    x2 = collect(1.:N2)
    X = flat_grid(x1, x2)
    w = rand(N1*N2)
    shape = (N1, N2)
    @test typeof(GridMeasure(X, w, shape)) == GridMeasure{2}

    # 3D 
    N3 = 3
    x3 = collect(1.:N3)
    X = flat_grid(x1, x2, x3)
    w = rand(N1*N2*N3)
    shape = (N1, N2, N3)
    @test typeof(GridMeasure(X, w, shape)) == GridMeasure{3}
end

@testset ExtendedTestSet "CloudMeasure" begin
    N = 10
    X = rand(1, N)
    w = rand(N)

    Mu = CloudMeasure(X, w)
    @test typeof(Mu) == CloudMeasure{1}
    a, b = Mu.extents[1]
    a2, b2 = extrema(X)
    @test a ≤ a2 ≤ b2 ≤ b
    
    # Test giving the wrong numer of dimensions
    @test_throws ErrorException CloudMeasure(X, w, ((a, b), (a, b)))

    # Test giving the wrong extrema
    @test_throws ErrorException CloudMeasure(X, w, ((a+1, b+1),))

    # Test giving wrong length of weights
    push!(w, 1)
    @test_throws ErrorException CloudMeasure(X, w, ((a, b),))


    # 2D 
    X = rand(2, N)
    w = rand(N)
    @test typeof(CloudMeasure(X, w)) == CloudMeasure{2}

    # 3D 
    X = rand(3, N)
    w = rand(N)
    @test typeof(CloudMeasure(X, w)) == CloudMeasure{3}
end

@testset ExtendedTestSet "fine_to_coarse(<:GridMeasure)" begin
    # Fine to coarse
    # 1D
    # Some setup: points on the line [1,2,3,4], with masses [1,2,3,4]
    x1 = [1,2,3,4]
    X = MOT.flat_grid(x1)
    weights = [1,2,3,4]
    shape = (4,)
    Mu = MOT.GridMeasure(X, weights, shape)

    # Try for different cellsizes
    mu2, cells = MOT.fine_to_coarse(Mu, 2)
    mu2test = MOT.GridMeasure([1.5 3.5], [3., 7.], (2,))
    @test (mu2test.points == mu2.points) & (mu2test.weights == mu2.weights) & (mu2test.gridshape == mu2.gridshape)
    @test all(mu2.weights[i] == sum(Mu.weights[cell]) for (i, cell) in enumerate(cells))

    mu2, cells = MOT.fine_to_coarse(Mu, 3)
    mu2test = MOT.GridMeasure([2 4], [6, 4], (2,))
    @test (mu2test.points == mu2.points) & (mu2test.weights == mu2.weights) & (mu2test.gridshape == mu2.gridshape)
    @test all(mu2.weights[i] == sum(Mu.weights[cell]) for (i, cell) in enumerate(cells))

    mu2, cells = MOT.fine_to_coarse(Mu, 4)
    mu2test = MOT.GridMeasure(reshape([2.5],1,1), [10], (1,))
    @test (mu2test.points == mu2.points) & (mu2test.weights == mu2.weights) & (mu2test.gridshape == mu2.gridshape)
    @test all(mu2.weights[i] == sum(Mu.weights[cell]) for (i, cell) in enumerate(cells))

    # 2D, expand setup and do the same
    # In this case we have the a 2D grid with following masses
    #    \ x2  1  2  3
    # x1    --------------
    #  1   |   1  5  9
    #  2   |   2  6 10
    #  3   |   3  7 11
    #  4   |   4  8 12
    x2 = [1, 2, 3]
    X = MOT.flat_grid(x1, x2)
    gridshape = (length(x1), length(x2))
    weights = collect(1. : prod(gridshape))
    Mu = MOT.GridMeasure(X, weights, gridshape)

    # Test for different cellsizes
    mu2, cells = MOT.fine_to_coarse(Mu, 2)
    mu2test = MOT.GridMeasure([1.5 3.5 1.5 3.5; 1.5 1.5 3 3], [14, 22, 19, 23], (2,2))
    @test (mu2test.points == mu2.points) & (mu2test.weights == mu2.weights) & (mu2test.gridshape == mu2.gridshape)
    @test all(mu2.weights[i] == sum(Mu.weights[cell]) for (i, cell) in enumerate(cells))

    mu2, cells = MOT.fine_to_coarse(Mu, 3)
    mu2test = MOT.GridMeasure([2 4; 2 2], [54, 24], (2,1))
    @test (mu2test.points == mu2.points) & (mu2test.weights == mu2.weights) & (mu2test.gridshape == mu2.gridshape)
    @test all(mu2.weights[i] == sum(Mu.weights[cell]) for (i, cell) in enumerate(cells))

    mu2, cells = MOT.fine_to_coarse(Mu, 4)
    mu2test = MOT.GridMeasure(reshape([2.5, 2],:,1), [78], (1,1))
    @test (mu2test.points == mu2.points) & (mu2test.weights == mu2.weights) & (mu2test.gridshape == mu2.gridshape)
    @test all(mu2.weights[i] == sum(Mu.weights[cell]) for (i, cell) in enumerate(cells))

    # TODO: 3D test?
end

# TODO: coarse_to_fine CloudMeasure