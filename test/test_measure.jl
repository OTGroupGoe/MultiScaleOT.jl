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