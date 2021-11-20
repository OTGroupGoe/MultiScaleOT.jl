@testset ExtendedTestSet "flat_grid and get_grid_nodes" begin
    x = [1,2]
    y = [1,2,3]
    z = [2,3]

    X1D = [1 2]
    X2D = [
        1 2 1 2 1 2
        1 1 2 2 3 3
    ]
    X3D = [
        1 2 1 2 1 2 1 2 1 2 1 2
        1 1 2 2 3 3 1 1 2 2 3 3
        2 2 2 2 2 2 3 3 3 3 3 3 
    ]

    # 1D
    @test MOT.flat_grid(x) == X1D

    # 2D
    @test MOT.flat_grid(x,y) == X2D
    @test_throws MethodError MOT.flat_grid([1, 2.], y) # They need to be the same type
    
    # 3D
    @test MOT.flat_grid(x,y,z) == X3D

    # get_grid_nodes
    
    shape1D = (2,)
    shape2D = (2, 3)
    shape3D = (2, 3, 2)
    @test MOT.get_grid_nodes(X1D, shape1D) == (x,)
    @test MOT.get_grid_nodes(X2D, shape2D) == (x,y)
    @test MOT.get_grid_nodes(X3D, shape3D) == (x,y,z)
end

@testset ExtendedTestSet "pad_extrapolate" begin
    # Mutating version
    x = [3,2,4.]
    y = [4,3,2,4,6.]
    y0 = zeros(5)
    MOT.pad_extrapolate!(y0, x)
    @test y0 == y
    @test y0 == MOT.pad_extrapolate(x)
    
    # Matrix version
    A = [1 2.; 
         3 4.]
    B = MOT.pad_extrapolate(A)
    @test all(B[i+1,:] == MOT.pad_extrapolate(A[i,:]) for i in 1:2)
    @test all(B[:,i] == MOT.pad_extrapolate(B[2:3,i]) for i in 1:4)

    # TODO: 3D version
end