import MultiScaleOT as MOT

@testset ExtendedTestSet "get_cells" begin
    # Cells 1D: we consider a sequence of 8 points
    # `[1, 2, 3, 4, 5, 6, 7, 8]` 
    # that must be divided according to some sizes and offsets.
    cells, cells_shape = MOT.get_cells_1D(8, 1, 0)
    @test cells == [[i] for i in 1:8]
    @test cells_shape == (8,)

    cells, cells_shape = MOT.get_cells_1D(8, 2, 0)
    @test cells == [[1, 2], [3, 4], [5, 6], [7, 8]]
    @test cells_shape == (4,)

    cells, cells_shape = MOT.get_cells_1D(8, 2, 1)
    @test cells == [[1], [2, 3], [4, 5], [6, 7], [8]]
    @test cells_shape == (5,)

    cells, cells_shape = MOT.get_cells_1D(8, 3, 2)
    @test cells == [[1, 2], [3, 4, 5], [6, 7, 8]]
    @test cells_shape == (3,)

    # Cells 2D: we consider a grid of 3x4 points
    # `[ 1  4  7  10
    #    2  5  8  11
    #    3  6  9  12]`
    # that must be divided according to some sizes and offsets.
    # That way we can check it works well for uneven and 
    # anisotropic data.
    shape2D = (3, 4)
    cells, cells_shape = MOT.get_cells_2D(shape2D, 1, 0)
    @test cells == [[i] for i in 1:prod(shape2D)]
    @test cells_shape == (3, 4)

    cells, cells_shape = MOT.get_cells_2D(shape2D, 2, 0)
    @test cells == [[1, 2, 4, 5], [3, 6], 
                    [7, 8, 10, 11], [9, 12]]
    @test cells_shape == (2, 2)

    cells, cells_shape = MOT.get_cells_2D(shape2D, 2, 1)
    @test cells == [[1], [2, 3], 
                    [4, 7], [5, 6, 8, 9],
                    [10], [11, 12]]
    @test cells_shape == (2, 3)

    # Cells 3D: we consider a grid of 3×3×3 points: 
    # 
    # [:, :, 1] =
    # 1  4  7
    # 2  5  8
    # 3  6  9
    #
    # [:, :, 2] =
    # 10  13  16
    # 11  14  17
    # 12  15  18
    #
    # [:, :, 3] =
    # 19  22  25
    # 20  23  26
    # 21  24  27
    
    # Dividing this grid into cells with cellsize 2 and offsets 0
    # or 1 should yield:
    shape3D = (3, 3, 3)
    cells_shape = (2, 2, 2)
    cells_offset_0 = [
        [1, 2, 4, 5, 10, 11, 13, 14], 
        [3, 6, 12, 15], 
        [7, 8, 16, 17], 
        [9, 18], 
        [19, 20, 22, 23], 
        [21, 24], 
        [25, 26], 
        [27]
    ]
    cells_offset_1 = [
        [1], 
        [2, 3], 
        [4, 7], 
        [5, 6, 8, 9], 
        [10, 19], 
        [11, 12, 20, 21], 
        [13, 16, 22, 25], 
        [14, 15, 17, 18, 23, 24, 26, 27]
    ]
    @test MOT.get_cells_3D(shape3D, 2, 0) == (cells_offset_0, cells_shape)
    @test MOT.get_cells_3D(shape3D, 2, 1) == (cells_offset_1, cells_shape)

    # General `get_cells` without specified dimension
    @test MOT.get_cells((4,), 2, 0) == MOT.get_cells_1D(4, 2, 0)
    @test MOT.get_cells(shape2D, 2, 0) == MOT.get_cells_2D(shape2D, 2, 0)
    @test MOT.get_cells(shape3D, 2, 0) == MOT.get_cells_3D(shape3D, 2, 0)
    @test_throws ErrorException("not implemented for dimension > 3.") MOT.get_cells((2, 2, 2, 2), 2, 0)
end