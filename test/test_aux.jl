import MultiScaleOT as MOT
import LinearAlgebra: norm
import SparseArrays: spdiagm

@testset ExtendedTestSet "l1" begin
    N = 5
    a = rand(N); b = rand(N)

    # Test that it gives the correct result
    @test MOT.l1(a, b) ≈ norm(a.-b, 1)

    # Test that it gives the same result as its inbounds version
    @test MOT.l1(a, b) == MOT.l1_inbounds(a, b)

    # Test that it raises DimensionMismatch when vectors of
    # different length
    pop!(b)
    @test_throws DimensionMismatch MOT.l1(a, b)
end

@testset ExtendedTestSet "lpp" begin
    N = 5
    # Test for different ps
    for p in [1.5, 2, 3]
        a = rand(N); b = rand(N)
        # Test that it gives the correct result
        @test MOT.lpp(a, b, p) ≈ norm(a.-b, p)^p
        @test MOT.lp(a, b, p) ≈ norm(a.-b, p)

        # Test that it gives the same result as its inbounds version
        @test MOT.lpp(a, b, p) == MOT.lpp_inbounds(a, b, p)

        # Test that it raises DimensionMismatch when vectors of
        # different length
        pop!(b)
        @test_throws DimensionMismatch MOT.lpp(a, b, p)
    end
end

@testset ExtendedTestSet "l2" begin
    N = 5
    a = rand(N); b = rand(N)

    # Test that it gives the correct result
    @test MOT.l22(a, b) ≈ norm(a.-b, 2)^2
    @test MOT.l2(a, b) ≈ norm(a.-b, 2)

    # Test that it gives the same result as its inbounds version
    @test MOT.l2(a, b) == MOT.lp_inbounds(a, b, 2)

    # Test that it raises DimensionMismatch when vectors of
    # different length
    pop!(b)
    @test_throws DimensionMismatch MOT.l2(a, b)
end

@testset ExtendedTestSet ExtendedTestSet "KL" begin
    
    @test_throws DomainError MOT.KL(-1,1)
    
    a = [0.5, 0.5]
    b = [1.0, 0.0]
    
    @test MOT.KL(a, b) == Inf
    @test MOT.KL(b, a) == log(2)
end


@testset ExtendedTestSet "normalize!" begin
    N = 5
    a = rand(N) .+ 0.001 # don't allow arbitrarily low for well-posedness of the test
    b = copy(a)
    MOT.normalize!(b)

    # Test that mass is one
    @test sum(b) ≈ 1
    # Check that proportions remain the same
    r = b[1]/a[1]
    @test all(b./a .≈ r)

    # Test for fixed mass `m`
    m = 0.3
    MOT.normalize!(b, m)

    # Test that mass is one
    @test sum(b) ≈ m
    # Check that proportions remain the same
    r = b[1]/a[1]
    @test all(b./a .≈ r)
end

@testset ExtendedTestSet "euclidean-barycenter" begin
    X = [
        0 1 0 1
        0 0 1 1
    ]
    w = [0,0,1,0]; MOT.normalize!(w)
    @test all(MOT.euclidean_barycenter(X, w) .== [0,1])

    w = [0,0,1,1.]; MOT.normalize!(w)
    @test all(MOT.euclidean_barycenter(X, w) .== [0.5, 1])
end