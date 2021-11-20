using MultiScaleOT 

@testset ExtendedTestSet "display" begin
    io = IOBuffer()
    # GridMeasure
    mu = GridMeasure([1. 1], [1., 1], (2,))
    show(io, mu)
    @test String(take!(io)) == "1D GridMeasure with gridshape (2,)"

    mu = GridMeasure([1. 2; 1 1], [1.,1], (2,1))
    show(io, mu)
    @test String(take!(io)) == "2D GridMeasure with gridshape (2, 1)"
    
    # CloudMeasure
    nu = CloudMeasure([1. 2], [1,1.])
    show(io, nu)
    @test String(take!(io)) == "1D CloudMeasure with 2 stored entries in the segment [1.0, 2.0]"

    nu = CloudMeasure([1 2; 1 1.], [1,1.])
    show(io, nu)
    @test String(take!(io)) == "2D CloudMeasure with 2 stored entries in the box\n   [1.0, 2.0] × [1.0, 1.0]"

    # MultiScaleMeasure
    muH = MultiScaleMeasure(mu)
    show(io, muH)
    @test String(take!(io)) == "MultiScaleMeasure with depth 2.\nFinest level given by 2D GridMeasure with gridshape (2, 1)"
    
    # TODO: support MultiScaleMeasure{CloudMeasure} when they are defined
end
    
