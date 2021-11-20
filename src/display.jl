# TODO: test
function Base.show(io::IO, M::GridMeasure{D}) where D
    print(io, D,"D GridMeasure with gridshape ", M.gridshape)
end

function Base.show(io::IO, M::CloudMeasure{D}) where D
    if D == 1
        print(io, D,"D CloudMeasure with ", length(M.weights)," stored entries in the segment ")
    else
        print(io, D,"D CloudMeasure with ", length(M.weights)," stored entries in the box\n   ")
    end
    print(io, "[", M.extents[1][1], ", ", M.extents[1][2], "]")
    for i in 2:D
        print(io, " × [", M.extents[i][1], ", ", M.extents[i][2], "]")
    end
end

function Base.show(io::IO, msm::MultiScaleMeasure{M}) where {M}
    print(io, "MultiScaleMeasure with depth ",msm.depth,".\n")
    print(io, "Finest level given by ")
    show(io, msm.measures[end])
end
