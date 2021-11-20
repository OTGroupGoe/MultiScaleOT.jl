# TODO: test
function Base.show(io::IO, M::GridMeasure{D}) where D
    print(io, D,"D GridMeasure with gridshape ", M.gridshape)
end

function Base.show(io::IO, M::CloudMeasure{D}) where D
    print(io, D,"D CloudMeasure with ", length(M.weights)," stored entries in the box ")
    print("[", M.extents[1][1], ", ", M.extents[1][2], "]")
    for i in 2:D
        print("\n × [", M.extents[i][1], ", ", M.extents[i][2], "]")
    end
end

function Base.show(io::IO, msm::MultiScaleMeasure{M}) where {M}
    print(io, "MultiScaleMeasure with depth ",msm.depth,".\n")
    print(io, "Finest level given by ")
    show(io, msm.measures[end])
end
