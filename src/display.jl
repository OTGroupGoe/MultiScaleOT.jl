# TODO: test
function Base.show(io::IO, M::GridMeasure{D}) where D
    print(io, D,"D GridMeasure with gridshape ", M.gridshape)
end

function Base.show(io::IO, M::CloudMeasure{D}) where D
    print(io, D,"D CloudMeasure with ", length(M.weights)," stored entries in the box ")
    print("\n   [", M.extents[1][1], ", ", M.extents[1][2], "]")
    for i in 2:D
        print("\n × [", M.extents[i][1], ", ", M.extents[i][2], "]")
    end
end