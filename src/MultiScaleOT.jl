module MultiScaleOT
using SparseArrays

include("aux.jl")

include("cells.jl")

include("grid-utils.jl")

include("measure.jl")

include("multiscale.jl")

include("parameters.jl")

include("sinkhorn.jl")

include("scores.jl")

include("hierarchical-sinkhorn.jl")

end # module
