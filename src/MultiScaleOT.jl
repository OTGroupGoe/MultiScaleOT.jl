module MultiScaleOT

import Base: copy, ==, show, getindex, setindex, firstindex, lastindex, length
using SparseArrays
import Interpolations
import StructArrays: StructArray
import LinearAlgebra: dot, norm, mul!, normalize!

include("aux.jl")
export l1, l2, l22, lp, lpp, KL

include("cells.jl")
export get_cells

include("measure.jl")
export GridMeasure, CloudMeasure

include("scores.jl")
export primal_score_dense,
    dual_score_dense,
    primal_score_sparse,
    dual_score_sparse,
    PD_gap_dense,
    PD_gap_sparse

include("grid-utils.jl")
export flat_grid, get_grid_nodes

include("multiscale.jl")
export compute_multiscale_depth, 
    MultiScaleMeasure,
    refine_dual

include("parameters.jl")
export template_schedule, 
    scaling_schedule, 
    make_schedule

include("sinkhorn.jl")
export sinkhorn!, 
    sinkhorn_stabilized!,
    logsumexp,
    logsumexp!,
    log_sinkhorn!

include("hierarchical-sinkhorn.jl")
export hierarchical_sinkhorn

include("display.jl")

end # module
