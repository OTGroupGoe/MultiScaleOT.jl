# CODE STATUS: REVISED, UNTESTED

function hierarchical_sinkhorn(mu::AbstractMeasure, nu::AbstractMeasure, c, params_schedule, layer0::Int = 3)
    i = layer0
    
    # Setup variables
    depth_mu = compute_multiscale_depth(mu)
    depth_nu = compute_multiscale_depth(nu)

    # Adjust to minimum depth
    depth = min(depth_mu, depth_nu)

    # Get multiscale measures
    muH = MultiScaleMeasure(mu; depth)
    nuH = MultiScaleMeasure(nu; depth)

    # Unwrap some fields for convenience
    mui = muH.measures[i]
    nui = nuH.measures[i]
    μ = mui.weights
    X = mui.points
    ν = nui.weights
    Y = nui.points
    m = length(μ)
    n = length(ν)

    # Initialize duals
    a = zeros(Float64, m)
    b = zeros(Float64, n)

    # Init K to a sparse but full matrix
    K = sparse(μ .* ν')
    colptr = K.colptr
    rowval = K.rowval
    status = -1

    k0 = findfirst(params_schedule.layer .== i)
    for k in k0:length(params_schedule)
        params = params_schedule[k]

        # Get solver parameters
        ε = params.solver_eps
        θ = params.solver_truncation
        i = params.layer
        verbose = params.solver_verbose
        
        
        # Get new kernel: 
        # TODO: this can be threaded
        K = get_stabilized_kernel(c, a, b, ε, X, Y, μ, ν, θ, colptr, rowval)
        # Solve OT
        # TODO: allow different solvers?
        status = sinkhorn_stabilized!(a, b, μ, ν, K, ε; 
                                        max_error = params.solver_max_error,
                                        max_iter = params.solver_max_iter,
                                        verbose = params.solver_verbose,
                                        max_error_rel = params.solver_max_error_rel)
        colptr = K.colptr
        rowval = K.rowval
        if verbose
            println("Layer\t", i,"\t",
                    "eps =\t", ε,"\t",
                    "status\t",status,"\t",
                    "PD gap\t", PD_gap_sparse(a, b, K, c, X, Y, μ, ν, ε))
        end

        # Refine
        if (i < depth) && (i != params_schedule.layer[k+1])
            # Save previous data
            m0 = m
            n0 = n
            # This is a shallow copy of the previous measure
            muprev = copy(mui)
            nuprev = copy(nui)

            # Get new marginals
            mui = muH.measures[i+1]
            nui = nuH.measures[i+1]

            μ = mui.weights
            X = mui.points
            ν = nui.weights
            Y = nui.points
            m = length(μ)
            n = length(ν)

            # Refine support of K
            colptr, rowval = refine_support(m0, n0, 
                                            colptr, rowval, 
                                            m, n, 
                                            muH.refinements[i], 
                                            nuH.refinements[i])

            # Refine potentials:
            nodesX = get_grid_nodes(muprev.points, muprev.gridshape)
            a = refine_dual(a, nodesX, X, muprev.gridshape)

            nodesY = get_grid_nodes(nuprev.points, nuprev.gridshape)
            b = refine_dual(b, nodesY, Y, nuprev.gridshape)
        end
    end

    return K, a, b, status 
end