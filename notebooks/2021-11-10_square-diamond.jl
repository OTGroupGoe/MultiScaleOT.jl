# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Julia 1.6.1
#     language: julia
#     name: julia-1.6
# ---

import MultiScaleOT as MOT
using HDF5
using Plots
# Some plots defaults
default(legend = :none, aspect_ratio = :equal)
using SparseArrays

# +
# Load data
img1, img2, shape1, shape2 = h5open("data/square_diamond.hdf5", "r") do file
    (
        read(file, "img1"), read(file, "img2"), 
        read(file, "shape1"), read(file, "shape2")
    )
end

plot(heatmap(img1), heatmap(img2), size = (600, 300))

# +
# Build measures:
# Weights
mu1 = img1[:] .+ 1e-8
mu2 = img2[:] .+ 1e-8

MOT.normalize!(mu1)
MOT.normalize!(mu2)

# Supporting points
x1 = collect(1:128)
X = MOT.flat_grid(x1, x1)
Y = copy(X)

# Gridshapes
shapeX = size(img1)
shapeY = size(img2)

# Measure struct
mu = MOT.GridMeasure(X, mu1, shapeX)
nu = MOT.GridMeasure(Y, mu2, shapeY)

# +
# Prepare parameters for the solver and multiscale
depth = MOT.compute_multiscale_depth(mu)

c(x,y) = MOT.l22(x,y)

# Epsilon schedule
Nsteps = 3
factor = 2.
eps_target = 0.5
last_iter = [eps_target/2]

# Epsilon scaling
eps_schedule = MOT.scaling_schedule(depth, eps_target, Nsteps, factor; last_iter = last_iter)

layer_schedule = MOT.template_schedule(depth, Nsteps, collect(1:depth); last_iter = [depth])

truncation = 1e-15

params_schedule = MOT.make_schedule(
                layer = layer_schedule,
                solver_eps = eps_schedule, 
                solver_truncation = truncation,
                solver_max_error = 1e-4,
                solver_verbose = true,
                solver_max_iter = 10000
        );
# -

# Solve OT problem hierarchically
@time K, a, b, status = MOT.hierarchical_sinkhorn(mu, nu, c, params_schedule, 2)

# Visualize displacement interpolation
function displacement_interpolation(P::SparseMatrixCSC, X, Y, shapeX, t)
    (0 ≤ t ≤ 1) || erorr("t must be in [0,1]")
    # Z = Matrix{Float64}(undef, size(X, 1), length(P.nzval))
    Z = zeros(shapeX...)
    for j in 1:size(P, 2)
        for r in P.colptr[j]:P.colptr[j+1]-1
            i = P.rowval[r]
            k1 = (1-t).*X[1,i] .+ t.*Y[1,j]
            k2 = (1-t).*X[2,i] .+ t.*Y[2,j]
            Z[Int(round(k1)), Int(round(k2))] += P.nzval[r]
        end
    end
    return Z
end

plots = []
for t in 0:0.2:1
    Z = displacement_interpolation(K, X, Y, shape1, t)
    plot_i = heatmap(Z, axis = :off)
    xticks!(Int[])
    yticks!(Int[])
    push!(plots, plot_i)
end
plot(plots..., layout = (1,6), size = (900,150))


