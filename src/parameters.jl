
"""
    DEFAULT_PARAMETERS

Default parameters for the multiscale routines
"""
const DEFAULT_PARAMETERS = (;
    epsilon = 1.0,
    solver_max_error = 1e-6,
    solver_max_error_rel = true,
    solver_max_iter = 10000,
    solver_verbose = true,
    solver_truncation = 1e-15
) 

"""
    constant_schedule(value, N)

Repeat `value` `N` times
"""
constant_schedule(value, N::Int) = fill(value, N)

"""
    template_schedule(depth, template, factors; last_iter = [])

Generate the vector
`[factors[1].*template; ... ; factors[depth].*template; last_iter]`
Useful to define a schedule with a certain substructure. 
When `factors` is a singleton, it is transformed into `fill(factors, depth)`.
When `template` is an `Int`, it is transformed into `ones(template)`.

# Examples
```julia
julia> depth = 3; Ns = [1, 2, 4]; Nsteps = 2;
julia> template_schedule(depth, Nsteps, Ns)
6-element Vector{Int64}:
 1
 1
 2
 2
 4
 4

julia> depth = 3; maxiters_template = [2000, 1000]; factor = 1;
julia> template_schedule(depth, maxiters_template, factor)
6-element Vector{Int64}:
 2000
 1000
 2000
 1000
 2000
 1000
```
"""
function template_schedule(depth::Int, template::Vector{T}, factors::Vector{T}; last_iter = T[]) where T
    length(factors) == depth || error("length of factors must match depth")
    value_schedule = repeat(template, depth)
    L = length(template)
    for i in eachindex(factors)
        value_schedule[(i-1)*L+1:i*L] .*= factors[i]
    end
       
    # Add last iterations
    for last_value_i in last_iter
        push!(value_schedule, last_value_i)
    end
    return value_schedule
end

function template_schedule(depth::Int, template::Vector{T}, factor::T; last_iter = T[]) where T
    template_schedule(depth, template, fill(factor, depth); last_iter)
end

function template_schedule(depth, Nsteps::Int, factors::Vector{T}; last_iter= T[]) where T
    template_schedule(depth, ones(T, Nsteps), factors; last_iter)
end


"""
    scaling_schedule(depth, target_value, Nsteps, factor; last_iter = [])

Generate the vector
```julia
target_value.* [factor^(depth*(Nsteps-1)), 
                factor^(depth*(Nsteps-1)-1),
                factor^(depth*(Nsteps-1)-2),
                ...
                factor^((depth-1)*(Nsteps-1)),
                factor^((depth-1)*(Nsteps-1)),
                factor^((depth-1)*(Nsteps-1)-1),
                ...
                factor^((depth-2)*(Nsteps-1)),
                factor^((depth-2)*(Nsteps-1)),
                ...
                ...
                1
                ;
                last_iter]
```
Useful for generating scaling schedules for the regularization `ε`.

# Examples
```julia 
julia> eps_target = 1.; factor = 2.; Nsteps = 3;
julia> eps_schedule = scaling_schedule(depth, eps_target, Nsteps, factor)
9-element Vector{Float64}:
 64.0
 32.0
 16.0
 16.0
  8.0
  4.0
  4.0
  2.0
  1.0
```
"""
function scaling_schedule(depth::Int, target_value::T, Nsteps::Int, factor::T; last_iter = T[]) where T
    value::T = target_value * factor^(depth*(Nsteps-1))
    value_schedule = T[]
    for i in 1:depth
        for j in 1:Nsteps
            push!(value_schedule, value/factor^(j-1))
        end
        value = value/factor^(Nsteps-1)
    end
    
    # Add last iterations
    for last_value_i in last_iter
        push!(value_schedule, last_value_i)
    end
    return value_schedule
end

function default_eps_schedule(depth::Int, target_eps; Nsteps = 3, factor = 2., last_iter = Float64[])
    eps_schedule = scaling_schedule(depth, Float64(target_eps), Nsteps, Float64(factor); last_iter)
    layer_schedule = template_schedule(depth, fill(1, Nsteps), collect(1:depth); last_iter = fill(depth, length(last_iter)))
    layer_schedule, eps_schedule
end
"""
    make_schedule(; nt...) 

Generate a `StructArray`` representing a schedule, this is,
columns represent the values a parameter, is taking on each
step of the algorithm; conversely, rows represent the set of
parameters to be used at a given stage of the algorithm.

The elements in `nt` are arrays of singletons. In the later case
they are repeated to match the length of the array ones.

# Examples
julia> sinkhorn_schedule = make_schedule(
                ε = [4,  2,  2,  1], 
                N = [64, 64, 32, 32],
                θ = 1e-20,
                max_error = 1e-6
        )

4-element StructArray(::Vector{Int64}, ::Vector{Int64}, ::Vector{Float64}, ::Vector{Float64}) with eltype NamedTuple{(:ε, :N, :θ, :max_error), Tuple{Int64, Int64, Float64, Float64}}:
(ε = 4, N = 64, θ = 1.0e-20, max_error = 1.0e-6)
(ε = 2, N = 64, θ = 1.0e-20, max_error = 1.0e-6)
(ε = 2, N = 32, θ = 1.0e-20, max_error = 1.0e-6)
(ε = 1, N = 32, θ = 1.0e-20, max_error = 1.0e-6)
"""
function make_schedule(; nt...) 
    schedule_length = -1
    # Check consistency of lengths
    needs_expansion = Symbol[]
    for k in keys(nt)
        if length(nt[k]) > 1
            if schedule_length == -1
                schedule_length = length(nt[k])
            else
                length(nt[k]) == schedule_length || error("all non-trivial schedules must have same length.")
            end
        else
            push!(needs_expansion, k)
        end
    end

    if schedule_length == -1
        error("need some non-trivial schedule")
    end

    # Overwrite singleton keys with constant expansion
    overwrite_nt = NamedTuple([k=>constant_schedule(nt[k], schedule_length) for k in needs_expansion])
    nt = (; nt..., overwrite_nt...)
    
    return StructArray(; nt...)
end

function make_multiscale_schedule(; nt...) 
    # Add from DEFAULT_PARAMETERS the ones that might be missing
    nt = (; DEFAULT_PARAMETERS..., nt...)
    make_schedule(; nt...) 
end