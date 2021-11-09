# CODE STATUS: REVISED, TESTED

# lp distances

"""
    l1_inbounds(a, b)

Compute ℓ^1 distance between `a` and `b`. Does not allocate nor
perform bound checks on the vectors
"""
function l1_inbounds(a, b)
    s = 0.0
    for i in eachindex(a)
        @inbounds s += abs(a[i] - b[i])
    end
    return s
end

"""
    lpp_inbounds(a, b, p)

Compute p-th power of the ℓ^p distance between `a` and `b`.
Does not allocate nor perform bounds checks on the vectors
"""
function lpp_inbounds(a, b, p)
    s = 0.0
    for i in eachindex(a)
        @inbounds s += abs(a[i] - b[i])^p
    end
    return s
end

"""
    lp_inbounds(a, b, p)

Compute the ℓ^p distance between `a` and `b`.
Does not allocate nor perform bounds checks on the vectors
"""
lp_inbounds(a, b, p) = lpp_inbounds(a, b, p)^(1/p)

"""
    l1(a, b)

Compute ℓ^1 distance between `a` and `b`. Does not allocate. If the sizes of the
vectors are known with certainty, `l1_inbounds` might be preferred.
"""
function l1(a, b)
    (length(a) == length(b)) || throw(DimensionMismatch("lengths of a and b does not match"))
    l1_inbounds(a, b)
end

"""
    lpp(a, b, p)

Compute the p-th power of the ℓ^p distance between `a` and `b`. Does not allocate.
If the sizes of the vectors are known with certainty, `lpp_inbounds` might be
preferred.
"""
function lpp(a, b, p)
    (length(a) == length(b)) || throw(DimensionMismatch("lengths of a and b does not match"))
    lpp_inbounds(a, b, p)
end

"""
    lp(a, b, p)

Compute the ℓ^p distance between `a` and `b`. Does not allocate. If the sizes of
the vectors are known with certainty, `lp_inbounds` might be preferred.
"""
lp(a, b, p) = lpp(a, b, p)^(1/p)

l22(a, b) = lpp(a, b, 2)

l2(a, b) = sqrt(l22(a, b))

# KL divergence

"""
    KL(a, b)


Computes KL divergence between vectors `a` and `b`. 
Yield `Inf` if `a[i]>0, b[i]=0` for some `i`.
"""
KL(a::Real, b::Real) = (a == 0 ? b : a*log(a/b)-a+b)

KL(a, b) = sum(KL.(a, b))

# Normalization utils

"""
    normalize!(a[, mass = 1])

Normalize the vector `a` so that it adds up to `mass`.
Does not perform any check on the positivity of `a`.
"""
normalize!(a, mass=1) = (a ./= (sum(a)/mass); nothing)

""" 
    mean(x, J)

Mean of `x[J]`, non-allocating
"""
function mean(x, J)
    m = zero(eltype(x))
    for j in J
        m += x[j]
    end
    return m/length(J)
end

"""
    euclidean_barycenter(X, w)

Compute the euclidean barycenter of the columns of `X`
with weights `w`
"""
euclidean_barycenter(X, w) = sum(X .* w', dims = 2)
