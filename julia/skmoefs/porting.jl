"""
    Utility methods.
    This file contains the implementation of Python methods for which a
    corresponding Julia method is not available.
"""

using Base.Broadcast

"""
The only meaningful way in which a scalar can be defined in julia, is of the
behavior of broadcast. Broadcast.DefaultArrayStyle{N}() is a BroadcastStyle
indicating that an object behaves as an N-dimensional array for broadcasting.
"""
isscalar(x::T) where T = isscalar(T)
isscalar(::Type{T}) where T = BroadcastStyle(T) isa Broadcast.DefaultArrayStyle{0}

"""
Replaces NaN with zero and infinity with largest finite numbers.
"""
nan_to_num(v::Float64) = map(x -> isnan(x) ? zero(x) : isinf(x) ? prevfloat(typemax(Float64)) : x, v)