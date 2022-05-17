"""
"""

mutable struct fuzzyDiscretization
    method::String
    numSet::Int64
end

fuzzyDiscretization() = fuzzyDiscretization("uniform", 7)

function __init__(self::fuzzyDiscretization, method::String="uniform", numSet::Int64=7)
    @assert mathod in ["uniform", "equifreq"] "Invalid discretization method."
    self.method = method
    @assert numSet >= 3 "Invalid "
end