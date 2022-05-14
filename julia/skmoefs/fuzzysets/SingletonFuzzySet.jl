"""
Fuzzy Singleton
Fuzzy set whose support is a single point in universe of discourse.
"""

mutable struct SingletonFuzzySet
    value::Float64
    left::Float64
    right::Float64
end

function __init__(self::SingletonFuzzySet, value::Float64, index::Int64=None)
    self.value = value
    self.left = self.value
    self.right = self.value
    if isnothing(index)
        self.index = index
    end
end

show(io::IO, self::SingletonFuzzySet) = print(io,
    "value = $(self.value)"
)

function isInSupport(self::SingletonFuzzySet, x::Float64)
    return x == self.value
end

function isFirstOfPartition(self::SingletonFuzzySet)
    return self.value == -Inf
end

function isLastOfPartition(self::SingletonFuzzySet)
    return self.value == Inf
end

function membershipDegree(self::SingletonFuzzySet, x::Float64)
    if isInSupport(self, x)
        return 1.0
    else
        return 0.0
    end
end
