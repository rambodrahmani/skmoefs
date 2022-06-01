"""
    Fuzzy Singleton
    Fuzzy set which support is a single point in universe of discourse.
"""

include("../porting.jl")
import Base.show

mutable struct SingletonFuzzySet
    value::Float64
    left::Float64
    right::Float64
    index::Int64
end

SingletonFuzzySet() = SingletonFuzzySet(0.0, 0.0, 0.0, 0)

function __init__(self::SingletonFuzzySet, value::Float64, index::Int64=nothing)
    self.value = value
    self.left = self.value
    self.right = self.value
    if !isnothing(index)
        self.index = index
    end

    return self
end

show(io::IO, self::SingletonFuzzySet) = print(io,
    "value = $(self.value)"
)

function membershipDegree(self::SingletonFuzzySet, x::Float64)
    if isInSupport(self, x)
        return 1.0
    else
        return 0.0
    end
end

function isInSupport(self::SingletonFuzzySet, x::Float64)
    return x == self.value
end

function isFirstOfPartition(self::SingletonFuzzySet)
    return self.value == -Inf
end

function isLastOfPartition(self::SingletonFuzzySet)
    return self.value == Inf
end

function createSingletonFuzzySet(value::Float64, index::Int64=nothing)
    @assert isscalar(value) "Invalid parameter for Singleton Fuzzy Set!"
    return __init__(SingletonFuzzySet(), value, index)
end

function createSingletonFuzzySets(points::Array{Float64}, isStrongPartition::Bool=false)
    return [createSingletonFuzzySet(point[2], point[1]) for point in enumerate(points)]
end