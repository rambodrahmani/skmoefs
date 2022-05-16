"""
Fuzzy Singleton
Fuzzy set whose support is a single point in universe of discourse.
"""

include("../utils.jl")
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

function createSingletonFuzzySet(value::Float64, index::Int64=nothing)
    @assert isscalar(value) "Invalid parameter for Singleton Fuzzy Set!"
    return __init__(SingletonFuzzySet(), value, index)
end

function createSingletonFuzzySets(points::Array{Float64}, isStrongPartition::Bool=false)
    return [createSingletonFuzzySet(point[1], point[0]) for point in enumerate(points)]
end

fuzzy_singleton_example = createSingletonFuzzySet(43.2, 3)
println(fuzzy_singleton_example)

fuzzy_singletons_example = createSingletonFuzzySets([43.2, 24.3])
println(fuzzy_singleton_example)