"""
Triangular Fuzzy Set
"""

mutable struct TriangularFuzzySet
    a::Float64
    b::Float64
    c::Float64
    index::Int64
end

TriangularFuzzySet() = TriangularFuzzySet(0.0, 0.0, 0.0, 0)

function __init__(self::TriangularFuzzySet, a::Float64, b::Float64, c::Float64, index::Int64=nothing)
    self.a = a
    self.b = b
    self.c = c
    self.__leftSupportWidth = b-a
    self.__rightSupportWidth = c-b
    self.left = self.a
    self.right = self.b
    if !isnothing(index)
        self.index = index
    end

    return self
end

function isInSupport(self::TriangularFuzzySet, x::Float64)
    return x > self.a && x < self.c    
end


function membershipDegree(self::TriangularFuzzySet, x::Float64)
    if !isInSupport(self, x)
        return 0.0
    end
    
    isInLowerRange = x <= self.b && self.a == -inf
    isInUpperRange = x >= self.b && self.c == inf
    if isInLowerRange || isInUpperRange
        return 1.0
    end

    valForLeftSupportWidth = (x - self.a) / self.__leftSupportWidth
    valForRightSupportWidth = 1.0 - ((x - self.b) / self.__rightSupportWidth)
    return uAi = xi <= self.b ? valForLeftSupportWidth : valForRightSupportWidth
end


function isFirstOfPartition(self::TriangularFuzzySet)
    return self.a == -Inf
end

function isLastOfPartition(self::TriangularFuzzySet)
    return self.c == Inf
end