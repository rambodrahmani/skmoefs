"""
Trapezoidal Fuzzy Set
"""

import Base.show

mutable struct TrapezoidalFuzzySet
    a::Float64
    b::Float64
    c::Float64
    trpzPrm::Float64
    __leftPlateau::Float64
    __rightPlateau::Float64
    __leftSlope::Float64
    __rightSlope::Float64
    left::Float64
    right::Float64
    index::Int64
end

TrapezoidalFuzzySet() = TrapezoidalFuzzySet(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

function __init__(self::TrapezoidalFuzzySet, a::Float64, b::Float64, c::Float64, trpzPrm::Float64, index::Int64=nothing)
    self.a = a
    self.b = b
    self.c = c
    self.trpzPrm = trpzPrm
    self.__leftPlateau = (b - a)*self.trpzPrm
    self.__rightPlateau = (c - b)*self.trpzPrm
    self.__leftSlope = (b - a)*(1-2*self.trpzPrm)
    self.__rightSlope = (c - b)*(1-2*self.trpzPrm)
    self.left = self.a
    self.right = self.b
    if !isnothing(index)
        self.index = index
    end

    return self
end

show(io::IO, self::TrapezoidalFuzzySet) = print(io,
    "a=$(self.a), b=$(self.b), c=$(self.c)"
)

function isInSupport(self::TrapezoidalFuzzySet, x::Float64)
    if self.a == -Inf
        return x < self.c - self.__rightPlateau
    end

    if self.c == Inf
        return x > self.a + self.__leftPlateau
    end
    
    return (x > self.a + self.__leftPlateau && x < self.c - self.__rightPlateau)
end

function membershipDegree(self::TrapezoidalFuzzySet, x::Float64)
    if isInSupport(self, x)
        if (x <= self.b && self.a == -Inf) || (x >= self.b && self.c == Inf)
            return 1.0
        elseif (x  <= self.b - self.__leftPlateau)
            uAi = (x - self.a - self.__leftPlateau) / self.__leftSlope
        elseif (x >= self.b - self.__leftPlateau) && (xi <= self.b + self.__rightPlateau)
            uAi = 1.0
        elseif (xi <= self.c-self.__rightPlateau)
            uAi = 1.0 - ((xi - self.b- self.__rightPlateau) / self.__rightSlope)
        else
            uAi = 0
        end

        return uAi
    else
        return 0.0
    end
end

function isFirstofPartition(self::TrapezoidalFuzzySet)
    return self.a == -Inf
end

function isLastOfPartition(self::TrapezoidalFuzzySet)
    return self.c == Inf
end

function createTrapezoidalFuzzySet(params::Array{Float64}, trpzPrm::Float64)
    @assert length(params) == 3 "Trapezoidal Fuzzy Set Builder requires four parameters (left, peak_b, "*
                                "peak_c and right),"*
                                "but $(length(params)) values have been provided."
    sortedParameters = sort(params)
    return __init__(TrapezoidalFuzzySet(), sortedParameters[1], sortedParameters[2],
                sortedParameters[3], trpzPrm, 0)
end

function createTrapezoidalFuzzySets(params::Array{Float64}, trpzPrm::Float64, isStrongPartition::Bool=false)
    @assert length(params) > 1 "Fuzzy Set Builder requires at least two points, " *
                               "but $(length(params)) values have been provided."
    sortedPoints = sort(params)
    if isStrongPartition
        return createTrapezoidalFuzzySetsFromStrongPartition(sortedPoints, trpzPrm)
    else
        return createTrapezoidalFuzzySetsFromNoStrongPartition(sortedPoints, trpzPrm)
    end
end

function createTrapezoidalFuzzySetsFromStrongPartition(points::Array{Float64}, trpzPrm::Float64)
    fuzzySets = []
    push!(fuzzySets, __init__(TrapezoidalFuzzySet(), -Inf, points[1], points[2], trpzPrm, 0))

    for index in range(2, length(points)-2)
        push!(fuzzySets, __init__(TrapezoidalFuzzySet(), points[index - 1], points[index], points[index + 1], trpzPrm, index))
    end

    push!(fuzzySets, __init__(TrapezoidalFuzzySet(), points[end - 1], points[end], Inf, trpzPrm, length(points)-1))
    return fuzzySets
end

function createTrapezoidalFuzzySetsFromNoStrongPartition(points::Array{Float64}, trpzPrm::Float64)
    @assert (length(points)-4)%3 == 0 " Fuzzy Set Builder requires a multiple of three plus 4 " *
                                      "as valid number of points, but $(length(points)) points have been provided."
    fuzzySets = []
    push!(fuzzySets, __init__(TrapezoidalFuzzySet(), -Inf, points[1], points[2], trpzPrm, 0))

    for index in range(2, length(points)-2)
        indexPoints = index*3
        push!(fuzzySets, __init__(TrapezoidalFuzzySet(), points[indexPoints - 1], points[indexPoints], points[indexPoints + 1], trpzPrm, 0))
    end

    push!(fuzzySets, __init__(TrapezoidalFuzzySet(), points[end - 1], points[end], Inf, trpzPrm, 0))
    return fuzzySets
end