"""
Trapezoidal Fuzzy Set
"""

include("../utils.jl")
import Base.show
using Ranges

mutable struct __TrapezoidalFuzzySet
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    __leftSupportWidth::Float64
    __rightSupportWidth::Float64
    index::Int64
end

__TrapezoidalFuzzySet() = __TrapezoidalFuzzySet(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

function __init__(self::__TrapezoidalFuzzySet, a::Float64, b::Float64, c::Float64, d::Float64, index::Int64=nothing)
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.__leftSupportWidth = b - a
    self.__rightSupportWidth = d - c
    self.left = self.a
    self.right = self.b
    if !isnothing(index)
        self.index = index
    end

    return self
end

show(io::IO, self::__TrapezoidalFuzzySet) = print(io,
    "a=$(self.a), b=$(self.b), c=$(self.c), d=$(self.d)"
)

function isInSupport(self::__TrapezoidalFuzzySet, x::Float64)
    return x > self.a && x < self.d
end

function membershipDegree(self::__TrapezoidalFuzzySet, x::Float64)
    if self.isInSupport(x)
        if (x <= self.b && self.a == -Inf) || (x >= self.b && self.c == Inf)
            return 1.0
        else
            if (x <= self.b)
                uAi = (x - self.a) / self.__leftSupportWidth
            elseif (x >= self.c)
                uAi = 1.0 - ((x - self.c) / self.__rightSupportWidth)
            else
                uAi = 1.0
            return uAi
            end
        end
    else
        return 0.0
    end
end

function isFirstofPartition(self::__TrapezoidalFuzzySet)
    return self.a == -Inf
end

function isLastOfPartition(self::__TrapezoidalFuzzySet)
    return self.c == Inf
end

function createTrapezoidalFuzzySet(params::Array{Float64})
    @assert length(params) == 4 "Trapezoidal Fuzzy Set Builder requires four parameters (left, peak_b, "*
                            "peak_c and right),"*
                            "but $(length(params)) values have been provided."
    sortedParameters = sort(params)
    return __init__(__TrapezoidalFuzzySet(), sortedParameters[1], sortedParameters[2],
                sortedParameters[3], sortedParameters[4])
end

function createTrapezoidalFuzzySets(params::Array{Float64}, isStrongPartition::Bool=false)
        @assert length(params) > 3 "Triangular Fuzzy Set Builder requires at least four points,"*
                                "but $(length(params)) values have been provided."
        sortedPoints = sort(params)
    
    if isStrongPartition
        return create_TrapezoidalFuzzySetsFromStrongPartition(sortedPoints)
    else
        return create_TrapezoidalFuzzySetsFromNoStrongPartition(sortedPoints)
    end
end

function createTrapezoidalFuzzySetsFromStrongPartition(points::Array{Float64})
    fuzzySets = []
    push!(fuzzySets, __init__(__TrapezoidalFuzzySet(), -Inf, points[1], points[2], points[3], 0))
    fCount = 1

    for index in range(2, length(points)-3, 2)
        push!(fuzzySets, __init__(__TrapezoidalFuzzySet(), points[index], points[index + 1], points[index + 2], points[index + 3], index ÷ 2))
        fCount += 1
    end

    push!(fuzzySets, __init__(__TrapezoidalFuzzySet(), points[end - 2], points[end - 1], points[end], Inf, fCount))

    return fuzzySets
end

function createTrapezoidalFuzzySetsFromNoStrongPartition(points::Array{Float64})
        @assert  (len(points)-6) % 4 == 0, "Triangular Fuzzy Set Builder requires a multiple of four plus 6 "*
                                           "as valid number of points, but $(length(params)) values have been provided."
            
    fuzzySets = []
    push!(fuzzySets, __init__(__TrapezoidalFuzzySet(), -Inf, points[1], points[2], points[3]))

    for index in range(2, length(points)-3)
    indexPoints = index*4
    push!(fuzzySets, __init__(__TrapezoidalFuzzySet(), points[indexPoints], points[indexPoints + 1], points[indexPoints + 2], points[indexPoints+3] ))
    end

    push!(fuzzySets, __init__(__TrapezoidalFuzzySet(), points[end - 2], points[end - 2], points[end], Inf))

    return fuzzySets
end

mutable struct TrapezoidalFuzzySet
    a::Float64
    b::Float64
    c::Float64
    trpzPrm::Float64
    __leftPlateau::Float64
    __rightPlateau::Float64
    __leftSlope::Float64
    __rightSlope::Float64
    index::Int64
end

TrapezoidalFuzzySet() = TrapezoidalFuzzySet(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

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
    return __init__(TrapezoidalFuzzySet(), sortedParameters[0], sortedParameters[1],
                sortedParameters[2], trpzPrm)
end

function createTrapezoidalFuzzySets(params::Array{Float64}, trpzPrm::Float64, isStrongPartition::Bool=false)
        @assert length(params) > 1,  " Fuzzy Set Builder requires at least two points,"*
                                     "but $(length(params)) values have been provided."
        if isStrongPartition
             return createTrapezoidalFuzzySetsFromStrongPartition(sortedPoints, trpzPrm)
        else
             return createTrapezoidalFuzzySetsFromNoStrongPartition(sortedPoints, trpzPrm)
        end
end

function createTrapezoidalFuzzySetsFromStrongPartition(points::Array{Float64}, trpzPrm::Float64)
    fuzzySets = []
    push!(fuzzySets, __init__(TrapezoidalFuzzySet(), -Inf, points[1], points[2], 0, trpzPrm))

    for index in range(1, length(points)-2)
        push!(fuzzySets, __init__(TrapezoidalFuzzySet(), points[index], points[index + 1], points[index + 2], trpzPrm, index ÷ 2))
    end

    push!(fuzzySets, __init__(TrapezoidalFuzzySet(), points[end - 1], points[end], Inf, trpzPrm, index))

    return fuzzySets
end

function createTrapezoidalFuzzySetsFromNoStrongPartition(points::Array{Float64}, trpzPrm::Float64)
    @assert  (len(points)-4) % 3 == 0, "Triangular Fuzzy Set Builder requires a multiple of four plus 6 "*
                                        "as valid number of points, but $(length(params)) values have been provided."

    fuzzySets = []
    push!(fuzzySets, __init__(TrapezoidalFuzzySet(), -Inf, points[1], points[2], trpzPrm))

    for index in range(1, length(points)-2)
        indexPoints = index*3
        push!(fuzzySets, __init__(TrapezoidalFuzzySet(), points[indexPoints], points[indexPoints + 1], points[indexPoints + 2], trpzPrm))
    end

    push!(fuzzySets, __init__(TrapezoidalFuzzySet(), points[end - 1], points[end], Inf, trpzPrm))

    return fuzzySets
end