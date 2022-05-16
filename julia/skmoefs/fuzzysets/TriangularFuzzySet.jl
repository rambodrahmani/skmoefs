"""
Triangular Fuzzy Set
"""

include("../utils.jl")

mutable struct TriangularFuzzySet
    a::Float64
    b::Float64
    c::Float64
    __leftSupportWidth::Float64
    __rightSupportWidth::Float64
    index::Int64
end

TriangularFuzzySet() = TriangularFuzzySet(0.0, 0.0, 0.0, 0.0, 0.0, 0)

function __init__(self::TriangularFuzzySet, a::Float64, b::Float64, c::Float64, index::Int64=nothing)
    self.a = a
    self.b = b
    self.c = c
    self.__leftSupportWidth = b - a
    self.__rightSupportWidth = c - b
    self.left = self.a
    self.right = self.b
    if !isnothing(index)
        self.index = index
    end

    return self
end

show(io::IO, self::TriangularFuzzySet) = print(io,
    "a=$(self.a), b=$(self.b), c=$(self.c)"
)

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

function createFuzzySet(params::Array{Float64})
    @assert length(params) == 3 "Fuzzy Set Builder requires three parameters " *
                            "(left, peak and rigth), but $(length(params)) values " *
                            "have been provided."
sortedParameters = sort(params)
return __init__(TriangularFuzzySet(), sortedParameters[0], sortedParameters[1],
                            sortedParameters[2])
end

function createFuzzySets(params::Array{Float64}, isStrongPartition::Bool=false)
    @assert length(params) > 1 "Fuzzy Set Builder requires at least two points, " *
                               "but $(length(params)) values have been provided."
    sortedPoints = sort(params)
    if isStrongPartition
        return createFuzzySetsFromStrongPartition(sortedPoints)
    else
        return createFuzzySetsFromNoStrongPartition(sortedPoints)
    end
end

function createFuzzySetsFromStrongPartition(points::Array{Float64})
    fuzzySets = []
    fuzzySets.append!(TriangularFuzzySets(-Inf, points[0], points[1], index = 0))

    for index in range(1,length(points)-1)
        fuzzySets.append!(TriangularFuzzySets(points[index - 1], points[index], points[index + 1], index))   
        fuzzySets.append(TriangularFuzzySets(points[-2], points[-1], inf, length(points)-1))
    end
    return fuzzySets
end

function createFuzzySetsFromNoStrongPartition(points::Array{Float64})
    @assert (length(points)-4)%3 == 0, "Triangular Fuzzy Set Builder requires a multiple of three plus 4 " \
                                      "as valid number of points, but %d points have been provided."% len(points)
                
    fuzzySets = []
    fuzzySets.append!(TriangularFuzzySets(-Inf, points[0], points[1]))

    for index in range(1,length(points)-1)
        indexPoints = index*3
        fuzzySets.append!(TriangularFuzzySets(points[indexPoints - 1], points[indexPoints], points[indexPoints + 1])) 
        fuzzySets.append!(TriangularFuzzySets(points[-2], points[-1], inf))
    end
    return fuzzySets
end


