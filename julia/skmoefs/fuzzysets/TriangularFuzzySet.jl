"""
Triangular Fuzzy Set
"""

include("../utils.jl")
import Base.show

mutable struct TriangularFuzzySet
    a::Float64
    b::Float64
    c::Float64
    __leftSupportWidth::Float64
    __rightSupportWidth::Float64
    left::Float64
    right::Float64
    index::Int64
end

TriangularFuzzySet() = TriangularFuzzySet(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

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
    if isInSupport(self, x)
        if (x <= self.b && self.a == -Inf) || (x >= self.b && self.c == Inf)
            return 1.0
         elseif x  <= self.b
             uAi = (x - self.a) / self__leftSupportWidth
         else 
            uAi = 1.0 - ((x - self.b) / self.__rightSupportWidth)
        end
        return uAi
    else
        return 0.0
    end
end

function isFirstOfPartition(self::TriangularFuzzySet)
    return self.a == -Inf
end

function isLastOfPartition(self::TriangularFuzzySet)
    return self.c == Inf
end

function createTriangularFuzzySet(params::Array{Float64})
    @assert length(params) == 3 "Fuzzy Set Builder requires three parameters " *
                            "(left, peak and rigth), but $(length(params)) values " *
                            "have been provided."
    sortedParameters = sort(params)
    return __init__(TriangularFuzzySet(), sortedParameters[0], sortedParameters[1],
                    sortedParameters[2])
end

function createTriangularFuzzySets(params::Array{Float64}, isStrongPartition::Bool=false)
    @assert length(params) > 1 "Fuzzy Set Builder requires at least two points, " *
                               "but $(length(params)) values have been provided."
    sortedPoints = sort(params)
    if isStrongPartition
        return createFuzzySetsFromStrongPartition(sortedPoints)
    else
        return createFuzzySetsFromNoStrongPartition(sortedPoints)
    end
end

function createTriangularFuzzySetsFromStrongPartition(points::Array{Float64})
    fuzzySets = []
    push!(fuzzySets, __init__(TriangularFuzzySet(), -Inf, points[1], points[2], 0))

    for index in range(1, length(points)-2)
        push!(fuzzySets, __init__(TriangularFuzzySet(), points[index], points[index + 1], points[index + 2], index))
    end

    push!(fuzzySets, __init__(TriangularFuzzySet(), points[end - 1], points[end], Inf, length(points) - 1))

    return fuzzySets
end

function createTriangularFuzzySetsFromNoStrongPartition(points::Array{Float64})
    @assert (length(points)-4)%3 == 0, "Triangular Fuzzy Set Builder requires a multiple of three plus 4 " \
                                      "as valid number of points, but %d points have been provided."% len(points)
                
    fuzzySets = []
    push!(fuzzySets, __init__(TriangularFuzzySet(), -Inf, points[1], points[2]))

    for index in range(1, length(points)-2)
        indexPoints = index*3
        push!(fuzzySets, __init__(TriangularFuzzySet(), points[indexPoints], points[indexPoints + 1], points[indexPoints + 2]))
    end

    push!(fuzzySets, __init__(TriangularFuzzySet(), points[end - 1], points[end], Inf))

    return fuzzySets
end

test1 = [0.02777778, 0.27083333, 0.51388889, 0.75694444, 1.        ]
test2 = [0.08333333 0.29166667 0.5        0.70833333 0.91666667]
test3 = [0.   0.25 0.5  0.75 1.  ]
test4 = [0.   0.25 0.5  0.75 1.  ]
fuzzySets = createTriangularFuzzySets(test1, true)
for fuzzySet in fuzzySets
    println(fuzzySet)
end