"""
Fuzzy Singleton
Fuzzy set whose support is a single point in universe of discourse.
"""

include("../utils.jl")
import Base.show

mutable struct TrapezoidalFuzzySet
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    __leftSupportWidth::Float64
    __rightSupportWidth::Float64
    index::Int64
end

TrapezoidalFuzzySet() = TrapezoidalFuzzySet(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

function __init__(self::TrapezoidalFuzzySet, a::Float64, b::Float64, c::Float64, d::Float64, index::Int64=nothing)
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.__leftSupportWidth = b - a
    self.__rightSupportWidth = d - c
    if !isnothing(index)
        self.index = index
    end

    return self
end