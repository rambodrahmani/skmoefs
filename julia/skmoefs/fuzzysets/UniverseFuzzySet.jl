"""
    Universe Fuzzy Set
"""

import Base.show

mutable struct UniverseFuzzySet
end

function __init__(self::UniverseFuzzySet)
end

show(io::IO, self::UniverseFuzzySet) = print(io,
    "Universe Fuzzy Set"
)

function isInSupport(self::UniverseFuzzySet, x)
    return true
end

function isFirstOfPartition(self::UniverseFuzzySet)
    return true
end

function isLastOfPartition(self::UniverseFuzzySet)
    return true
end

function membershipDegree(self::UniverseFuzzySet,x)
    return 1.0
end

function createFuzzySet()
    return UniverseFuzzySet()
end