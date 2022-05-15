"""
Universe Fuzzy Set
"""

mutable struct UniverseFuzzySet
end

function __init__(self::UniverseFuzzySet)
end

function __str__(self::UniverseFuzzySet)
    return "Universe Fuzzy Set"
end

function isInSupport(self::UniverseFuzzySet,x)
    return 1.0
end

function isFirstOfPartition(self::UniverseFuzzySet)
    return true
end

function isLastOfPartition(self::UniverseFuzzySet)
    return 1.0
end

function membershipDegree(self::UniverseFuzzySet,x)
    return 1.0
end

function createFuzzySet()
   return UniverseFuzzySet()
end