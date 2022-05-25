"""
    Fuzzy Decision Tree for generating the initial set of rules.
"""

include("fuzzysets/UniverseFuzzySet.jl")
include("fuzzysets/SingletonFuzzySet.jl")
include("fuzzysets/TriangularFuzzySet.jl")
include("fuzzysets/TrapezoidalFuzzySet.jl")
include("discretization/discretizer_base.jl")

mutable struct FuzzyImpurity
end

function calculate(counts::Array{Float64}, totalCount::Float64)
    """
    Evaluates fuzzy impurity, given a list of fuzzy cardinalities.

    # Arguments
    - `counts::Array{Float64}`: an array containing the fuzzy cardinality for a fuzzy set for each class;
    - `totalCount::Float64`: global cardinality for a fuzzy set;

    # Returns
    - `fuzzy_impurity::Float64`: fuzzy impurity evaluated on the given set.
    """
    if totalCount == 0
        return 0.0
    end
    numClasses = length(counts)
    impurity = 0.0
    classIndex = 1

    while (classIndex <= numClasses)
        classCount = counts[classIndex]
        if classCount != 0
            freq = classCount / totalCount
            impurity -= freq * log2(freq)
        end
        classIndex += 1
    end
    return impurity
end