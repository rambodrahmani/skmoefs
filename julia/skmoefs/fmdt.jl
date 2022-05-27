"""
    Fuzzy Decision Tree for generating the initial set of rules.
"""

include("fuzzysets/UniverseFuzzySet.jl")
include("fuzzysets/SingletonFuzzySet.jl")
include("fuzzysets/TriangularFuzzySet.jl")
include("fuzzysets/TrapezoidalFuzzySet.jl")
include("discretization/discretizer_base.jl")

mutable struct FuzzyImpurity
    """
    Fuzzy Impurity.
    """
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

mutable struct FMDT
    """
    Fuzzy Decision Tree.
    """
    max_depth::Int64
    discr_minImpurity::Float64
    discr_minGain::Float64
    minGain::Float64
    min_num_examples::Int64
    max_prop::Float64
    prior_discretization::Bool
    discr_threshold::Int64
    verbose::Bool
    features::String
end

FMDT() = FMDT(5, 0.02, 0.01, 0.01, 2, 1.0, true, 0, true, "all")

function __init__(self::FMDT, max_depth::Int64=5, discr_minImpurity::Float64=0.02,
                discr_minGain::Float64=0.01, minGain::Float64=0.01, min_num_examples::Int64=2,
                max_prop::Float64=1.0, prior_discretization::Bool=true,
                discr_threshold::Int64=0, verbose::Bool=true, features::String="all")
    """
    FMDT with fuzzy discretization on the node.

    # Arguments
    - `max_depth::Int64`: optional, default = 5
        maximum tree depth;
    - `discr_minImpurity::Float64`: optional, default = 0.02
        minimum imputiry for a fuzzy set during discretization;
    - `discr_minGain::Float64`: optional, default = 0.01
        minimum entropy gain during discretization;
    - `minGain::Float64`: optional, default = 0.01
        minimum entropy gain for a split during tree induction;
    - `minNumExamples::Int64`: optional, default = 2
        minimum number of example for a node during tree induction;
    - `max_prop::Float64`: optional, default = 1.0
        min proportion of samples pertaining to the same class in the same node
        for stop splitting;
    - `prior_discretization::Bool`: optional, default = True
        define whether performing prior or node level discretization;
    - `discr_threshold::Int64`: optional, default = 0
        if discr_threshold != 0 the discretization is stopped
        at n = discr_threshold + 2 fuzzy sets.

    # Notes
    - When using the FMDT in a fuzzy random forest implementation, programmers
      are encouraged to use prior_discretization = True and providing the cPoints
      upon fit. This would help in speeding up computation.
    """
    self.max_depth = max_depth
    self.discr_minImpurity = discr_minImpurity
    self.discr_minGain = discr_minGain
    self.minGain = minGain
    self.min_num_examples = min_num_examples
    self.max_prop = max_prop
    self.prior_discretization = prior_discretization
    self.discr_threshold = discr_threshold
    self.verbose = verbose
    self.features = features

    return self
end

function createFMDT(max_depth::Int64=5, discr_minImpurity::Float64=0.02,
        discr_minGain::Float64=0.01, minGain::Float64=0.01, min_num_examples::Int64=2,
        max_prop::Float64=1.0, prior_discretization::Bool=true,
        discr_threshold::Int64=0, verbose::Bool=true, features::String="all")
    return __init__(FMDT(), max_depth, discr_minImpurity, discr_minGain, minGain,
    min_num_examples, max_prop, prior_discretization, discr_threshold, verbose, features)
end