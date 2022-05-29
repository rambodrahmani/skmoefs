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
    K::Int64
    cont::Array{Bool}
    N::Int64
    M::Int64
    fSets::Array{Any}
    cPoints::Array{Array{Float64}, 1}
end

FMDT() = FMDT(5, 0.02, 0.01, 0.01, 2, 1.0, true, 0, true, "all", 0, [], 0, 0, [], [])

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

function fit(self::FMDT, X::Matrix{Float64}, y::Vector{Int64}, continuous::Array{Bool}=nothing,
                cPoints::Array{Array{Float64}}=nothing, numClasses::Any=nothing,
                ftype::Any="triangular", trpzPrm::Any=-1.0)
    """
    Build a multi-way fuzzy decision tree classifier from the training set (X, y).

    # Arguments
    - `X::Matrix{Float64}`: shape = [n_samples, n_features]
        The training input samples. Internally, it will be converted to
        ``dtype=np.float32`` and if a sparse matrix is provided
        to a sparse ``csc_matrix``.

    - `y::Vector{Int64}`: shape = [n_samples] or [n_samples, n_outputs]
        The target values (class labels)

    - `numClasses::Int64`: Number of classes for the dataset.

    - `cPoints::Array{Array{Float64}}`: Array of array of cut points.

    # Returns
    -------
    - `self::FMDT`
    """
    @assert ftype in ["triangular", "trapezoidal"] "Invalid fuzzy set type: $(ftype)"
    if !isnothing(numClasses)
        self.K = numClasses
    else
        self.K = Int64(maximum(y) + 1)
    end

    if isnothing(continuous)
        self.cont = findContinous(X)
    else
        self.cont = continuous
    end

    self.N, self.M = size(X)

    if self.prior_discretization
        if !isnothing(cPoints)
            self.cPoints = cPoints
        else
            fuzzy_mdlf_discretizer = createFuzzyMDLDiscretizer(self.K, X, y, self.cont,
                                        self.discr_minImpurity, self.discr_minGain, self.discr_threshold)
            self.cPoints = runFuzzyMDLDiscretizer(fuzzy_mdlf_discretizer)
        end
    end

    self.fSets = []

    for (k,points) in enumerate(self.cPoints)
        if !continuous[k] == true
            if self.cPoints[k]
                points = self.cPoints[k]
            else
                points = unique(X[:,k])
                self.cPoints[k] = points
            end
            append!(self.fSets, [createSingletonFuzzySets(points)])
        elseif length(points) == 0
            append!(self.fSets, [[]])
        else
            if ftype == "triangular"
                append!(self.fSets, [createTriangularFuzzySets(points, true)])
            elseif ftype == "trapezoidal"
                append!(self.fSets, [createTrapezoidalFuzzySets(points, trpzPrm, true)])
            end
        end
    end

    println(self.fSets)
end

function findContinous(X::Matrix{Float64})
    """
    # Arguments
    - `X::Matrix{Float64}`: shape (Nsample, Nfeatures)

    # Returns
    - A list containing True if the corresponding element is regarded as continous,
      False otherwise
    """
    N, M = size(X)
    red_axis = [length(unique(X[:, k])) for k in range(1, M)] / Float64(N)
    return red_axis .> 0.05
end

function createFMDT(max_depth::Int64=5, discr_minImpurity::Float64=0.02,
        discr_minGain::Float64=0.01, minGain::Float64=0.01, min_num_examples::Int64=2,
        max_prop::Float64=1.0, prior_discretization::Bool=true,
        discr_threshold::Int64=0, verbose::Bool=true, features::String="all")
    return __init__(FMDT(), max_depth, discr_minImpurity, discr_minGain, minGain,
    min_num_examples, max_prop, prior_discretization, discr_threshold, verbose, features)
end

function fitFMDTTree(self::FMDT, X::Matrix{Float64}, y::Vector{Int64}, continuous::Array{Bool}=nothing,
                cPoints::Array{Array{Float64}}=nothing, numClasses::Any=nothing,
                ftype::Any="triangular", trpzPrm::Any=-1.0)
    fit(self, X, y, continuous, cPoints, numClasses, ftype, trpzPrm)
end