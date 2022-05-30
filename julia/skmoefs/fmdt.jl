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

function calculateFuzzyImpurity(counts::Array{Float64}, totalCount::Float64)
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

function findcontinous(X::Matrix{Float64})
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

mutable struct DecisionNode
    """
    Decision Node.
    """
    feature::Int64
    fSet::Any
    isLeaf::Bool
    results::Array{Float64}
    weight::Float64
    child::Any
end

DecisionNode() = DecisionNode(0, nothing, false, [], 0.0, nothing)

function __init__(self::DecisionNode, feature::Int64, fSet::Any, isLeaf::Bool,
            results::Array{Float64}, weight::Float64, child::Any=nothing)
    """
    # Arguments:
     - `feature`: feature corresponding to the set on the node
            it is not the feature used for splitting the child
    """
    self.feature = feature
    self.fSet = fSet
    self.isLeaf = isLeaf
    self.results = results
    self.child = child
    self.weight = weight

    return self
end

function createDecisionNode(feature::Int64, fSet::Any, isLeaf::Bool,
    results::Array{Float64}, weight::Float64, child::Any=nothing)
    return __init__(DecisionNode(), feature, fSet, isLeaf, results, weight, child)
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
    dom_class::Int64
end

FMDT() = FMDT(5, 0.02, 0.01, 0.01, 2, 1.0, true, 0, true, "all", 0, [], 0, 0, [], [], 0)

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
        self.cont = findcontinous(X)
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

    tree = buildtree(self, hcat(X, reshape(y, length(y), 1)), 0, createUniverseFuzzySet())
    self.tree = tree

    return self
end

function buildtree(self::FMDT, rows::Matrix{Float64}, depth::Int64=0, fSet::UniverseFuzzySet=nothing,
                    scoref::Function=calculateFuzzyImpurity, memb_degree::Any=nothing,
                    leftAttributes::Any=nothing, feature::Any=-1)
    # Attributes are "consumed" on a given path from root to leaf,
    # Please note: [] != None
    if isnothing(leftAttributes)
        leftAttributes = collect(1:self.M)
    end

    # Membership degree of the given set of samples for the current node
    if isnothing(memb_degree)
        memb_degree = ones(length(rows))
    end

    # Set up some variables to track the best criteria
    best_gain = 0.0
    best_criteria = nothing
    best_sets = nothing

    # Class counts on the current node
    class_counts = classCounts(self, rows, memb_degree)
    if depth == 0
        self.dom_class = findmax(class_counts)[2]
    end

    # Stop splitting if
    # - the proportion of dataset of a class k is greater than a threshold max_prop
    # - the cardinality of the dataset in the node is lower then a threshold self.min_num_examples
    # - there are no attributes left    
    if (!isempty(findall(>(0), (class_counts / Float64(sum(class_counts))) .>= self.max_prop))) ||
        sum(class_counts) < self.min_num_examples || isempty(leftAttributes)
        return createDecisionNode(feature, fSet, true, class_counts / Float64(sum(class_counts)),
                                sum(memb_degree))
    end

    # Calculate entropy for the node
    current_score = scoref(class_counts, sum(class_counts))

    # Iterate among features
    gain = best_gain
    for col in leftAttributes
        if !isempty(self.fSets[col])
            row_vect, memb_vect = __multidivide(self, rows, memb_degree, col, self.fSets[col])
        end
    end
end

function classCounts(self::FMDT, rows::Matrix{Float64}, memb_degree::Array{Float64})
    """
    Returns the sum of the membership degree for each class in a given set.
    """
    labels = rows[:, end]
    numClasses = self.K
    llab = sort(unique(labels))
    counts = zeros(numClasses)
    for k in range(1, length(llab))
        ind = Int64(llab[k]+1)
        counts[ind] = sum(memb_degree[findall(==(llab[k]), labels)])
    end

    return counts
end

function __multidivide(self::FMDT, rows::Matrix{Float64}, membership::Array{Float64},
                    column::Int64, fSets::Any)
    memb_vect = []
    row_vect = []
    for fSet in fSets
        mask = findall(>(0), map((x) -> isInSupport(fSet, x), rows[:,column]))
        append!(row_vect, [rows[mask,:]])
        append!(memb_vect, [tNorm(self, map((x) -> membershipDegree(fSet, x), rows[mask, column]), membership[mask], "product")])
    end

    return row_vect, memb_vect
end

function tNorm(elf::FMDT, array1::Array{Float64}, array2::Array{Float64}, tnorm::String="product")
    """
    Method for calculating various elemntwise t_norms.

    # Arguments:
    - `array1`, numpy.array()
        First array
    array2, numpy.array()
        Second array
    """
    if tnorm == "product"
        return array1 .* array2
    elseif tnorm == "min"
        return map((x, y) -> min(x, y), array1, array2)
    end
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