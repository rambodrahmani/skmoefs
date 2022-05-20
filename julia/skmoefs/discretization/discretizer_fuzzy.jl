"""
    Fuzzy Discretization
    Class for performing fuzzy discretization of the dataset.

    # Attributes
    - `numClasses::Int64`: number of classes in the dataset;
    - `data::Matrix{Float64}`: samples, shape (numSamples N, numFeatures M)
    - `labels::Array{Int64}`: class labels for each sample, shape (numSamples);
    - `continuous::Array{Bool}`: True for each continuous feature, False for each
            categorical feature, shape (numFeatures);
    - `minNumExamples::Int64`: minimum number of examples per node;
    - `minGain::Float64`: minimum entropy gain per spit;
    - `numBins::Int64`: number of bins for the discretization step, default=500.
"""

mutable struct FuzzyMDLFilter
    numClasses::Int64
    data::Matrix{Float64}
    labels::Array{Int64}
    continuous::Array{Bool}
    minNumExamples::Int64
    minGain::Float64
    threshold::Int64
    numBins::Int64
    ignore::Bool
    ftype::String
    trpzPrm::Float64
    candidateSplits::Array{Array{Float64}, 1}
    cutPoints::Array{Array{Float64}, 1}
    N::Int64
    M::Int64
    initEntr::Array{Float64}
end

FuzzyMDLFilter() = FuzzyMDLFilter(0, Array{Float64, 2}(undef, 1, 1),
                                Array{Int64, 1}(undef, 1),
                                Array{Bool, 1}(undef, 1),
                                0, 0.0, 0, 0, true, "", 0.0,
                                [], [], 0, 0, [])

function __init__(self::FuzzyMDLFilter, numClasses::Int64, data::Matrix{Float64},
    labels::Array{Int64}, continuous::Array{Bool}, minImpurity::Float64=0.02,
    minGain::Float64=0.000001, threshold::Int64=0, numBins::Int64=500,
    ignore::Bool=true, ftype::String="triangular", trpzPrm::Float64=0.1)
    
    self.numClasses = numClasses
    self.data = data
    self.labels = labels
    self.continuous = continuous
    self.minNumExamples = Int64(minImpurity*size(data)[1])
    self.minGain = minGain
    self.threshold = threshold
    self.numBins = numBins
    self.ignore = ignore
    self.trpzPrm = trpzPrm
    self.ftype = ftype

    return self
end

function run(self::FuzzyMDLFilter)
    """
    # Arguments

    # Returns
    List of arrays of candidate splits.
    """

    if self.ftype == "triangular"
        @info "Running Discretization with fuzzyset " * self.ftype * "."
    end

    if self.ftype == "trapezoidal"
        @info "nRunning Discretization with fuzzyset " * self.ftype * " and trpzPrm = " * self.trpzPrm * "."
    end

    self.candidateSplits = findCandidateSplits(self, self.data)
    self.initEntr = zeros(size(self.data)[2])
    self.cutPoints = findBestSplits(self.data)
    #if sum([len(k) for k in self.cutPoints]) == 0
        #logger.warning('Empty cut points for all the features!')
    #end

    return self.cutPoints
end

function findCandidateSplits(self::FuzzyMDLFilter, data::Matrix{Float64})
    """
    Finds candidate splits on the dataset.

    In the simplest case, the candidate splits are the unique elements in the
    sorted feature array.
    """

    self.N, self.M = size(data)
    numBins = self.numBins

    vector = []
    for k in range(1, self.M)
        uniques = unique!(sort(data[:,k]))
        if length(uniques) > numBins
            append!(vector, uniques[LinRange(0, length(uniques)-1, num_bins)])
        else
            append!(vector, [uniques])
        end
    end

    return vector
end

function createFuzzyMDLDiscretizer(numClasses::Int64, data::Matrix{Float64},
        labels::Array{Int64}, continuous::Array{Bool}, minImpurity::Float64=0.02,
        minGain::Float64=0.000001, threshold::Int64=0, num_bins::Int64=500,
        ignore::Bool=true, ftype::String="triangular", trpzPrm::Float64=0.1)
    return __init__(FuzzyMDLFilter(), numClasses, data, labels, continuous, minImpurity,
            minGain, threshold, num_bins, ignore, ftype, trpzPrm)
end

function runFuzzyMDLDiscretizer(self::FuzzyMDLFilter)
    return run(self)
end