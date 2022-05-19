"""
    Crisp Discretization
    Class for performing crisp discretization of the dataset.

    # Attributes
    - `numClasses::Int64`: number of classes in the dataset;
    - `data::Matrix{Float64}`: samples, shape (numSamples N, numFeatures M)
    - `labels::Array{Int64}`: class labels for each sample, shape (numSamples);
    - `continous::Array{Bool}`: True for each continous feature, False for each
            categorical feature, shape (numFeatures);
    - `minNumExamples::Int64`: minimum number of examples per node;
    - `minGain::Float64`: minimum entropy gain per spit.
"""

mutable struct CrispMDLFilter
    numClasses::Int64
    data::Matrix{Float64}
    labels::Array{Int64}
    continuous::Array{Bool}
    minNumExamples::Int64
    minGain::Float64
    candidateSplits::Array{Array{Float64}, 1}
    cutPoints::Array{Array{Float64}, 1}
    N::Int64
    M::Int64
    histograms::Array{Array{Float64}, 1}
end

CrispMDLFilter() = CrispMDLFilter(0, Array{Float64, 2}(undef, 1, 1),
                                Array{Int64, 1}(undef, 1),
                                Array{Bool, 1}(undef, 1),
                                0, 0.0, [], [], 0, 0, [])

function __init__(self::CrispMDLFilter, numClasses::Int64, data::Matrix{Float64},
    labels::Array{Int64}, continuous::Array{Bool}, minNumExamples::Int64=2, minGain::Float64=0.01)

    self.numClasses = numClasses
    self.data = data
    self.labels = labels
    self.continuous = continuous
    self.minNumExamples = minNumExamples
    self.minGain = minGain

    return self
end

function run(self::CrispMDLFilter) 
    """
    # Arguments

    # Returns
    Array of arrays of candidate splits.
    """

    self.candidateSplits = __findCandidateSplits(self, self.data)
    self.cutPoints = __findBestSplits(self, self.data)
    return self.cutPoints
end

function __findCandidateSplits(self::CrispMDLFilter, data::Matrix{Float64})
    """
    Finds candidate splits on the dataset.

    In the simplest case, the candidate splits are the unique elements in the
    sorted feature array.
    """

    self.N, self.M = size(data)
    return [unique!(sort!(data[:,k])) for k in range(1, self.M)]
end

function __findBestSplits(self::CrispMDLFilter, data::Matrix{Float64})
    """
    Finds best splits among the data
    """

    # Define histogram vectors
    for k in range(1, self.M)
        push!(self.histograms, zeros(((length(self.candidateSplits[k]) + 1) * self.numClasses)))
    end

    return splits
end

function __simpleHist(self::CrispMDLFilter, e::Float64, fIndex::Int64)
    """
    Simple binary histogram function.

    # Arguments
    - `e::Float64`: point to locate in the histogram
    - `fIndex::Int64`: feature index

    # Returns
    - Index of e in the histogram of feature fIndex.
    """

    ind = bisect.bisect_left(self.candidateSplits[fIndex], e)
    if ind < 0
        return -ind - 1
    end

    return ind
end

function createCrispDiscretizer(numClasses::Int64, data::Matrix{Float64}, labels::Array{Int64},
    continuous::Array{Bool}, minNumExamples::Int64=2, minGain::Float64=0.01)
    return __init__(CrispMDLFilter(), numClasses, data, labels, continuous, minNumExamples, minGain)
end

function runCrispDiscretizer(self::CrispMDLFilter)
    return run(self)
end