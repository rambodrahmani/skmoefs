"""
Crisp Discretization
"""
using Logging
logger = logging.getLogger("CrispMDLFilter")

mutable struct CrispMDLFilter
  numClasses::Int64
  continuous::Array{Bool}
  minNumExamples::Int64
  minGain::Float64
  candidateSplits::Nothing
  cutPoints::Nothing
  data::Matrix{Float64}
  label::Array{Int64}
  N::Int64
  M::Int64
end

CrispMDLFilter() = CrispMDLFilter()

function __init__(self::CrispMDLFilter, numClasses::Int64,  data::Matrix{Float64}, 
    label::Array{Int64}, continuous::Array{Bool}, minNumExamples::Int64 = 2, 
    minGain::Float64 = 0.01)

    """Class for performing crisp discretization of the dataset.

    Attributes
    ----------
    numClasses: int
        Number of classes in the dataset.
    data: np.array, shape (numSamples, numFeatures)
        N samples, M features
    label: np.array, shape (numSamples)
        Class label for each sample.
    continous: list, shape(numFeatures)
        True for each continous feature, False for each categorical feature.
    minNumExamples: int
        Minimum number of examples per node.
    minGain: float
        Minimum entropy gain per spit.

    """
    self.numClasses = numClasses
    self.continous = continous
    self.minNumExamples = minNumExamples
    self.data = data
    self.label = label
    self.minGain = minGain
    self.candidateSplits = isnothing
    self.cutPoints = isnothing
    
end

function run(self::CrispMDLFilter) 
    """
    Parameters
    ----------

    Returns
    -------
     Array of arrays of candidate splits
    """

    self.candidateSplits = self.__findCandidateSplits(self.data)
    self.cutPoints = self.__findBestSplits(self.data)

    return self.cutPoints
end

function __findCandidateSplits(self::CrispMDLFilter, data::Matrix{Float64})
    """Find candidate splits on the dataset.

    .. note:
        In the simplest case, the candidate splits are the unique elements
        in the sorted feature array.
    """
    
    self.N, self.M = size(data)
    return unique!(sort!(data[:,k])) for k in range(self.M)
end

function findBestSplits(self::CrispMDLFilter, data::Matrix{Float64})
    """
    Find best splits among the data.
    """
    
    # Define histogram vectors
    logger.debug("BUILDING HISTOGRAMS...")
    self.histograms = []
    for k in range(self.M)
        self.histograms.push!(zeros([(length(self.candidateSplits[k]) + 1) *
                              self.numClasses], dtype::int))
    
    # Iterate among features

    # Histograms built
    
end