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
    histograms::Array{Array{Int64}, 1}
end

FuzzyMDLFilter() = FuzzyMDLFilter(0, Array{Float64, 2}(undef, 1, 1),
                                Array{Int64, 1}(undef, 1),
                                Array{Bool, 1}(undef, 1),
                                0, 0.0, 0, 0, true, "", 0.0,
                                [], [], 0, 0, [], [])

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
    self.cutPoints = findBestSplits(self, self.data)
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

function findBestSplits(self::FuzzyMDLFilter, data::Matrix{Float64})
    """
    Find best splits among the data.
    """

    @info "Building histograms."

    # Define histogram vectors
    for k in range(1, self.M)
        push!(self.histograms, zeros((length(self.candidateSplits[k]) + 1) * self.numClasses))
    end

    # Iterate among features
    for k in range(1, self.M)
        # Iterate among features
        if self.continous[k]
            for ind in range(1, self.N)
                x = simpleHist(self, data[ind][k], k)
                self.histograms[k][Int64(x * self.numClasses + self.label[ind])-2] += 1
            end
        end
    end

    # Histograms built

    splits = []
    for k in range(1, self.M)
        if self.continous[k]
            if self.threshold == 0
                indexCutPoints = calculateCutPoints(k, 0, length(self.histograms[k]) - self.numClasses)
            else
                indexCutPointsIndex = self.calculateCutPointsIndex(k, 0, length(self.histograms[k]) - self.numClasses)
                if length(indexCutPointsIndex) != 0
                    depth, points, _ =  zip(*self.traceback(indexCutPointsIndex))
                    indexCutPoints = sorted(points[:self.threshold])
                else
                    indexCutPoints = []
                end
            end
                
            if length(indexCutPoints) != 0
                cutPoints = np.zeros(length(indexCutPoints) + 2)
                cutPoints[0] = self.candidateSplits[k][0]
                
                for i in range(1, length(indexCutPoints))
                    cSplitIdx = indexCutPoints[i] /self.numClasses
                    if (cSplitIdx > 0 && cSplitIdx < length(self.candidateSplits[k]))
                        cutPoints[i+1] = self.candidateSplits[k][int(cSplitIdx)]
                    end
                end

                cutPoints[-1] = self.candidateSplits[k][-1]
            
                splits.append(cutPoints)
            else
                splits.append([])
            end
        else
            splits.append([])
        end
    end

    return splits
end

function simpleHist(self::FuzzyMDLFilter, e::Float64, fIndex::Int64)
    """
    Simple binary histogram function.

    # Arguments
    - `e::Float64`: point to locate in the histogram
    - `fIndex::Int64`: feature index

    # Returns
    - Index of e in the histogram of feature fIndex.
    """

    ind = BisectPy.bisect_left(self.candidateSplits[fIndex], e)
    if ind < 0
        return -ind - 1
    end

    return ind
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