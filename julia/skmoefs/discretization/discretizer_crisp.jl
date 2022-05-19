"""
    Crisp Discretization
    Class for performing crisp discretization of the dataset.

    # Attributes
    - `numClasses::Int64`: number of classes in the dataset;
    - `data::Matrix{Float64}`: samples, shape (numSamples N, numFeatures M)
    - `labels::Array{Int64}`: class labels for each sample, shape (numSamples);
    - `continuous::Array{Bool}`: True for each continuous feature, False for each
            categorical feature, shape (numFeatures);
    - `minNumExamples::Int64`: minimum number of examples per node;
    - `minGain::Float64`: minimum entropy gain per spit.
"""

using BisectPy

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
    histograms::Array{Array{Int64}, 1}
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

    @info "Building histograms."

    # Define histogram vectors
    for k in range(1, self.M)
        push!(self.histograms, zeros(((length(self.candidateSplits[k]) + 1) * self.numClasses)))
    end

    # Iterate among features
    for k in range(1, self.M)
        # Iterate among features
        if self.continuous[k]
            for ind in range(1, self.N)
                x = __simpleHist(self, data[ind,k], k)
                self.histograms[k][Int64(x * self.numClasses + self.labels[ind])-2] += 1
            end
        end
    end
    # Histograms built

    splits = []
    for k in range(1, self.M)
        indexCutPoints = __calculateCutPoints(self, k, 0, length(self.histograms[k]) - self.numClasses)
        if length(indexCutPoints != 0)
            cutPoints = zeros(length(indexCutPoints))

            for i in range(1, length(indexCutPoints))
                cSplitIdx = Int64(indexCutPoints[i] / self.numClasses)
                if (cSplitIdx > 0 && cSplitIdx < length(self.candidateSplits[k]))
                    cutPoints[i] = self.candidateSplits[k][cSplitIdx]
                end
            end

            append!(splits, cutPoints)
        else
            append!(splits, [])
        end
    end

    return convert(Array{Array{Float64}, 1}, splits)
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

    ind = BisectPy.bisect_left(self.candidateSplits[fIndex], e)
    if ind < 0
        return -ind - 1
    end

    return ind
end

function __calculateCutPoints(self::CrispMDLFilter, fIndex::Int64, first::Int64, lastPlusOne::Int64)
    """
    Main iterator.
    """

    counts = zeros(2, self.numClasses)
    counts[1, :] = evalCounts(self.histograms[fIndex], self.numClasses, first, lastPlusOne)
    totalCounts = sum(counts[1, :])

    if totalCounts < self.minNumExamples
        return []
    end

    priorCounts = copy(counts[1, :])
    priorEntropy = entropy(priorCounts, totalCounts)

    bestEntropy = priorEntropy

    bestCounts = zeros(2, self.numClasses)
    bestIndex = -1
    leftNumInstances = 0
    currSplitIndex = first

    while (currSplitIndex < lastPlusOne)
        if (leftNumInstances > self.minNumExamples && (totalCounts - leftNumInstances) > self.minNumExamples)
            leftImpurity = entropy(counts[1, :], leftNumInstances)
            rightImpurity = entropy(counts[2, :], totalCounts - leftNumInstances)
            leftWeight = float(leftNumInstances) / totalCounts
            rightWeight = float(totalCounts - leftNumInstances) / totalCounts
            currentEntropy = leftWeight * leftImpurity + rightWeight * rightImpurity

            if currentEntropy < bestEntropy
                bestEntropy = currentEntropy
                bestIndex = currSplitIndex
                bestCounts = copy(counts)
            end
        end

        for currClassIndex in range(1, self.numClasses)
            leftNumInstances += self.histograms[fIndex][currSplitIndex + currClassIndex]
            counts[1, currClassIndex] += self.histograms[fIndex][currSplitIndex + currClassIndex]
            counts[2, currClassIndex] -= self.histograms[fIndex][currSplitIndex + currClassIndex]
        end

        currSplitIndex += self.numClasses
    end

    gain = priorEntropy - bestEntropy
    if (gain < self.minGain || !self.__mdlStopCondition(gain, priorEntropy, bestCounts, entropy))
        logger.debug("Feature %d index %d, gain %f REJECTED" % (fIndex, bestIndex, gain))
        return np.array([])
    end
    logger.debug("Feature %d index %d, gain %f ACCEPTED" % (fIndex, bestIndex, gain))

    left = self.__calculateCutPoints(fIndex, first, bestIndex)
    right = self.__calculateCutPoints(fIndex, bestIndex, lastPlusOne)

    indexCutPoints = np.zeros((length(left) + 1 + length(right)))

    for k in range(length(left))
        indexCutPoints[k] = left[k]
    end
    indexCutPoints[length(left)] = bestIndex
    for k in range(length(right))
        indexCutPoints[length(left) + 1 + k] = right[k]
    end

    return indexCutPoints
end

function evalCounts(hist::Array{Int64}, numClasses::Int64, first::Int64, lastPlusOne::Int64)
    """
    Number of counts.
    """

    counts = zeros(numClasses)
    index = first
    while (index < lastPlusOne)
        for k in range(1, length(counts))
            counts[k] += hist[index + k]
        end
        index += numClasses
    end
    return counts
end

function entropy(counts::Array{Float64}, totalCount)
    """
    Evaluate entropy.
    """

    if totalCount == 0
        return 0
    end

    numClasses = length(counts)
    impurity = 0.0
    classIndex = 1
    while (classIndex <= numClasses)
        classCount = counts[classIndex]
        if classCount != 0
            freq = classCount / totalCount
            if freq != 0
                impurity -= freq * log2(freq)
            end
        end
        classIndex += 1
    end

    return impurity
end

function createCrispMDLFDiscretizer(numClasses::Int64, data::Matrix{Float64}, labels::Array{Int64},
    continuous::Array{Bool}, minNumExamples::Int64=2, minGain::Float64=0.01)
    return __init__(CrispMDLFilter(), numClasses, data, labels, continuous, minNumExamples, minGain)
end

function runCrispMDLFDiscretizer(self::CrispMDLFilter)
    return run(self)
end