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
        if self.continuous[k]
            for ind in range(1, self.N)
                x = simpleHist(self, data[ind, k], k)
                self.histograms[k][Int64(x * self.numClasses + self.labels[ind])-2] += 1
            end
        end
    end

    # Histograms built

    splits = []
    for k in range(1, self.M)
        if self.continuous[k]
            if self.threshold == 0
                indexCutPoints = calculateCutPoints(self, k, 0, length(self.histograms[k]) - self.numClasses)
            else
                indexCutPointsIndex = calculateCutPointsIndex(k, 0, length(self.histograms[k]) - self.numClasses)
                if length(indexCutPointsIndex) != 0
                    depth, points, _ =  zip(traceback(indexCutPointsIndex))
                    indexCutPoints = sorted(points[:self.threshold])
                else
                    indexCutPoints = []
                end
            end

            if length(indexCutPoints) != 0
                cutPoints = zeros(length(indexCutPoints) + 2)
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

function calculateCutPoints(self::FuzzyMDLFilter, fIndex::Int64, first::Int64,
                            lastPlusOne::Int64, depth::Int64=0)
    """
    Main iterator
    """

    s = sum(evalCounts(self.histograms[fIndex], self.numClasses, first, lastPlusOne))

    # Evaluating prior cardinality
    if self.ftype == "trapezoidal"
        s0s1 = calculatePriorTrapezoidalCardinality(self.histograms[fIndex],
                                                self.candidateSplits[fIndex],
                                                self.numClasses, first, lastPlusOne,self.trpzPrm)
    elseif self.ftype == "triangular"
        s0s1 = calculatePriorTriangularCardinality(self.histograms[fIndex],
                                                self.candidateSplits[fIndex],
                                                self.numClasses, first, lastPlusOne)
    end

    wPriorFuzzyEntropy = calculateWeightedFuzzyImpurity(s0s1, s, entropy)
    s0s1s2 = zeros(3, self.numClasses)
    bestEntropy = wPriorFuzzyEntropy
    bestS0S1S2 = np.zeros_like(s0s1s2)
    bestIndex = -1
    currentEntropy = 0.0
    leftNumInstances = 0
    currSplitIndex = first
    currClassIndex = 0

    while (currSplitIndex < lastPlusOne)
        if (leftNumInstances > self.minNumExamples && (s-leftNumInstances) > self.minNumExamples)
            if self.ftype == "trapezoidal"
                s0s1s2 = calculateNewTrapezoidalCardinality(self.histograms[fIndex], 
                                                            self.candidateSplits[fIndex],
                                                            currSplitIndex/self.numClasses,
                                                            self.numClasses, first, lastPlusOne, self.trpzPrm)
            elseif self.ftype == "triangular"
                s0s1s2 = calculateNewTriangularCardinality(self.histograms[fIndex], 
                                                            self.candidateSplits[fIndex],
                                                            currSplitIndex/self.numClasses,
                                                            self.numClasses, first, lastPlusOne)
            end

            currentEntropy = calculateWeightedFuzzyImpurity(s0s1s2, s, entropy)

            if currentEntropy < bestEntropy
                bestEntropy = currentEntropy
                bestIndex = currSplitIndex
                bestS0S1S2 = copy(s0s1s2)
            end
        end

        for currClassIndex in range(0, self.numClasses)
            leftNumInstances += self.histograms[fIndex][currSplitIndex + currClassIndex]
        end

        currSplitIndex += self.numClasses
    end

    gain = wPriorFuzzyEntropy - bestEntropy

    if (gain < self.minGain || !mdlStopCondition(gain, wPriorFuzzyEntropy, bestS0S1S2, entropy, None))
        @info "Feature $(fIndex) index $(bestIndex), gain $(gain) REJECTED"
        return []
    end
    @info "Feature $(fIndex) index $(bestIndex), gain $(gain) ACCEPTED"

    left = calculateCutPoints(self, fIndex, first, bestIndex, depth+1)
    right = calculateCutPoints(self, fIndex, bestIndex, lastPlusOne, depth+1)

    indexCutPoints = zeros(length(left)+1 + length(right))

    for k in range(1, length(left))
        indexCutPoints[k] = left[k]
    end
    indexCutPoints[length(left)] = bestIndex
    for k in range(1, length(right))
        indexCutPoints[len(left)+1+k] = right[k]
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

function calculatePriorTrapezoidalCardinality(hist::Array{Int64}, candidateSplits::Array{Float64},
                                            numClasses::Int64, first::Int64, lastPlusOne::Int64,
                                            trpzPrm::Float64=0.2)
    """
    Prior trapezoidal cardinality.
    """
    s0s1 = zeros(Float64, 2, numClasses)
    minimum = candidateSplits[first ÷ numClasses]
    diff = candidateSplits[lastPlusOne ÷ numClasses - 1] - minimum
    plateau = diff*trpzPrm
    slope = diff*(1-2*trpzPrm)
    uS1 = 0.0
    xi = 0.0
    for i in range(Int64(first / numClasses)+1, Int64(lastPlusOne / numClasses))
        xi = candidateSplits[i]
        uS1 = 0.0
        if slope != 0.
             uS1 = (xi-minimum-plateau)/slope
             uS1 = maximum(0, uS1)
             uS1 = minimum(1, uS1)
        end
        for j in range(1, numClasses)
            s0s1[1,j] += hist[i * numClasses + j] * (1 - uS1)
            s0s1[2,j] += hist[i * numClasses + j] * uS1
        end
    end

    return s0s1
end

function calculatePriorTriangularCardinality(hist::Array{Int64}, candidateSplits::Array{Float64},
                                            numClasses::Int64, first::Int64, lastPlusOne::Int64)
    """
    Prior triangular cardinality.
    """

    s0s1 = zeros(Float64, 2, numClasses)
    minimum = candidateSplits[(first ÷ numClasses)+1]
    diff = candidateSplits[lastPlusOne ÷ numClasses - 1] - minimum
    uS1 = 0.0
    xi = 0.0
    for i in range(Int64(first / numClasses)+1, Int64(lastPlusOne / numClasses))
        xi = candidateSplits[i]
        uS1 = 0.0
        if diff != 0.0
            uS1 = (xi - minimum) / diff
        end
        for j in range(1, numClasses)
            println(hist[i * numClasses + j] * (1 - uS1))
            s0s1[1,j] += hist[i * numClasses + j] * (1 - uS1)
            s0s1[2,j] += hist[i * numClasses + j] * uS1
        end
    end

    return s0s1
end

function calculateCutPointsIndex(self::FuzzyMDLFilter, fIndex::Int64, first::Int64,
                            lastPlusOne::Int64, depth::Int64=0, baseIndex::String="T")
    """
    Main iterator
    """

    s = sum(evalCounts(self.histograms[fIndex], self.numClasses, first, lastPlusOne))

    # Evaluating prior cardinality
    if self.ftype == "trapezoidal"
        s0s1 = calculatePriorTrapezoidalCardinality(self.histograms[fIndex],
                                                self.candidateSplits[fIndex],
                                                self.numClasses, first, lastPlusOne, self.trpzPrm)
    elseif self.ftype == "triangular"
        s0s1 = calculatePriorTriangularCardinality(self.histograms[fIndex],
                                                self.candidateSplits[fIndex],
                                                self.numClasses, first, lastPlusOne)
    end
    
    wPriorFuzzyEntropy = calculateWeightedFuzzyImpurity(s0s1, s, entropy)
    if depth == 0
        self.initEntr[fIndex] = wPriorFuzzyEntropy
    end

    s0s1s2 = zeros(3, self.numClasses)
    bestEntropy = wPriorFuzzyEntropy
    bestS0S1S2 = np.zeros_like(s0s1s2)
    bestIndex = -1
    currentEntropy = 0.0
    leftNumInstances = 0
    currSplitIndex = first
    currClassIndex = 0

    while(currSplitIndex < lastPlusOne)
        if (leftNumInstances > self.minNumExamples && (s-leftNumInstances) > self.minNumExamples)
            s0s1s2 = calculateNewTriangularCardinality(self.histograms[fIndex],
                                                        self.candidateSplits[fIndex],
                                                        currSplitIndex/self.numClasses,
                                                        self.numClasses, first, lastPlusOne)
            currentEntropy = calculateWeightedFuzzyImpurity(s0s1s2, s, entropy)

            if currentEntropy < bestEntropy
                bestEntropy = currentEntropy
                bestIndex = currSplitIndex
                bestS0S1S2 = copy(s0s1s2)
            end
        end

        for currClassIndex in range(0, self.numClasses)
            leftNumInstances += self.histograms[fIndex][currSplitIndex+currClassIndex]
        end
        
        currSplitIndex += self.numClasses
    end

    gain = wPriorFuzzyEntropy - bestEntropy

    if depth < 5 && self.ignore
        if gain < self.minGain
            @info "Feature $(fIndex) index $(bestIndex), gain $(gain) REJECTED"
            return []
        end
        @info "Feature $(fIndex) index $(bestIndex), gain $(gain) ACCEPTED"
    else
        if (gain < self.minGain || !mdlStopCondition(gain, wPriorFuzzyEntropy, bestS0S1S2, entropy,None))
            @info "Feature $(fIndex) index $(bestIndex), gain $(gain) REJECTED"
            return []
        end
        @info "Feature $(fIndex) index $(bestIndex), gain $(gain) ACCEPTED"
    end
    
    left = self.calculateCutPointsIndex(self, fIndex, first, bestIndex, depth+1, baseIndex + ".l")
    right = self.calculateCutPointsIndex(self, fIndex, bestIndex, lastPlusOne, depth+1, baseIndex + ".r")

    return left + [(gain*s/size(self.data)[1], bestIndex, baseIndex)] + right
end

function traceback(splitList)
    """
    Trace a list of splits back to its original order.
    
    Given a list of split points, the gain value, and its position in the
    splitting "chain". Given by the regular expression T(.[lr]){0,}
    A split in position T.x..[lr] can exist only if exist the corresponding
    father split.
    """

    if length(splitList) == 0
        return []
    end

    base = 'T'
    listValues = []
    k = [k for (k,val) in enumerate(splitList) if val[2] == base][0]

    # The first split point 'T' is added to the list of selected splits.
    append!(listValues, splitList.pop(k))

    available = []
    while splitList
        # The child of the latest chosen point are inserted in the availability list
        vals = [val for val in splitList if val[2] == (base + ".r") || val[2] == (base + ".l")]
        for val in vals
            available.append(splitList.pop(splitList.index(val)))
        end

        # Choose one of the split point from the available points in the list
        chosen = maximum(available)

        # Update the current chosen point
        base = chosen[2]

        # Insert the chosen point in the list
        append!(listValues, available.pop(available.index(chosen)))
    end

    return listValues
end

function calculateWeightedFuzzyImpurity(partition::Matrix{Float64}, partitionCardinality::Float64, impurity::Function)
    """
    Perform weighted fuzzy impurity evaluation on a fuzzy partition.

    # Arguments
    - `partition`: np.array, shape(numPartitions, numClasses)
        Partition with numPartitions subpartitions and numClasses classes.
    partitionCardinality: float
        cardinality of the partition
    impurity: function
        evaluate impurity
    Returns
    -------
    Weighted impurity.    
    """
    
    summedImp = 0.0
    
    for k in range(1, length(partition))
        summedImp += sum(partition[k])/partitionCardinality * impurity(partition[k], sum(partition[k])) 
    end

    return summedImp
end

function entropy(counts::Array{Float64}, totalCount)
    """
    Evaluates the entropy.
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