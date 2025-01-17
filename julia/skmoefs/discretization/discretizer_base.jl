"""
    Fuzzy Discretization
"""

mutable struct FuzzyDiscretization
    method::String
    numSet::Int64
    continous::Array{Bool}
    N::Int64
    M::Int64
end

FuzzyDiscretization() = FuzzyDiscretization("uniform", 7, [true], 0, 0)

function __init__(self::FuzzyDiscretization, method::String="uniform", numSet::Int64=7)
    @assert method in ["uniform", "equifreq"] "Invalid discretization method."
    self.method = method
    @assert numSet >= 3 "Invalid number of sets."
    self.numSet = numSet

    return self
end

function run(self::FuzzyDiscretization, data::Matrix{Float64}, continous::Vector{Bool})
    self.continous = continous
    self.N, self.M = size(data)
    
    splits = []
    for k in range(1, self.M)
        cutPoints = []
        if self.continous[k]
            if self.method == "equifreq"
                [append!(cutPoints, sort(data[:,k])[floor(Int64, index) + 1]) for index in LinRange(0, self.N-1, self.numSet)]
            elseif self.method == "uniform"
                cutPoints = collect(LinRange(minimum(data[:,k]), maximum(data[:,k]), self.numSet))
            end
            if length(unique(cutPoints)) < 3
                append!(splits, hcat(zeros(1,1), ones(1,self.numSet-1)))
            else
                uPoints = unique(cutPoints)
                append!(splits, [append!(uPoints, ones(1, clamp(self.numSet - length(uPoints), 0, self.numSet - length(uPoints))))])
            end
        else
            append!(splits, [[]])
        end
    end

    return convert(Array{Array{Float64}, 1}, splits)
end

function createFuzzyDiscretizer(method::String="uniform", numSet::Int64=7)
    return __init__(FuzzyDiscretization(), method, numSet)
end

function runFuzzyDiscretizer(self::FuzzyDiscretization, data::Matrix{Float64}, continous::Vector{Bool})
    return run(self, data, continous)
end