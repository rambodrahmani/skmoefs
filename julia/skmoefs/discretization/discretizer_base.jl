"""
"""

using Ranges

mutable struct fuzzyDiscretization
    method::String
    numSet::Int64
    continous::Array{Bool}
    N::Int64
    M::Int64
end

fuzzyDiscretization() = fuzzyDiscretization("uniform", 7, True, 0, 0)

function __init__(self::fuzzyDiscretization, method::String="uniform", numSet::Int64=7)
    @assert mathod in ["uniform", "equifreq"] "Invalid discretization method."
    self.method = method
    @assert numSet >= 3 "Invalid number of sets."
    self.numSet = numSet
end

function run(self::fuzzyDiscretization, data::Matrix{Float64}, continous::Array{Bool})
    self.continous = continous
    self.N, self.M = size(data)
    
    splits = []
    for k in range(self.M)
        if self.continous[k]
            if self.method == "equifreq"
                cutPoints = sort(data[:,k])[range(0, self.N-1, self.numSet)]
            elseif self.method == "uniform"
                cutPoints = range(minimum(data[:,k]), maximum(data[:,k]), self.numSet)
            end
            if length(unique!(cutPoints)) < 3
                append!(splits, hcat(zeros(1,1), ones(1,self.numSet-1)))
            else
                uPoints = unique!(cutPoints)
                append!(uPoints, ones(1, self.numSet - len(uPoints)))
                append!(splits, uPoints)
            end
        else
            append!(splits, [])
        end
    end

    return splits
end