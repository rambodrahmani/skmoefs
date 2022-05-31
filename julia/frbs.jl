mutable struct ClassificationRule
    antecedent::Dict
    fuzzyset::Dict
    consequent::Int64
    weight::Float64
    label::UInt64
end

function __init__(self::ClassificationRule, antecedent::Dict, fuzzyset::Dict, consequent::Int64, weight::Float64=1.0, label::UInt64=nothing)
    """
    @param antecedent: antecedents in the rule. For each one [a, b, c] that represents the triangular fuzzyset
    @param fuzzyset:
    @param consequent:
    @param weight:
    @param label:
    """
    self.antecedent = antecedent
    self.fuzzyset = fuzzyset
    self.consequent = consequent
    self.weight = weight

    if !isnothing(label)
        self.label = label
    else
        self.label = objectid(self)
    end
end

function membership_value(mf::Array{Float64}, value::Float64, type::String="triangular")
    if type=="triangular" && length(mf) == 3
        if mf[0] == mf[1] # left triangular
            if value < mf[0]
                return 1.0
            elseif value > mf[2]
                return 0.0
            else
                return 1.0 - ((value - mf[1]) / (mf[2] - mf[1]))
            end
        elseif mf[1] == mf[2] # right triangular
            if value < mf[0]
                return 0.0
            elseif value > mf[2]
                return 1.0
            else
                return (value - mf[0]) / (mf[1] - mf[0])
            end
        else # triangular
            if value < mf[0] || value > mf[2]
                return 0.0
            elseif value <= mf[1]
                return (value - mf[0]) / (mf[1] - mf[0])
            elseif value <= mf[2]
                return 1.0 - ((value - mf[1]) / (mf[2] - mf[1]))
            end
        end
    end

    # Not implemented
    return 0.0
end