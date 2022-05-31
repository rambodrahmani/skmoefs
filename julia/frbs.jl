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
        if mf[0] == mf[1]       # left triangular
            if value < mf[0]
                return 1.0
            elseif value > mf[2]
                return 0.0
            else
                return 1.0 - ((value - mf[1]) / (mf[2] - mf[1]))
            end
        elseif mf[1] == mf[2]   # right triangular
            if value < mf[0]
                return 0.0
            elseif value > mf[2]
                return 1.0
            else
                return (value - mf[0]) / (mf[1] - mf[0])
            end
        else                    # triangular
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

function predict_fast(x, ant_matrix, cons_vect, weights, part_matrix)
    """
    :param x: input matrix NxM where N is the number of samples and M is the number of features
    :param ant_matrix: antecedents of every rule in the RB
    :param cons_vect: consequents of every rule in the RB
    :param weights:
    :param part_matrix: partitions of fuzzysets
    :return:
    """
    sample_size = size(x)[1]
    y = zeros(sample_size)

    # for each sample
    for i in range(1, sample_size)
        best_match_index = 0
        best_match = 0.0
        # for each rule
        for j in range(1, size(ant_matrix)[1])
            matching_degree = 1.0
            for k in range(1, size(ant_matrix)[2])
                if !isnan(ant_matrix[j][k])
                    base = Int64(ant_matrix[j][k])
                    ant = part_matrix[k][base:base+3]
                    m_degree = membership_value(ant, x[i][k])
                    matching_degree *= m_degree
                end
            end
            if (weights[j] * matching_degree) > best_match
                best_match_index = j
                best_match = weights[j] * matching_degree
            end
        end
        y[i] = cons_vect[best_match_index]
    end

    return y
end