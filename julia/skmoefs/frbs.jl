using ScikitLearn
@sk_import metrics: roc_auc_score
@sk_import preprocessing: LabelBinarizer

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

function compute_weights_fast(train_x, train_y, ant_matrix, cons_vect, part_matrix)
    """
    :param train_x: Training input
    :param train_y: Training output
    :param ant_matrix: antecedents
    :param cons_vect: consequents
    :param part_matrix: partitions of fuzzysets
    :return: for each rule in the RB, compute the weight from the provided training set
    """

    weights = np.ones(size(ant_matrix)[1])

    # for each rule
    for i in range(1, size(ant_matrix)[1])
        matching = 0.0
        total = 0.0
        for j in range(1, size(train_y)[1])
            matching_degree = 1.0
            for k in range(1, size(ant_matrix)[2])
                if !isnan(ant_matrix[i][k])
                    index = int(ant_matrix[i][k])
                    ant = part_matrix[k][index:index+3]
                    m_degree = membership_value(ant, train_x[j][k])
                    matching_degree *= m_degree
                end
            end
            if train_y[j] == cons_vect[i]
                matching += matching_degree
            end
            total += matching_degree
        end
        if total == 0
            weights[i] = total
        else
            weights[i] = matching/total
        end
    end

    return weights
end

mutable struct FuzzyRuleBasedClassifier
    """
    Fuzzy Rule-Based Classifier.
    """
    rules::Array{ClassificationRule}
    partitions
    weights
end

function __init__(self::FuzzyRuleBasedClassifier, rules::Array{ClassificationRule}, partitions)
    """
    :param rules: a list of ClassificationRule objects
    :param partitions: fuzzyset partitions for each fuzzy input
    """

    # RB info
    self.rules = rules

    # DB info
    self.partitions = partitions
end

function addrule(self::FuzzyRuleBasedClassifier, new_rule::ClassificationRule)
    append!(self.rules, new_rule)
end

function num_rules(self::FuzzyRuleBasedClassifier)
    return length(self.rules)
end

function predict(self::FuzzyRuleBasedClassifier, x)
    return predict_fast(x, self.ant_matrix, self.cons_vect, self.weights, self.part_matrix)
end

function compute_weights(self::FuzzyRuleBasedClassifier, train_x, train_y)
    self.weights = compute_weights_fast(train_x, train_y, self.ant_matrix, self.cons_vect, self.part_matrix)
end

function trl(self::FuzzyRuleBasedClassifier)
    n_antecedents = [length(rule.antecedent) for rule in self.rules]
    return sum(n_antecedents)
end

function accuracy(self::FuzzyRuleBasedClassifier, x, y)
    return sum(predict(self, x) == y) / length(y)
end

function auc(self::FuzzyRuleBasedClassifier, x, y)
    y_pred = predict(self, x)

    # Binarize labels (One-vs-All)
    lb = LabelBinarizer()
    lb.fit(y)

    # Transform labels
    y_bin = lb.transform(y)
    y_pred_bin = lb.transform(y_pred)
    return roc_auc_score(y_bin, y_pred_bin, average="macro")
end

function _get_labels(self::FuzzyRuleBasedClassifier, size::Int64)
    if size == 3
        return ["L", "M", "H"]
    elseif size == 5
        return ["VL", "L", "M", "H", "VH"]
    elseif size == 7
        return ["VL", "L", "ML", "M", "MH", "H", "VH"]
    end
end

function show_RB(self::FuzzyRuleBasedClassifier, inputs, outputs, f::Any=nothing)
    if !isnothing(f)
        write(f, "RULE BASE\n")
        startBold = ""
        endBold = ""
    else
        print("RULE BASE")
        startBold = "\033[1m"
        endBold = "\033[0m"
    end
    if_keyword = startBold * "IF" * endBold
    then_keyword = startBold * "THEN" * endBold
    is_keyword = startBold * "is" * endBold

    for (i,rule) in enumerate(self.rules)
        if_part = if_keyword * " "
        count = 0
        for key in rule.antecedent
            size = length(self.partitions[key])
            labels = _get_labels(self, size)
            if count > 0
                if_part += startBold * "AND " * endBold
            end
            count += 1
            if isnothing(inputs)
                feature = "X_" * string(key + 1)
            else
                feature = inputs[key]
            end
            if_part *= feature
            if_part *= " " * is_keyword * " " * labels[rule.fuzzyset[key] - 1] * " "
        end
        if isnothing(outputs)
            output = "Class"
        else
            output = outputs[0]
        end
        then_part = then_keyword * " " * output * " is " * string(rule.consequent)
        if !isnothing(f)
            f.write(string(i+1) * ":\t" * if_part * then_part * "\n")
        else
            print(string(i+1) * ":\t" * if_part * then_part)
        end
    end
end