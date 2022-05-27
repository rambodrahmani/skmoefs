"""
    Multi-objective Evolutionary Learning Scheme.
""" 

mutable struct MOELScheme
end

function fit(self::MOELScheme, X::Matrix{Float64}, y::Array{Int64})
end

function cross_val_score(self::MOELScheme, X::Matrix{Float64}, y::Array{Int64}, num_fold::Int64)
end

function show_pareto(self::MOELScheme)
    """
    Extracts and plots the values of accuracy and explainability. Returns a
    plot of the approximated Pareto front, both on the training and the test
    sets.
    """
end

function show_model(self::MOELScheme, position)
    """
    Given the position of a model in the Pareto front, this method shows the
    set of fuzzy linguistic rules and the fuzzy partitions associated with
    each linguistic attribute. The predefined model of choice is, as always,
    the FIRST solution.
    """
end

function __getitem__(self::MOELScheme, item)
end

mutable struct MOEL_FRBC
    """
    MOEL scheme for Fuzzy Rule-based Classifiers (FRBCs).
    """
    classifiers::Array{Any}
end

MOEL_FRBC() = MOEL_FRBC([])

function __init__(self::MOEL_FRBC, classifiers::Array{Any}=nothing)
    if isnothing(classifiers)
        classifiers = []
    end
    self.classifiers = classifiers
end

function predict(self::MOEL_FRBC, X::Matrix{Float64}, position::String="first")
    """
    In charge of predicting the class labels associated with a new set of
    input patterns.
    """
    n_classifiers = length(self.classifiers)
    if n_classifiers == 0
        return nothing
    else
        index = Dict("first"=>1, "median"=>floor(Int64, n_classifiers/2), "last"=>n_classifiers)
        return self.classifiers[index[lowercase(position)]].predict(X)
    end
end

function score(self::MOEL_FRBC, X::Matrix{Float64}, y::Array{Int64}, sample_weight=nothing)
    """
    Generates the values of the accuracy and explainability measures for the
    selected model.
    """
    n_classifiers = length(self.classifiers)
    if n_classifiers > 1
        indexes = Dict(0, floor(Int64, n_classifiers / 2), n_classifiers)
        for index in indexes
            clf = self.classifiers[index]
            accuracy = accuracy(self, clf)
            complexity = trl(self, clf)
            print(accuracy, complexity)
        end
    end
end

function accuracy(self::MOEL_FRBC, classifier)
end

function trl(self::MOEL_FRBC, classifier)
end

mutable struct MOEL_FRBR
    """
    MOEL scheme for Fuzzy Rule-based Regressors (FRBRs).
    """
    regressors::Array{Any}
end

MOEL_FRBR() = MOEL_FRBR([])

function __init__(self::MOEL_FRBR, regressors::Array{Any}=nothing)
    if isnothing(regressors)
        regressors = []
    end
    self.regressors = regressors
end

function predict(self::MOEL_FRBR, X::Matrix{Float64}, position::String="first")
    """
    In charge of predicting the class labels associated with a new set of
    input patterns.
    """
    n_regressors = length(self.regressors)
    if n_regressors == 0
        return nothing
    elseif n_regressors == 1
        return self.regressors[1].predict()
    else
        index = Dict("first"=>1, "median"=>floor(Int64, n_regressors/2), "last"=>n_regressors)
        return self.regressors[index[lowercase(position)]].predict(X)
    end
end

function score(self::MOEL_FRBR, X::Matrix{Float64}, y::Array{Int64}, sample_weight=nothing)
    """
    Generates the values of the accuracy and explainability measures for the
    selected model.
    """
    n_regressors = length(self.regressors)
    if n_regressors > 1
        indexes = Dict(0, floor(Int64, n_regressors / 2), n_regressors)
        for index in indexes
            rgr = self.regressors[index]
            accuracy = accuracy(self, rgr)
            complexity = trl(self, rgr)
            print(accuracy, complexity)
        end
    end
end

function accuracy(self::MOEL_FRBR, regressor)
end

function trl(self::MOEL_FRBR, regressor)
end