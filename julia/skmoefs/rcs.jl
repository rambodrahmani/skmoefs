"""
    Implementation of the rule and condition selection (RCS) learning scheme for
    classification problems.
"""

include("fmdt.jl")

mutable struct RCSInitializer
    discretizer::FuzzyDiscretization
    tree::FMDT
    fTree
    splits
    rules
end

RCSInitializer() = RCSInitializer(createFuzzyDiscretizer("uniform", 5), createFMDT(), nothing, nothing, nothing)

function __init__(self::RCSInitializer, discretizer::FuzzyDiscretization=createFuzzyDiscretizer("uniform", 5), tree::FMDT=createFMDT())
    self.discretizer = discretizer
    self.tree = tree
    self.fTree = nothing
    self.splits = nothing
    self.rules = nothing

    return self
end

function fit_tree(self::RCSInitializer, X::Matrix{Float64}, y)
    continous = [true] * size(X)[2]
    cPoints = runFuzzyDiscretizer(self.discretizer, X, continous)
    self.fTree = self.tree.fit(X, y, cPoints, continous)
    self.rules = np.array(self.fTree.tree._csv_ruleMine(size(X)[2], []))
    self.rules[:, end] -= 1
    self.splits = np.array(self.fTree.cPoints)
end

function get_splits(self::RCSInitializer)
    return self.splits
end

function get_rules(self::RCSInitializer)
    return self.rules
end

function createRCSInitializer(discretizer::FuzzyDiscretization=createFuzzyDiscretizer("uniform", 5), tree::FMDT=createFMDT())
    return __init__(RCSInitializer(), discretizer, tree)
end