"""
    Implementation of the rule and condition selection (RCS) learning scheme for
    classification problems.
"""

include("fmdt.jl")

mutable struct RCSVariator
    p_crb::Float64
    p_cdb::Float64
    alpha::Float64
    p_mrb::Float64
    p_mrb_add::Float64
    p_mrb_del::Float64
    p_mrb_flip::Float64
    p_mdb::Float64
    p_cond::Float64
end

RCSVariator() = RCSVariator(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function __init__(self::RCSVariator, p_crb::Float64=0.2, p_cdb::Float64=0.5, alpha::Float64=0.5,
                p_mrb::Float64=0.1, p_mrb_add::Float64=0.5, p_mrb_del::Float64=0.2,
                p_mrb_flip::Float64=0.7, p_mdb::Float64=0.2, p_cond::Float64=0.5)
    self.p_crb = p_crb
    self.p_cdb = p_cdb
    self.alpha = alpha
    self.p_mrb = p_mrb
    self.p_mrb_add = p_mrb_add
    self.p_mrb_del = p_mrb_del
    self.p_mrb_flip = p_mrb_flip
    self.p_mdb = p_mdb
    self.p_cond = p_cond

    return self
end

function createRCSVariator(p_crb::Float64=0.2, p_cdb::Float64=0.5, alpha::Float64=0.5,
                    p_mrb::Float64=0.1, p_mrb_add::Float64=0.5, p_mrb_del::Float64=0.2,
                    p_mrb_flip::Float64=0.7, p_mdb::Float64=0.2, p_cond::Float64=0.5)
    return __init__(RCSVariator(), p_crb, p_cdb, alpha, p_mrb, p_mrb_add, p_mrb_del,
                    p_mrb_flip, p_mdb, p_cond)
end

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

function fit_tree(self::RCSInitializer, X::Matrix{Float64}, y::Vector{Int64})
    continuous = ones(Bool, size(X)[2])
    cPoints = runFuzzyDiscretizer(self.discretizer, X, continuous)
    self.fTree = fitFMDTTree(self.tree, X, y, continuous, cPoints)

    error("Implementation To Be Continued.")
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