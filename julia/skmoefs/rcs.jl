"""
    Implementation of the rule and condition selection (RCS) learning scheme for
    classification problems.
"""

include("fmdt.jl")
using PyCall

# import python modules
platypus = pyimport("platypus")

mutable struct RCSVariator
    variator::PyObject
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

RCSVariator() = RCSVariator(platypus.Variator, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function __init__(self::RCSVariator, p_crb::Float64=0.2, p_cdb::Float64=0.5, alpha::Float64=0.5,
                p_mrb::Float64=0.1, p_mrb_add::Float64=0.5, p_mrb_del::Float64=0.2,
                p_mrb_flip::Float64=0.7, p_mdb::Float64=0.2, p_cond::Float64=0.5)
    self.variator = platypus.Variator(2)
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

function _remove_duplicates(self::RCSVariator, solution)
    """
    Remove duplicates from a solution.
    """
    problem = solution.problem
    rules = set(solution.variables[:problem.crb_l:2])
    for i in range(1, problem.M - 1)
        if !(solution.variables[2 * i] in rules)
            solution.variables[2 * i] = 0
            solution.variables[2 * i + 1][:] = [false for _ in range(problem.F)]
        else
            rules.remove(solution.variables[2 * i])
        end
    end

    return solution
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
    fTree::FMDT
    splits::Array{Array{Float64}}
    rules::Matrix{Int64}
end

RCSInitializer() = RCSInitializer(FuzzyDiscretization(), FMDT(), FMDT(), [], Array{Int64, 2}(undef, 1, 1))

function __init__(self::RCSInitializer, discretizer::FuzzyDiscretization=createFuzzyDiscretizer("uniform", 5), tree::FMDT=createFMDT())
    self.discretizer = discretizer
    self.tree = tree
    self.splits = []

    return self
end

function fit_tree(self::RCSInitializer, X::Matrix{Float64}, y::Vector{Int64})
    continuous = ones(Bool, size(X)[2])
    cPoints = runFuzzyDiscretizer(self.discretizer, X, continuous)
    self.fTree = fitFMDTTree(self.tree, X, y, continuous, cPoints)
    self.rules = _csv_ruleMine(self.fTree.tree, size(X)[2], [])
    self.rules[:, end] .-= 1
    self.splits = self.fTree.cPoints
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

mutable struct RCSProblem
    Amin::Int64
    M::Int64
    Mmax::Int64
    Mmin::Int64
    F::Int64
    TFmax::Int64
    J::Matrix{Int64}
    initialBfs::Array{Array{Float64}}
    G::Array{Int64}
    objectives::Array{String}
    crb_l::Int64
    gran_l::Int64
    cdb_l::Int64
    problem::PyObject
    types::Array{Any}
    directions::Array{Int64}
    train_x::Matrix{Float64}
    train_y::Array{Int64}
end

RCSProblem() = RCSProblem(0, 0, 0, 0, 0, 0, Array{Int64, 2}(undef, 1, 1), [],
                        [], [], 0, 0, 0, platypus.Problem, [], [], Array{Float64, 2}(undef, 1, 1), [])

function __init__(self::RCSProblem, Amin::Int64, M::Int64, J::Matrix{Int64},
                splits::Array{Array{Float64}}, objectives::Array{String}, TFmax::Int64=7)
    """
    @param Amin: minimum number of antecedents per rule
    @param M: maximum number of rules per individual
    @param J: matrix representing the initial rulebase
    @param splits: fuzzy partitions for input features
    @param objectives: array of objectives (2 or more). The first one is typically expresses the performance
                    for predictions. Indeed, the solutions will be sorted according to this objective (best-to-worst)
    @param TFmax: maximum number of fuzzy sets per feature (i.e. maximum granularity)
    """
    self.Amin = Amin
    self.M = M
    self.Mmax = size(J)[1]
    self.Mmin = length(unique(J[:, end]))
    self.M = clamp(self.M, self.Mmin, self.Mmax)
    self.F = size(J)[2] - 1
    self.TFmax = TFmax

    self.J = J
    self.initialBfs = splits
    self.G = [length(split) for split in splits]

    self.objectives = objectives
    self.crb_l = 2 * self.M
    self.gran_l = self.F
    self.cdb_l = self.gran_l * self.TFmax

    # Creation of the problem and assignment of the types
    #      CRB: rule part
    #      Granularities: number of fuzzy sets
    #      CDB: database part
    self.problem = platypus.Problem(self.crb_l + self.gran_l + self.cdb_l, length(objectives))
    self.types = []
    for _ in range(1, self.M)
        push!(self.types, platypus.Real(0, self.M))
        push!(self.types, platypus.Binary(self.F))
    end
    [push!(self.types, platypus.Real(2, self.TFmax)) for _ in range(1, self.F)]
    [push!(self.types, platypus.Real(0, 1)) for _ in range(1, self.cdb_l)]

    # Maximize objectives by minimizing opposite
    self.directions = [platypus.Problem.MINIMIZE for _ in range(1, length(objectives))]

    return self
end

function set_training_set(self::RCSProblem, train_x::Matrix{Float64}, train_y::Array{Int64})
    self.train_x = train_x
    self.train_y = train_y
end

function CreateRCSProblem(Amin::Int64, M::Int64, J::Matrix{Int64}, splits::Array{Array{Float64}},
                    objectives::Array{String}, TFmax::Int64=7)
    return __init__(RCSProblem(), Amin, M, J, splits, objectives, TFmax)
end