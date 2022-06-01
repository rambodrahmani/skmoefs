"""
    SK-MOEFS Toolbox.
"""

include("rcs.jl")
include("moel.jl")
include("moea.jl")
using Random
using JLD

milestones = [500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]

function set_rng_seed(seed::Int64)
    Random.seed!(seed)
end

function make_path(path::String)
    mkpath(path)
end

function is_object_present(name)
    """
    Check if file exists on the filesystem.
    """
    return isfile(name * ".obj")
end

function store_object(filename::String, name::String, object::Any)
    """
    Save object as Julia Data format (JLD).
    """
    save(filename * ".jld", name, object)
end

function load_object(filename::String)
    """
    Load JLD file to object.
    """
    return load(filename * ".jld")
end

function load_dataset(name::String)
    """
    Loads the dataset identified by the given name.

    # Arguments
    - `name::String`: dataset name

    # Return
    - `X`: matrix NxM (N: number of samples; M: number of features) representing input data
    - `y`: vector Nx1 representing the output data
    - `attributes`: range of values in the format [min, max] for each feature
    - `input`: names of the features
    - `output`: name of the outputs
    """

    attributes = []
    inputs = []
    outputs = []
    X = []
    y = []

    open("dataset/" * name * ".dat") do io
        while !eof(io)
            line = readline(io)
            if startswith(line, "@")
                txt = split(line)
                if txt[1] == "@attribute"
                    domain = findall(r"(\[|\{)(.+)(\]|\})", line)
                    range_str = SubString(line, domain[1])
                    range_str = replace(range_str, "{" => "", "}" => "", "[" => "", "]" => "")
                    range_fl = parse.(Float64, split(range_str, ','))
                    push!(attributes, range_fl)
                elseif txt[1] == "@inputs"
                    for i in range(1, length(txt) - 1)
                        push!(inputs, replace(txt[i + 1], "," => ""))
                    end
                elseif txt[1] == "@outputs"
                    push!(outputs, txt[2])
                end
            else
                if length(line) != 0
                    words = parse.(Float64, split(line, ','))
                    push!(X, words[begin:end-1])
                    push!(y, floor(Int, words[end]))
                end
            end
        end
        attributes = convert(Array{Array{Float64}, 1}, attributes)
        inputs = convert(Array{String, 1}, inputs)
        outputs = convert(Array{String,1}, outputs)
        X = copy(hcat(X...)')
        y = convert(Array{Int64, 1}, y)
    end

    return X, y, attributes, inputs, outputs
end

function normalize(X::Matrix{Float64}, y::Vector{Int64}, attributes::Vector{Array{Float64}})
    """
    Normalizes the dataset.

    # Arguments
    - `X`: matrix NxM (N: number of samples; M: number of features) representing input data
    - `y`: vector Nx1 representing the output data
    - `attributes`: range of values in the format [min, max] for each feature

    # Return
    - `X` normalized such that 0 <= X[i,j] <= 1 for every i,j
    - `y` normalized such that 0 <= y[i] <= 1 for every i
    """

    n_samples = size(X)[1]
    n_features = size(X)[2]
    X_n = Array{Float64, 2}(undef, n_samples, n_features)
    min_values = zeros(n_features)
    max_values = zeros(n_features)
    for j in range(1, n_features)
        min_values[j] = attributes[j][1]
        max_values[j] = attributes[j][2]
        for i in range(1, n_samples)
            X_n[i,j] = (X[i,j] - min_values[j]) / (max_values[j] - min_values[j])
        end
    end

    return X_n, y
end

mutable struct MPAES_RCS
    """
    In charge of handling the rule and condition selection multi-objective
    learning scheme, by means of (2 + 2)M-PAES algorithm.
    """
    learning_scheme::MOEL_FRBC
    M::Int64
    Amin::Int64
    capacity::Int64
    divisions::Int64
    variator::RCSVariator
    initializer::RCSInitializer
    objectives::Array{String}
    moea_type::String
    problem::RCSProblem
end

MPAES_RCS() = MPAES_RCS(MOEL_FRBC(), 0, 0, 0, 0, RCSVariator(), RCSInitializer(), [], "", RCSProblem())

function __init__(self::MPAES_RCS, M::Int64=100, Amin::Int64=1, capacity::Int64=64,
            divisions::Int64=8, variator::RCSVariator=createRCSVariator(),
            initializer::RCSInitializer=createRCSInitializer(), objectives=["accuracy", "trl"],
            moea_type::String="mpaes22")
    """
    # Arguments:
    - `M`: Number of maximum rules that a solution (Fuzzy classifier) can have
    - `Amin`: minimum number of antecedents for each rule
    - `capacity`: maximum capacity of the archive
    - `divisions`: number of divisions for the grid within the archive
    - `variator`: genetic operator (embeds crossover and mutation functionalities)
    - `initializer`: define the initial state of the archive
    - `objectives`: list of objectives to minimize/maximize. As for now, only the pairs:
        ('accuracy', 'trl') and ('auc', 'trl') are provided
    - `moea_type`: the Multi-Objective Evolutionary Algorithm from the ones available. If not
        provided, the default one is MPAES(2+2)
    """
    self.learning_scheme = __init__(MOEL_FRBC(), [])
    self.M = M
    self.Amin = Amin
    self.capacity = capacity
    self.divisions = divisions
    self.objectives = objectives
    self.variator = variator
    self.initializer = initializer
    self.moea_type = moea_type

    return self
end

function _initialize(self::MPAES_RCS, X::Matrix{Float64}, y::Array{Int64})
    fit_tree(self.initializer, X, y)
    J = get_rules(self.initializer)
    splits = get_splits(self.initializer)

    # Find initial set of rules and fuzzy partitions
    # MOEFS problem
    self.problem = CreateRCSProblem(self.Amin, self.M, J, splits, self.objectives)
    set_training_set(self.problem, X, y)
end

function _choose_algorithm(self::MPAES_RCS)
    if self.moea_type == "nsga2"
        self.algorithm = createNSGAIIS()
    elseif self.moea_type == "nsga3"
        self.algorithm = createNSGAIIIS()
    elseif self.moea_type == "gde3"
        self.algorithm = createGDE3S()
    elseif self.moea_type == "ibea"
        self.algorithm = createIBEAS()
    elseif self.moea_type == "moead"
        self.algorithm = createMOEADS()
    elseif self.moea_type == "spea2"
        self.algorithm = createSPEA2S()
    elseif self.moea_type == "epsmoea"
        self.algorithm = createEpsMOEAS()
    else
        self.algorithm = createMPAES2_2(self.problem, self.variator, self.capacity, self.divisions)
    end
end

function fit(self::MPAES_RCS, X::Matrix{Float64}, y::Array{Int64}, max_evals::Int64=10000)
    _initialize(self, X, y)
    _choose_algorithm(self)

    error("Implementation To Be Continued.")
end

function CREATE_MPAES_RCS(M::Int64=100, Amin::Int64=1, capacity::Int64=64, divisions::Int64=8,
    variator::RCSVariator=createRCSVariator(), initializer::RCSInitializer=createRCSInitializer(),
    objectives=["accuracy", "trl"], moea_type::String="mpaes22")
    return __init__(MPAES_RCS(), M, Amin, capacity, divisions, variator, initializer,
            objectives, moea_type)
end