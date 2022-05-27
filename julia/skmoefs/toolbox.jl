"""
    SK-MOEFS Toolbox.
"""

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
    return isfile(name * ".obj")
end

function store_object(filename::String, name::String, object::Any)
    save(filename * ".jld", name, object)
end

function load_object(filename::String)
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
                    range_str = replace(range_str, "{" => "[")
                    range_str = replace(range_str, "}" => "]")
                    range_str = replace(range_str, "[" => "")
                    range_str = replace(range_str, "]" => "")
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
        X = copy(hcat(X...)')
        y = convert(Array{Int64, 1}, y)
        attributes = convert(Array{Array{Float64}, 1}, attributes)
        inputs = convert(Array{String, 1}, inputs)
        outputs = convert(Array{String,1}, outputs)
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