"""
    SK-MOEFS Toolbox.
"""

milestones = [500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]

function set_rng_seed(seed::Int64)
    Random.seed!(seed)
end

function make_path(path::String)
    mkpath(path)
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
                    words = convert(Array{Float64, 1}, words)
                    println(words)
                    push!(X, words[begin:end-1])
                    push!(y, floor(Int, words[end]))
                end
            end
        end
        X = convert(Array{Array{Float64}, 1}, X)
        y = convert(Array{Int64, 1}, y)
        attributes = convert(Array{Array{Float64}, 1}, attributes)
        inputs = convert(Array{String, 1}, inputs)
        outputs = convert(Array{String,1}, outputs)
    end

    return X, y, attributes, inputs, outputs
end

function normalize(X, y, attributes)
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

    n_features = size(X)[1]
    min_values = zeros(n_features)
    max_values = zeros(n_features)
    for j in range(1, n_features)
        min_values[j] = attributes[j][0]
        max_values[j] = attributes[j][1]
        for i in range(1, len(X[:, j]))
            X[i, j] = (X[i, j] - min_values[j]) / (max_values[j] - min_values[j])
        end
    end

    return X, y
end