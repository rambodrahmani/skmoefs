"""
    SK-MOEFS Toolbox.
"""

using Ranges

milestones = [500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]

function load_dataset(name::String)
    """
    Loads the dataset identified by the given name.

    # Arguments
    - `name::String`: dataset name
    # Return
    - `X`: NumPy matrix NxM (N: number of samples; M: number of features) representing input data
    - `y`: Numpy vector Nx1 representing the output data
    - `attributes`: range of values in the format [min, max] for each feature
    - `input`: names of the features
    - `output`: name of the outputs
    """
    attributes = []
    inputs = []
    outputs = []
    X = []
    y = []
    open("skmoefs/dataset/" * name * ".dat") do io
        while !eof(io)
            line = readline(io)
            if startswith(line, "@")
                txt = split(line)
                if txt[1] == "@attribute"
                    domain = findall(r"(\[|\{)(.+)(\]|\})", line)
                    push!(attributes, SubString(line, domain[1]))
                elseif txt[1] == "@inputs"
                    for i in range(length(txt) - 1)
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
        X = reshape(X, length(X), 1)
    end

    return X, y, attributes, inputs, outputs
end