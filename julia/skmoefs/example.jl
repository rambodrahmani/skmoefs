"""
    Examples of Usage of SK-MOEFS.
"""

include("toolbox.jl")
using Random

function test1()
end

function test2()
end

function test3(dataset::String, algorithm::String, seed::Int64, nEvals::Int64=50000, store::Bool=false)
    path = "julia/results/" * dataset * '/' * algorithm * '/'
    make_path(path)
    set_rng_seed(seed)

    X, y, attributes, inputs, outputs = load_dataset(dataset)
    X_n, y_n = normalize(X, y, attributes)
end

test3("iris", "mpaes22", 2, 2000, true)