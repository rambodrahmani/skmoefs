"""
    Examples of Usage of SK-MOEFS.
"""

include("skmoefs/toolbox.jl")
using PyCall
using Random
using ScikitLearn.CrossValidation: train_test_split

# add path for importing local module
pushfirst!(PyVector(pyimport("sys")["path"]), "/home/rr/DevOps/skmoefs/python/")

# import python modules
skmoefs_py_toolbox = pyimport("skmoefs.toolbox")

function test1()
    X, y, attributes, inputs, outputs = load_dataset("newthyroid")
    X_n, y_n = normalize(X, y, attributes)
    Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=0.3)
end

function test2()
    X, y, attributes, inputs, outputs = load_dataset("newthyroid")
    X_n, y_n = normalize(X, y, attributes)
end

function test3(dataset::String, algorithm::String, seed::Int64, nEvals::Int64=50000, store::Bool=false)
    path = "julia/results/" * dataset * '/' * algorithm * '/'
    make_path(path)
    set_rng_seed(seed)

    X, y, attributes, inputs, outputs = load_dataset(dataset)
    X_n, y_n = normalize(X, y, attributes)

    Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=0.3, random_state=seed)

    Amin = 1
    M = 50
    capacity = 32
    divisions = 8
end

test1()
#test2()
#test3("iris", "mpaes22", 2, 2000, true)