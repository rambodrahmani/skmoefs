"""
    Examples of Usage of SK-MOEFS.
"""

include("skmoefs/toolbox.jl")
include("skmoefs/discretization/discretizer_base.jl")
using PyCall
using Random
using ScikitLearn.CrossValidation: train_test_split

# add path for importing local skmoefs package
pushfirst!(PyVector(pyimport("sys")["path"]), "python/")

# import python modules
skmoefs_py_discretization = pyimport("skmoefs.discretization.discretizer_base")
skmoefs_py_toolbox = pyimport("skmoefs.toolbox")
skmoefs_py_rcs = pyimport("skmoefs.rcs")


function example1()
    X, y, attributes, inputs, outputs = load_dataset("newthyroid")
    X_n, y_n = normalize(X, y, attributes)
    my_moefs = skmoefs_py_toolbox.MPAES_RCS(variator=skmoefs_py_rcs.RCSVariator(), initializer=skmoefs_py_rcs.RCSInitializer())
    my_moefs.cross_val_score(X_n, y_n, nEvals=1000, num_fold=5)
end


function example2(seed::Int64)
    set_rng_seed(seed)

    X, y, attributes, inputs, outputs = load_dataset("newthyroid")
    X_n, y_n = normalize(X, y, attributes)
    Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=0.3)

    my_moefs = skmoefs_py_toolbox.MPAES_RCS(capacity=32, variator=skmoefs_py_rcs.RCSVariator(), initializer=skmoefs_py_rcs.RCSInitializer())
    my_moefs.fit(Xtr, ytr, max_evals=1000)

    my_moefs.show_pareto()
    my_moefs.show_pareto(Xte, yte)
    my_moefs.show_model("median", inputs=inputs, outputs=outputs)
end


function example3(dataset::String, algorithm::String, seed::Int64, nEvals::Int64=50000, store::Bool=false)
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
    variator = skmoefs_py_rcs.RCSVariator()
    discretizer = skmoefs_py_discretization.fuzzyDiscretization(numSet=5)
    initializer = skmoefs_py_rcs.RCSInitializer(discretizer=discretizer)
    if store
        base = path * "moefs_" * string(seed)
        if !is_object_present(base)
            mpaes_rcs_fdt = skmoefs_py_toolbox.MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                                      divisions=divisions, variator=variator,
                                      initializer=initializer, moea_type=algorithm,
                                      objectives=["accuracy", "trl"])
            mpaes_rcs_fdt.fit(Xtr, ytr, max_evals=nEvals)
            skmoefs_py_toolbox.store_object(mpaes_rcs_fdt, base)
        else
            mpaes_rcs_fdt = skmoefs_py_toolbox.load_object(base)
        end
    else
        mpaes_rcs_fdt = skmoefs_py_toolbox.MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                                  divisions=divisions, variator=variator,
                                  initializer=initializer, moea_type=algorithm,
                                  objectives=["accuracy", "trl"])
        mpaes_rcs_fdt.fit(Xtr, ytr, max_evals=nEvals)
    end

    mpaes_rcs_fdt.show_pareto()
    mpaes_rcs_fdt.show_pareto(Xte, yte)
    mpaes_rcs_fdt.show_pareto_archives()
    mpaes_rcs_fdt.show_pareto_archives(Xte, yte)
    mpaes_rcs_fdt.show_model("first", inputs, outputs)
    mpaes_rcs_fdt.show_model("median", inputs, outputs)
    mpaes_rcs_fdt.show_model("last", inputs, outputs)
end

#example1()
#example2(2)
example3("iris", "mpaes22", 2, 20000, false)