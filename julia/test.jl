"""
    Developer Harness Test
"""

using Dates
using Statistics
using ScikitLearn.CrossValidation: train_test_split

include("skmoefs/rcs.jl")
include("skmoefs/toolbox.jl")
include("skmoefs/fuzzysets/SingletonFuzzySet.jl")
include("skmoefs/fuzzysets/TriangularFuzzySet.jl")
include("skmoefs/fuzzysets/TrapezoidalFuzzySet.jl")
include("skmoefs/discretization/discretizer_base.jl")
include("skmoefs/discretization/discretizer_crisp.jl")
include("skmoefs/discretization/discretizer_fuzzy.jl")

function test_fit_timed()
    n = 30
    intervals = zeros(n)
    for seed in range(1, n)
        start_time = datetime2unix(now())
        test_fit("iris", "mpaes22", 0)
        intervals[seed] = datetime2unix(now()) - start_time
    end
    println("Mean processing time is " * string(mean(intervals)) * " seconds")
end

function test_fit(dataset::String, algorithm::String, seed::Int64, nEvals::Int64=50000, store::Bool=false)
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
    variator = createRCSVariator()
    discretizer = createFuzzyDiscretizer("uniform", 5)
    initializer = createRCSInitializer(discretizer)
    if store
        base = path * "moefs_" * string(seed)
        if !is_object_present(base)
            mpaes_rcs_fdt = CREATE_MPAES_RCS(M, Amin, capacity, divisions, variator,
                                      initializer, ["accuracy", "trl"], algorithm)
            fit(mpaes_rcs_fdt, Xtr, ytr, nEvals)
            store_object(base, "MPAES_RCS", mpaes_rcs_fdt)
        else
            mpaes_rcs_fdt = load_object(base)["MPAES_RCS"]
        end
    else
        mpaes_rcs_fdt = CREATE_MPAES_RCS(M, Amin, capacity, divisions, variator,
                                      initializer, ["accuracy", "trl"], algorithm)
        fit(mpaes_rcs_fdt, Xtr, ytr, nEvals)
    end
end

###########
# FUZZYSETS
###########

# Fuzzy Universe
fuzzy_universe = createUniverseFuzzySet()
println(fuzzy_universe)

# Fuzzy Singleton
fuzzy_singleton = createSingletonFuzzySet(7.0, 3)
println(fuzzy_singleton)
plotSingletonFuzzySet(fuzzy_singleton)
savefig("./plots/singleton_fuzzy_set.pdf")

fuzzy_singletons = createSingletonFuzzySets([17.5, 24.3, 11.4, 23.1])
for fuzzy_singleton in fuzzy_singletons
    println(fuzzy_singleton)
end

# Fuzzy Triangular Set
fuzzy_triangular = createTriangularFuzzySet([17.5, 24.3, 11.4])
println(fuzzy_triangular)
plotTriangularFuzzySet(fuzzy_triangular)
savefig("./plots/triangular_fuzzy_set.pdf")

triangular_fuzzy_params = [0.02777778, 0.27083333, 0.51388889, 0.75694444, 1.]
fuzzy_triangulars = createTriangularFuzzySets(triangular_fuzzy_params, true)
for fuzzy_triangular in fuzzy_triangulars
    println(fuzzy_triangular)
end

# Fuzzy Trapezoidal Set
trapezoidal_fuzzy_params = [0.02777778, 0.27083333, 0.51388889, 0.982374]
trpzPrm = 0.56
fuzzy_trapezoidal = __createTrapezoidalFuzzySet(trapezoidal_fuzzy_params)
println(fuzzy_trapezoidal)
__plotTrapezoidalFuzzySet(fuzzy_trapezoidal)
savefig("./plots/trapezoidal_fuzzy_set.pdf")

trapezoidal_fuzzy_params = [0.02777778, 0.27083333, 0.51388889, 0.12345678]
trpzPrm = 0.56
fuzzy_trapezoidals = __createTrapezoidalFuzzySets(trapezoidal_fuzzy_params, true)
for fuzzy_trapezoidal in fuzzy_trapezoidals
    println(fuzzy_trapezoidal)
end

trapezoidal_fuzzy_params = [0.02777778, 0.27083333, 0.51388889]
trpzPrm = 0.56
fuzzy_trapezoidal = createTrapezoidalFuzzySet(trapezoidal_fuzzy_params, trpzPrm)
println(fuzzy_trapezoidal)

trapezoidal_fuzzy_params = [0.02777778, 0.27083333, 0.51388889, 0.12345678]
trpzPrm = 0.56
fuzzy_trapezoidals = createTrapezoidalFuzzySets(trapezoidal_fuzzy_params, trpzPrm, true)
for fuzzy_trapezoidal in fuzzy_trapezoidals
    println(fuzzy_trapezoidal)
end


#########
# TOOLBOX
#########

# load dataset
X, y, attributes, inputs, outputs = load_dataset("iris")
println(attributes)
println(inputs)
println(outputs)
println(X)
println(y)
println(typeof(attributes))
println(typeof(inputs))
println(typeof(outputs))
println(typeof(X))
println(typeof(y))

# normalize dataset
X_n, y_n = normalize(X, y, attributes)
println(X_n)
println(y_n)
println(typeof(X_n))
println(typeof(y_n))


##############
# DISCRETIZERS
##############
fuzzy_discretizer = createFuzzyDiscretizer("uniform", 5)
fuzzy_splits = runFuzzyDiscretizer(fuzzy_discretizer, X_n, [true, true, true, true])
println(fuzzy_splits)

crisp_mdlf_discretizer = createCrispMDLFDiscretizer(3, X_n, y, [true, true, true, true])
crisp_mdlf_splits = runCrispMDLFDiscretizer(crisp_mdlf_discretizer)
println(crisp_mdlf_splits)

fuzzy_mdlf_discretizer = createFuzzyMDLDiscretizer(3, X_n, y, [true, true, true, true])
fuzzy_mdlf_splits = runFuzzyMDLDiscretizer(fuzzy_mdlf_discretizer)
println(fuzzy_mdlf_splits)


#####
# RCS
#####
rcs_initializer = createRCSInitializer()


#test_fit_timed()
test_fit("iris", "mpaes22", 2, 2000, false)