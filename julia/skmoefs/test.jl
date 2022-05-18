"""
    Developer Harness Test
"""

include("toolbox.jl")
include("fuzzysets/SingletonFuzzySet.jl")
include("fuzzysets/TriangularFuzzySet.jl")
include("fuzzysets/TrapezoidalFuzzySet.jl")
include("discretization/discretizer_base.jl")

###########
# FUZZYSETS
###########

# Fuzzy Universe

# Fuzzy Singleton
fuzzy_singleton = createSingletonFuzzySet(43.2, 3)
println(fuzzy_singleton)

fuzzy_singletons = createSingletonFuzzySets([17.5, 24.3, 11.4, 23.1])
for fuzzy_singleton in fuzzy_singletons
    println(fuzzy_singleton)
end

# Fuzzy Triangular Set
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
println(X)
println(y)
println(attributes)
println(inputs)
println(outputs)
println(typeof(X))
println(typeof(y))
println(typeof(attributes))
println(typeof(inputs))
println(typeof(outputs))

# normalize dataset
X_n, y_n = normalize(X, y, attributes);
println(X_n)
println(y_n)

#############
# DISCRETIZER
#############
fuzzy_discretizer = createFuzzyDiscretizer("uniform", 5)
fuzzy_splits = runFuzzyDiscretizer(fuzzy_discretizer, X_n, [true, true, true, true])
print(fuzzy_splits)