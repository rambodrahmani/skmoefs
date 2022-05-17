"""
Developer Harness Test
"""

include("fuzzysets/SingletonFuzzySet.jl")
include("fuzzysets/TriangularFuzzySet.jl")
include("fuzzysets/TrapezoidalFuzzySet.jl")

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
Trapezoidal_fuzzy_params = [0.02777778, 0.27083333, 0.51388889, 0.75694444, 0.76944444, 1.]
fuzzy_trapezoidals = create_TrapezoidalFuzzySet(Trapezoidal_fuzzy_params)
for fuzzy_trapezoidal in fuzzy_trapezoidals 
    println(fuzzy_trapezoidal)    
end
