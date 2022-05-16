"""
Developer Harness Test
"""

include("fuzzysets/SingletonFuzzySet.jl")

# Fuzzy Singleton
fuzzy_singleton_example = createSingletonFuzzySet(43.2, 3)
println(fuzzy_singleton_example)

fuzzy_singletons_example = createSingletonFuzzySets([17.5, 24.3, 11.4, 23.1])
for fuzzy_singleton in fuzzy_singletons_example
    println(fuzzy_singleton)
end