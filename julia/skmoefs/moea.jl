"""
    Multiobjective evolutionary algorithms implementation based on Platypus.
"""

using PyCall

# import python modules
platypus = pyimport("platypus")

milestones = [500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]

mutable struct NSGAIIS
    """
        Extended version of NSGA2 algorithm with added support for snapshots.
    """
    snapshots::Array{Any}
    algorithm::PyObject
end

NSGAIIS() = NSGAIIS([], platypus.algorithms.NSGAII)

function initialize(self::NSGAIIS)
    self.snapshots = []
    algorithm = platypus.algorithms.NSGAII()
    algorithm.initialize()

    return self
end

function iterate(self::NSGAIIS)
    if (self.nfe % 100) == 0
        println("Fitness evaluations " * self.nfe)
    end
    if length(self.snapshots) < length(milestones) && self.nfe >= milestones[length(self.snapshots)]
        print("new milestone at ", string(self.algorithm.nfe))
        append!(self.snapshots, self.algorithm.archive)
    end
    self.algorithm.iterate()
end

function createNSGAIIS()
    return initialize(NSGAIIS())
end