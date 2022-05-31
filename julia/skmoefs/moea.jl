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
    if (self.algorithm.nfe % 100) == 0
        println("Fitness evaluations " * self.algorithm.nfe)
    end
    if length(self.snapshots) < length(milestones) && self.algorithm.nfe >= milestones[length(self.snapshots)]
        print("new milestone at " * string(self.algorithm.nfe))
        append!(self.snapshots, self.algorithm.archive)
    end
    self.algorithm.iterate()
end

function createNSGAIIS()
    return initialize(NSGAIIS())
end

mutable struct NSGAIIIS
    """
        Extended version of NSGA3 algorithm with added support for snapshots.
    """
    snapshots::Array{Any}
    algorithm::PyObject
end

NSGAIIIS() = NSGAIIIS([], platypus.algorithms.NSGAIII)

function initialize(self::NSGAIIIS)
    self.snapshots = []
    algorithm = platypus.algorithms.NSGAIII()
    algorithm.initialize()

    return self
end

function iterate(self::NSGAIIIS)
    if (self.algorithm.nfe % 100) == 0
        print("Fitness evaluations " * self.algorithm.nfe)
    end
    if length(self.snapshots) < length(milestones) && self.algorithm.nfe >= milestones[length(self.snapshots)]
        print("new milestone at " * string(self.algorithm.nfe))
        append!(self.snapshots, self.algorithm.population)
    end
    self.algorithm.iterate()
end

function createNSGAIIIS()
    return initialize(NSGAIIIS())
end

mutable struct GDE3S
    """
        Extended version of GDE3 algorithm with added support for snapshots.
    """
    snapshots::Array{Any}
    algorithm::PyObject
end

GDE3S() = GDE3S([], platypus.algorithms.GDE3)

function initialize(self::GDE3S)
    self.snapshots = []
    algorithm = platypus.algorithms.GDE3()
    algorithm.initialize()

    return self
end

function iterate(self::GDE3S)
    if (self.algorithm.nfe % 100) == 0
        print("Fitness evaluations " * self.algorithm.nfe)
    end
    if length(self.snapshots) < length(milestones) && self.algorithm.nfe >= milestones[length(self.snapshots)]
        print("new milestone at " * string(self.algorithm.nfe))
        append!(self.snapshots, self.algorithm.population)
    end
    self.algorithm.iterate()
end

function createGDE3S()
    return initialize(GDE3S())
end

mutable struct IBEAS
    """
        Extended version of IBEA algorithm with added support for snapshots.
    """
    snapshots::Array{Any}
    algorithm::PyObject
end

IBEAS() = IBEAS([], platypus.algorithms.IBEA)

function initialize(self::IBEAS)
    self.snapshots = []
    algorithm = platypus.algorithms.IBEA()
    algorithm.initialize()

    return self
end

function iterate(self::IBEAS)
    if (self.algorithm.nfe % 100) == 0
        print("Fitness evaluations " * self.algorithm.nfe)
    end
    if length(self.snapshots) < length(milestones) && self.algorithm.nfe >= milestones[length(self.snapshots)]
        print("new milestone at " * string(self.algorithm.nfe))
        append!(self.snapshots, self.algorithm.population)
    end
    self.algorithm.iterate()
end

function createIBEAS()
    return initialize(IBEAS())
end