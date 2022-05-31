"""
    Multiobjective evolutionary algorithms implementation based on Platypus.
"""

using PyCall

# import python modules
platypus = pyimport("platypus")

milestones = [500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]

mutable struct MOEAGenerator
    """
        Generates the initial population of a Multi-Objective GA.
    """
    counter::Int64
    generator::PyObject
end

MOEAGenerator() = MOEAGenerator(0, platypus.Generator)

function __init__(self::MOEAGenerator)
    self.generator.__init__()
    self.counter = 0
end

function generate(self::MOEAGenerator, problem)
    self.counter += 1
    solution = problem.random()

    return solution
end

function createMOEAGenerator()
    return __init__(MOEAGenerator())
end

mutable struct RandomSelector
    """
        Randomly selects an individual from the population
    """
    selector::PyObject
end

RandomSelector() = RandomSelector(platypus.Selector)

function __init_(self::RandomSelector)
    self.selector.__init_()
end

function select_one(self::RandomSelector, population)
    return random.choice(population)
end

function createRandomSelector()
    return __init__(RandomSelector())
end

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

mutable struct MOEADS
    """
        Extended version of MOEAD algorithm with added support for snapshots.
    """
    snapshots::Array{Any}
    algorithm::PyObject
end

MOEADS() = MOEADS([], platypus.algorithms.MOEAD)

function initialize(self::MOEADS)
    self.snapshots = []
    algorithm = platypus.algorithms.MOEAD()
    algorithm.initialize()

    return self
end

function iterate(self::MOEADS)
    if (self.algorithm.nfe % 100) == 0
        print("Fitness evaluations " * self.algorithm.nfe)
    end
    if length(self.snapshots) < length(milestones) && self.algorithm.nfe >= milestones[length(self.snapshots)]
        print("new milestone at " * string(self.algorithm.nfe))
        append!(self.snapshots, self.algorithm.population)
    end
    self.algorithm.iterate()
end

function createMOEADS()
    return initialize(MOEADS())
end

mutable struct SPEA2S
    """
        Extended version of SPEA2 algorithm with added support for snapshots.
    """
    snapshots::Array{Any}
    algorithm::PyObject
end

SPEA2S() = SPEA2S([], platypus.algorithms.SPEA2)

function initialize(self::SPEA2S)
    self.snapshots = []
    algorithm = platypus.algorithms.SPEA2()
    algorithm.initialize()

    return self
end

function iterate(self::SPEA2S)
    if (self.algorithm.nfe % 100) == 0
        print("Fitness evaluations " * self.algorithm.nfe)
    end
    if length(self.snapshots) < length(milestones) && self.algorithm.nfe >= milestones[length(self.snapshots)]
        print("new milestone at " * string(self.algorithm.nfe))
        append!(self.snapshots, self.algorithm.population)
    end
    self.algorithm.iterate()
end

function createSPEA2S()
    return initialize(SPEA2S())
end

mutable struct EpsMOEAS
    """
        Extended version of Epsilon-MOEA algorithm with added support for snapshots.
    """
    snapshots::Array{Any}
    algorithm::PyObject
end

EpsMOEAS() = EpsMOEAS([], platypus.algorithms.EpsMOEA)

function initialize(self::EpsMOEAS)
    self.snapshots = []
    algorithm = platypus.algorithms.EpsMOEA()
    algorithm.initialize()

    return self
end

function iterate(self::EpsMOEAS)
    if (self.algorithm.nfe % 100) == 0
        print("Fitness evaluations " * self.algorithm.nfe)
    end
    if length(self.snapshots) < length(milestones) && self.algorithm.nfe >= milestones[length(self.snapshots)]
        print("new milestone at " * string(self.algorithm.nfe))
        append!(self.snapshots, self.algorithm.archive)
    end
    self.algorithm.iterate()
end

function createEpsMOEAS()
    return initialize(EpsMOEAS())
end