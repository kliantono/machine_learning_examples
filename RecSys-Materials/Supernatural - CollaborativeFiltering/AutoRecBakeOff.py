from WorkoutLens import WorkoutLens
from AutoRecAlgorithm import AutoRecAlgorithm
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadWorkoutLensData():
    ml = WorkoutLens()
    print("Loading workout ratings...")
    data = ml.loadWorkoutLensLatestSmall()
    print("\nComputing workout popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadWorkoutLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#Autoencoder
AutoRec = AutoRecAlgorithm()
evaluator.AddAlgorithm(AutoRec, "AutoRec")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
