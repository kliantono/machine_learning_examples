from WorkoutLens import WorkoutLens
from RBMAlgorithm import RBMAlgorithm
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

#RBM
RBM = RBMAlgorithm(epochs=1) #change the epochs
evaluator.AddAlgorithm(RBM, "RBM")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
