from WorkoutLens import WorkoutLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
        
# testSubject = 'Ua61ZFw' # Dan
# testSubject = 'Uq6knIA' # Mom
# testSubject = 'UkgX8SQ' # Alex
# testSubject = 'UKXQuLg' # Sean
testSubject = 'UQM6URw' # Jenn
# testSubject = 'Uyyo4Kw' # Fharzana
k = 10

ml = WorkoutLens()
data = ml.loadWorkoutLensLatestSmall()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': False
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

testUserInnerID = trainSet.to_inner_uid(testSubject)

# Get the top K items we rated
testUserRatings = trainSet.ur[testUserInnerID]
kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])

# Get similar items to stuff we liked (weighted by rating)
candidates = defaultdict(float)
for itemID, rating in kNeighbors:
    similarityRow = simsMatrix[itemID]
    for innerID, score in enumerate(similarityRow):
        candidates[innerID] += score * (rating / 5.0)
    
# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1
    
# # Get top-rated items from similar users:
# pos = 0
# for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
#     if not itemID in watched:
#         workoutID = trainSet.to_raw_iid(itemID)
#         print(ml.getWorkoutID(ml.getWorkoutName(int(workoutID))),ml.getWorkoutName(int(workoutID)), ratingSum)
#         pos += 1
#         if (pos > 10):
#             break

# Get top-rated items from similar users:
# pos = 0
rec_list = list()
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        workoutID = trainSet.to_raw_iid(itemID)
        rec_list.append((ml.getWorkoutID(ml.getWorkoutName(int(workoutID))),ml.getWorkoutName(int(workoutID)),ratingSum))
        rec_list = list(set(rec_list))
        # if len(rec_list) > 100:
        # # pos += 1
        # # if (pos > 20):
        #     break

rec_list.sort(key=lambda x: x[2], reverse=True)
counter_list = []

rec_num = 0
for index, rec in enumerate(rec_list):
    if rec[0:2] not in counter_list:
        counter_list.append(rec[0:2])
        rec_num += 1
        print(f"{rec_num}.", rec[0:2])

    if rec_num == 20:
        break


