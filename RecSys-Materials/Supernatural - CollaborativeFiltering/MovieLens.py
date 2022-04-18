import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np
from snowflake import connector
import pandas as pd

conn = connector.connect(
    user='kliantono',
    password='ford.SIG7tuch_grar',
    account='ada15167',
    warehouse='COMPUTE_WH',
    database='WITHIN'
)

# cur_interactions = conn.cursor()
cur_items = conn.cursor()
cur_ratings = conn.cursor()

# interactions_sql = '''
#     select
#         ws.USER_ID
#         , ws.WORKOUT_ID as item_id
#         , date_part('epoch_millisecond', ws.CREATED_AT) as timestamp
#         -- Map workout 'starts' to click
#         , 'Click' as event_type
#         , null as event_value
#     from within.pgprodflow.workout_sessions ws
#     join within.pgprodflow.workouts w
#         on ws.WORKOUT_ID = w.ID
#     where ws.CREATED_AT >= '2020-04-23'
#         and w.WORKOUT_TYPE in ('classic', 'boxing')
#         and w.ID NOT IN (1407, 1405, 1361, 1319, 965)
#         and ws.CREATED_AT > w.LAUNCH_DATE
#
#     UNION
#
#     select
#         ws.USER_ID
#         , ws.WORKOUT_ID as item_id
#         , date_part('epoch_millisecond', ws.CREATED_AT) as timestamp
#         -- Map workout completes to watch
#         , 'Watch' as event_type
#         , ws.TOTAL_SCORE as event_value
#     from within.pgprodflow.workout_sessions ws
#     join within.pgprodflow.workouts w
#         on ws.WORKOUT_ID = w.ID
#     where ws.CREATED_AT >= '2020-04-23'
#         and w.WORKOUT_TYPE in ('classic', 'boxing')
#         and w.ID NOT IN (1407, 1405, 1361, 1319, 965)
#         and ws.CREATED_AT > w.LAUNCH_DATE
#         -- use the incomplete flag to determine workout complete
#         and ws.INCOMPLETE = false
# '''

items_sql = '''
    WITH genre AS (
        SELECT
            g.ID
            , IFF(g.genres_list != '', g.genres_list, 'No Genre') AS genres
            , IFF(g.collections_list != '', g.collections_list, 'No Collections') AS collections
            , IFF(g.programs_list != '', g.programs_list, 'No Programs') AS programs
            , IFF(g.carousels_list != '', g.carousels_list, 'No Carousels') AS carousels
        FROM ( -- created a subquery to make it easier to read. We need another IFF function outside of the subquery,
            -- because LISTAGG returns am empty STRING if we input NULL values, an we want to replace that will a STRING
            -- declaring that there are e.g. 'No Genre', 'No Collections', etc.
            SELECT
                w.ID
                -- There are multiple rows for the same WORKOUT_ID for each distinct TYPES & VALUE ({genre: pop, rock, collection: decades...}
                -- Two VALUES for the same TYPE (e.g. genre: pop, rock), equates to two rows for the WORKOUT_ID
                -- In order to aggregate the VALUES of the same TYPE into a single row for each WORKOUT_ID, we must use the IFF function
                -- inside the LISTAGG function and provide a NULL value for the ELSE condition (if the WORKOUT_ID is missing the declared TYPE.)
                -- If we substitute the NULL value with a STRING value (e.g.'No genre') for the ELSE condition, it would duplicate the STRING
                -- for the number of rows there are for a WORKOUT_ID (e.g. 'Pop | No genre | No genre', instead of just 'Pop')
                , LISTAGG(IFF(c.TYPE = 'genre', c.NAME, NULL), '|') WITHIN GROUP(ORDER BY c."ORDER") AS genres_list
                , LISTAGG(IFF(c.TYPE = 'collection', c.NAME, NULL), '|') WITHIN GROUP(ORDER BY c."ORDER") AS collections_list
                , LISTAGG(IFF(c.TYPE = 'program', c.NAME, NULL), '|') WITHIN GROUP(ORDER BY c."ORDER") AS programs_list
                , LISTAGG(IFF(c.TYPE = 'carousel', c.NAME, NULL), '|') WITHIN GROUP(ORDER BY c."ORDER") AS carousels_list
            FROM WITHIN.PGPRODFLOW.WORKOUTS w
            LEFT JOIN WITHIN.PGPRODFLOW.CATEGORY_WORKOUTS cw ON cw.WORKOUT_ID = w.ID
            LEFT JOIN WITHIN.PGPRODFLOW.CATEGORIES c ON c.ID = cw.CATEGORY_ID
            WHERE 1=1
                -- Has to be a workout in headset by the current date, no tutorials or demos, only boxing and flow
                AND w.DISABLED != TRUE
                AND w.VISIBILITY = 'public'
                AND w.IS_LIFECYCLE = 0
                AND w.WORKOUT_TYPE in ('classic', 'boxing')
                AND w.ID NOT IN (1407, 1405, 1361, 1319, 965)
                AND w.LAUNCH_DATE <= CURRENT_DATE()
                AND cw.PUBLISHED_VERSION = 1
            GROUP BY 1
            ORDER BY 1
        ) g
    )

    , rating AS (
        -- Normalize ratings to the first 14 days after launch
        SELECT
            r.TABLE_ID,
            AVG(r.RATING) avg_rating
        FROM WITHIN.PGPRODFLOW.RATINGS r
        LEFT JOIN WITHIN.PGPRODFLOW.WORKOUTS w ON w.ID = r.TABLE_ID
        WHERE 1=1
            -- Has to be a workout in headset by the current date, no tutorials or demos, only boxing and flow
            AND r.RATING != 0
            AND r.CREATED_AT < DATEADD('DAY', 14, w.LAUNCH_DATE)
            AND w.LAUNCH_DATE <= CURRENT_DATE()
            AND w.DISABLED != TRUE
            AND w.VISIBILITY = 'public'
            AND w.IS_LIFECYCLE = 0
            AND w.WORKOUT_TYPE in ('classic', 'boxing')
            AND w.ID NOT IN (1407, 1405, 1361, 1319, 965)
        GROUP BY 1
    )

    SELECT
        w.ID AS movieId -- WORKOUT_ID
        , w.TITLE AS title
        --, ROUND(r.AVG_RATING, 2) AS AVERAGE_RATING -- RATING
        --, DATE_PART('epoch_millisecond', w.LAUNCH_DATE) AS CREATION_TIMESTAMP -- LAUNCH_TIMESTAMP
        --, IFF(wi.NAME IS NOT NULL, wi.NAME, NULL) AS INTENSITY
        --, IFF(g.genres IS NOT NULL, g.genres, NULL) AS genres
        , IFF(g.genres IS NOT NULL, g.genres, 'No genre') AS genres
        --, IFF(g.collections IS NOT NULL, g.collections, NULL) AS COLLECTIONS
    --     , IFF(g.programs IS NOT NULL, g.programs, NULL) AS PROGRAMS
        --, IFF(g.carousels IS NOT NULL, g.carousels, NULL) AS CAROUSELS
        --, w.WORKOUT_TYPE AS CONTENT_CLASSIFICATION -- WORKOUT_TYPE
        --, CONCAT(i.FIRST_NAME, ' ', i.LAST_NAME) AS CONTENT_OWNER -- COACH
        --, ROUND(w.DURATION / 60, 2) AS DURATION -- DURATION
        --, w.DESCRIPTION -- PRODUCT_DESCRIPTION
    FROM WITHIN.PGPRODFLOW.WORKOUTS w
    LEFT JOIN WITHIN.PGPRODFLOW.WORKOUT_INTENSITY wi ON wi.id = w.intensity_id
    LEFT JOIN WITHIN.PGPRODFLOW.INSTRUCTOR i ON i.id = w.instructor_id
    LEFT JOIN rating r ON r.table_id = w.id
    LEFT JOIN genre g ON g.ID = w.ID
    WHERE 1=1
        -- Has to be a workout in headset by the current date, no tutorials or demos, only boxing and flow
        AND w.LAUNCH_DATE <= CURRENT_DATE()
        AND w.DISABLED != TRUE
        AND w.VISIBILITY = 'public'
        AND w.IS_LIFECYCLE = 0
        AND w.WORKOUT_TYPE in ('classic', 'boxing')
        AND w.ID NOT IN (1407, 1405, 1361, 1319, 965)
    ORDER BY 1
'''

ratings_sql = '''
    SELECT
        AUTHOR_ID AS userId
        , TABLE_ID AS movieId
        , RATING AS rating
        --# , DATE_PART('epoch_millisecond', CREATED_AT) as timestamp
    FROM WITHIN.PGPRODFLOW.RATINGS r
    WHERE 1=1
        AND RATING != 0
'''

# cur_interactions.execute(interactions_sql)
cur_items.execute(items_sql)
cur_ratings.execute(ratings_sql)

# interactions_df = cur_interactions.fetch_pandas_all()
items_df = cur_items.fetch_pandas_all()
ratings_df = cur_ratings.fetch_pandas_all()

# cur_interactions.close()
cur_items.close()
cur_ratings.close()

# interactions_df.to_csv('../ml-latest-small/workout_interactions.csv', header=True)
# items_df.to_csv('../ml-latest-small/workout_items.csv', index=False, header=True)
# ratings_df.to_csv('../ml-latest-small/workout_ratings.csv', index=False, header=True)

# print(interactions_df.head())
print(items_df.head())
print(ratings_df.head())

# print(df.dtypes)
# print(df.info())
# print(df.describe())
# print(isinstance(df, pd.DataFrame))

class MovieLens:

    movieID_to_name = {}
    name_to_movieID = {}
    # ratingsPath = '../ml-latest-small/workout_ratings.csv'
    ratingsPath = ratings_df
    # moviesPath = '../ml-latest-small/workout_items.csv'
    moviesPath = items_df

    def loadMovieLensLatestSmall(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}

        reader = Reader(rating_scale=(1, 5))

        ratingsDataset = Dataset.load_from_df(self.ratingsPath, reader=reader)

        # reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        # ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        for index, row in self.moviesPath.iterrows():
            # movieID = int(row[0])
            movieID = int(row['MOVIEID'])
            # movieName = row[1]
            movieName = row['TITLE']
            self.movieID_to_name[movieID] = movieName
            self.name_to_movieID[movieName] = movieID

        # with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
        #         movieReader = csv.reader(csvfile)
        #         next(movieReader)  #Skip header line
        #         for row in movieReader:
        #             movieID = int(row[0])
        #             movieName = row[1]
        #             self.movieID_to_name[movieID] = movieName
        #             self.name_to_movieID[movieName] = movieID

        return ratingsDataset

    def getUserRatings(self, user):
        userRatings = []
        hitUser = False

        for index, row in self.ratingsPath.iterrows():
            # userID = int(row[0])
            userID = int(row['USERID'])
            if (user == userID):
                # movieID = int(row[1])
                movieID = int(row['MOVIEID'])
                # rating = float(row[2])
                rating = float(row['RATING'])
                userRatings.append((movieID, rating))
                hitUser = True
            if (hitUser and (user != userID)):
                break


        # with open(self.ratingsPath, newline='') as csvfile:
        #     ratingReader = csv.reader(csvfile)
        #     next(ratingReader)
        #     for row in ratingReader:
        #         userID = int(row[0])
        #         if (user == userID):
        #             movieID = int(row[1])
        #             rating = float(row[2])
        #             userRatings.append((movieID, rating))
        #             hitUser = True
        #         if (hitUser and (user != userID)):
        #             break

        return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)

        for index, row in self.ratingsPath.iterrows():
            movieID = int(row['MOVIEID'])
            ratings[movieID] += 1

        rank = 1
        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1

        # with open(self.ratingsPath, newline='') as csvfile:
        #     ratingReader = csv.reader(csvfile)
        #     next(ratingReader)
        #     for row in ratingReader:
        #         movieID = int(row[1])
        #         ratings[movieID] += 1
        # rank = 1
        # for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        #     rankings[movieID] = rank
        #     rank += 1
        return rankings

    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0

        for index, row in self.moviesPath.iterrows():
            movieID = int(row['MOVIEID'])
            genreList = row['GENRES'].split('|')
            genreIDList = []
            for genre in genreList:
                if genre in genreIDs:
                    genreID = genreIDs[genre]
                else:
                    genreID = maxGenreID
                    genreIDs[genre] = genreID
                    maxGenreID += 1
                genreIDList.append(genreID)
            genres[movieID] = genreIDList

        # with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
        #     movieReader = csv.reader(csvfile)
        #     next(movieReader)  #Skip header line
        #     for row in movieReader:
        #         movieID = int(row[0])
        #         genreList = row[2].split('|')
        #         genreIDList = []
        #         for genre in genreList:
        #             if genre in genreIDs:
        #                 genreID = genreIDs[genre]
        #             else:
        #                 genreID = maxGenreID
        #                 genreIDs[genre] = genreID
        #                 maxGenreID += 1
        #             genreIDList.append(genreID)
        #         genres[movieID] = genreIDList
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (movieID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield

        return genres

    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)

        for index, row in self.moviesPath.iterrows():
            movieID = int(row['MOVIEID'])
            title = row['TITLE']
            m = p.search(title)
            year = m.group(1)
            if year:
                years[movieID] = int(year)

        # with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
        #     movieReader = csv.reader(csvfile)
        #     next(movieReader)
        #     for row in movieReader:
        #         movieID = int(row[0])
        #         title = row[1]
        #         m = p.search(title)
        #         year = m.group(1)
        #         if year:
        #             years[movieID] = int(year)
        return years

    def getMiseEnScene(self):
        mes = defaultdict(list)
        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
            mesReader = csv.reader(csvfile)
            next(mesReader)
            for row in mesReader:
                movieID = int(row[0])
                avgShotLength = float(row[1])
                meanColorVariance = float(row[2])
                stddevColorVariance = float(row[3])
                meanMotion = float(row[4])
                stddevMotion = float(row[5])
                meanLightingKey = float(row[6])
                numShots = float(row[7])
                mes[movieID] = [avgShotLength, meanColorVariance, stddevColorVariance,
                   meanMotion, stddevMotion, meanLightingKey, numShots]
        return mes

    def getMovieName(self, movieID):
        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]
        else:
            return ""

    def getMovieID(self, movieName):
        if movieName in self.name_to_movieID:
            return self.name_to_movieID[movieName]
        else:
            return 0