import pandas as pd
import numpy as np

dataSetOne = pd.DataFrame()
dataSetTwo = pd.DataFrame()
qualifiedMovies = pd.DataFrame()

def readDataSets():
    global dataSetOne, dataSetTwo
    dataSetOne = pd.read_csv("resources/datasets/tmdb_5000_credits.csv")
    dataSetTwo = pd.read_csv("resources/datasets/tmdb_5000_movies.csv")


def combineDataSets():
    global dataSetOne, dataSetTwo
    dataSetOne.columns = ["id", "title", "cast", "crew"]
    dataSetTwo = dataSetTwo.merge(dataSetOne, on="id")


def getDemographicRecom():
    # Computing weighted average for votes and users.
    global qualifiedMovies
    averageMeanAcrossDataSet = dataSetTwo["vote_average"].mean()
    minimumNumOfVotesToQualify = dataSetTwo["vote_count"].quantile(
        0.9
    )  # movies that have more votes that 90% of the total movies.
    qualifiedMovies = dataSetTwo.copy().loc[
        dataSetTwo["vote_count"] >= minimumNumOfVotesToQualify
    ]

    numberOfVotesForMovie = qualifiedMovies['vote_count']
    averageMovieRating = qualifiedMovies['vote_average']

    weightedRating= ((numberOfVotesForMovie/(numberOfVotesForMovie + minimumNumOfVotesToQualify)) * averageMovieRating) + ((minimumNumOfVotesToQualify/(minimumNumOfVotesToQualify + numberOfVotesForMovie)) *  averageMeanAcrossDataSet)
    qualifiedMovies["score"] = weightedRating

    bestMovies = qualifiedMovies.sort_values('score', ascending=False)
    return bestMovies[['original_title', 'score']].head(10)

def getTopPopularMovies():
    qualifiedMovies.sort_values('popularity', ascending=False)
    mostPopularMovies = qualifiedMovies[['original_title', 'score']].head(10)
    return mostPopularMovies


def putLineSeparation():
    print("-----------------------------------------------------------")

# Code launch
readDataSets()
combineDataSets()
print(getDemographicRecom())
putLineSeparation()
print(getTopPopularMovies())
