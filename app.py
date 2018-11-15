import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

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

def getSimilarDescriptionMovies(name):
    tfidf = TfidfVectorizer(stop_words='english')
    dataSetTwo['overview'] = dataSetTwo['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataSetTwo['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(dataSetTwo.index, index=dataSetTwo['original_title']).drop_duplicates()
    return getRecommendationsBasedOnDesc(name, cosine_sim, indices)


def getRecommendationsBasedOnDesc(title, cosine_sim, indices):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return dataSetTwo['original_title'].iloc[movie_indices]

def putLineSeparation():
    print("--------------------------------------------------------*---")

# Code launch
readDataSets()
combineDataSets()
print("DEMOGRAPHIC RECOMMENDATIONS")
print(getDemographicRecom())
putLineSeparation()
print("TOP POPULAR RECOMMENDATIONS")
print(getTopPopularMovies())
putLineSeparation()
print("MOVIES LIKE THE DARK KNIGHT")
print(getSimilarDescriptionMovies("The Dark Knight"))