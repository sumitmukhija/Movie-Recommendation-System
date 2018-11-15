import pandas as pd
import numpy as np


def readDataSets():
    dataSetOne = pd.read_csv("resources/datasets/tmdb_5000_credits.csv")
    dataSetTwo = pd.read_csv("resources/datasets/tmdb_5000_movies.csv")


def combineDataSets():
    dataSetOne.columns = ["id", "title", "cast", "crew"]
    dataSetTwo = dataSetTwo.merge(dataSetOne, on="id")
