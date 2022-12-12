import inline as inline
import pandas as pd
import matplotlib
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import numpy as np

from DecisionTree import DecisionTree
from KNN import KNN
from SVM import SVM
from adaboost import adaboost
from logistic import logistic

matplotlib.rcParams['figure.figsize'] = (12, 8)
movies_table = pd.read_csv("tmdb_5000_movies.csv")

# Create Classification version of target variable
# remove these columns from the spreadsheet 90000000
movies_filtered = movies_table.drop(['id', 'original_title', 'release_date', 'status', 'title','month'], axis=1)
movies_filtered['runtimequality'] = ['good' if x >= 120 else 'bad' for x in movies_filtered['runtime']]
movies_filtered['revenuequality'] = ['good' if x >= 90000000 else 'bad' for x in movies_filtered['revenue']]
movies_filtered['budgetquality'] = ['good' if x >= 40000000 else 'bad' for x in movies_filtered['budget']]
movies_filtered['popularityquality'] = ['good' if x >= 30 else 'bad' for x in movies_filtered['popularity']]
movies_filtered['revenueavgquality'] = ['good' if (x >= 4 and y >= 35000000) else 'bad' for x, y in zip(movies_filtered['vote_average'], movies_filtered['revenue'])]


# # # Separate feature variables and target variable
data_runtime = movies_filtered.drop(['runtime', 'runtimequality', 'revenuequality', 'budgetquality', 'popularityquality','revenueavgquality','year'], axis=1)
data_revenue = movies_filtered.drop(['revenue', 'runtimequality', 'revenuequality', 'budgetquality', 'popularityquality','revenueavgquality','year'], axis=1)
data_budget = movies_filtered.drop(['budget', 'runtimequality', 'revenuequality', 'budgetquality', 'popularityquality','revenueavgquality','year'], axis=1)
data_popularity = movies_filtered.drop(['popularity', 'runtimequality', 'revenuequality', 'budgetquality', 'popularityquality','revenueavgquality','year'], axis=1)
data_mix = movies_filtered.drop(['revenue','vote_average', 'runtimequality', 'revenuequality', 'budgetquality', 'popularityquality','revenueavgquality','year'], axis=1)

lbl_runtime = movies_filtered['runtimequality']
lbl_revenue = movies_filtered['revenuequality']
lbl_budget = movies_filtered['budgetquality']
lbl_popularity = movies_filtered['popularityquality']
lbl_mix = movies_filtered['revenueavgquality']

# runtime #
print("\nResult for runtime only :")
KNN(data_runtime, lbl_runtime)
DecisionTree(data_runtime, lbl_runtime)
logistic(data_runtime, lbl_runtime)
SVM(data_runtime, lbl_runtime)
adaboost(data_runtime, lbl_runtime)

# revenue #
print("\nResult for revenue only :")
KNN(data_revenue, lbl_revenue)
DecisionTree(data_revenue, lbl_revenue)
logistic(data_revenue, lbl_revenue)
SVM(data_revenue, lbl_revenue)
adaboost(data_revenue, lbl_revenue)

# budget #
print("\nResult for budget only :")
KNN(data_budget, lbl_budget)
DecisionTree(data_budget, lbl_budget)
logistic(data_budget, lbl_budget)
SVM(data_budget, lbl_budget)
adaboost(data_budget, lbl_budget)

# popularity #
print("\nResult for popularity only :")
KNN(data_popularity, lbl_popularity)
DecisionTree(data_popularity, lbl_popularity)
logistic(data_popularity, lbl_popularity)
SVM(data_popularity, lbl_popularity)
adaboost(data_popularity, lbl_popularity)

# mix #
print("\nResult for vote average and revenue :")
KNN(data_mix, lbl_mix)
DecisionTree(data_mix, lbl_mix)
logistic(data_mix, lbl_mix)
SVM(data_mix, lbl_mix)
adaboost(data_mix, lbl_mix)