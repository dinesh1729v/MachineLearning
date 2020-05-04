import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
games = pd.read_csv("games.csv")

# Print the names of the columns in games
print games.columns
print games.shape

# Make a histogram of all the ratings in the average_rating column
plt.hist(games["average_rating"])
plt.show()

# Print the first row of all the games with zero scores
print(games[games["average_rating"] == 0].iloc[0])

# Print the first row of games with scores greater 0
print(games[games["average_rating"] > 0].iloc[0])

# Remove any rows without user reviews
games = games[games["users_rated"] > 0]

# Remove any rows with missing values
games = games.dropna(axis=0)

# Make a histogram of all the average_ratings
plt.hist(games["average_rating"])
plt.show()

# Correlation matrix
corrmat = games.corr()
sns.heatmap(corrmat, vmax=.8, square=True)
fig = plt.figure(figsize=(12, 9))
plt.show()

# GEt all the columns from the dataFrame
columns = games.columns.tolist()

# Dropping the un necessary columns
columns = [c for c in columns if c not in ["bayes_averate_ratng", "average_rating", "type", "name", "id"]]

# Store the variable we'll be predicting on
target = "average_rating"

# Generate training and test datasets
from sklearn.cross_validation import train_test_split

# Generate training set
train = games.sample(frac=0.8, random_state = 1)

# Select anything not in the training set and pu it in test
test = games.loc[~games.index.isin(train.index)]

# Print shapes
print train.shape
print test.shape

# Import LinearRegression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the model class
LR = LinearRegression()

# Fit the model the training data
LR.fit(train[columns], train[target])

# Generate the predictions for the test set
predictions = LR.predict(test[columns])

# Compute error between our test predictions and actual values
print mean_squared_error(predictions, test[target])

# Using the Random Forest model
from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor(n_estimators=100,min_samples_leaf=10, random_state=1)
RFR.fit(train[columns], train[target])

predictions = RFR.predict(test[columns])

#Compute the error
print mean_squared_error(predictions, test[target])

# Make Predictions of rating with both models
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1,-1))

print rating_LR
print rating_RFR

print test[target].iloc[0]

