#import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:

df_raw = pd.read_csv('tennis_stats.csv')
print(df_raw.columns)


## perform exploratory analysis here:

inputs = ['FirstServe', 'FirstServePointsWon',
       'FirstServeReturnPointsWon', 'SecondServePointsWon',
       'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted',
       'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved',
       'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon',
       'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon',
       'TotalPointsWon', 'TotalServicePointsWon']
outputs = ['Wins', 'Losses', 'Winnings',
       'Ranking']
#for input in inputs:
#    for output in outputs:
#        plt.scatter(df_raw[input], df_raw[output], alpha=0.4)
#        plt.xlabel(input)
#        plt.ylabel(output)
#        plt.show()

# WEAK CORRELATIONS:
# Aces VS Loses
# Aces VS Winnings
# DoubleFaults VS Wins
# DoubleFaults VS Loses
# DoubleFaults VS Winnings

# VERY STRONG CORRELATIONS:
# Aces VS Wins
# BreakPointsFaced VS Wins
# BreakPointsFaced VS Loses
# BreakPointsFaced VS Winnings
# BreakPointsOpportunities VS Wins
# BreakPointsOpportunities VS Loses
# BreakPointsOpportunities VS Winnings
# ReturnGamesPlayed VS Wins
# ReturnGamesPlayed VS Loses
# ReturnGamesPlayed VS Winnings
# ServiceGamesPlayed VS Wins
# ServiceGamesPlayed VS Loses
# ServiceGamesPlayed VS Winings

## perform single feature linear regressions here:

df = df_raw[['Aces', 'BreakPointsFaced','BreakPointsOpportunities','ReturnGamesPlayed','ServiceGamesPlayed','DoubleFaults', 'Wins', 'Losses', 'Winnings']]

x = df[['Aces', 'BreakPointsFaced','BreakPointsOpportunities','ReturnGamesPlayed','ServiceGamesPlayed','DoubleFaults']]
y = df[[ 'Wins', 'Losses', 'Winnings']]

# one feature regression model using BreakPointsFaced to predict Wins

BreakPointsFaced = np.array(df['BreakPointsFaced']).reshape(-1,1)
Wins = np.array(df['Wins']).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(BreakPointsFaced,Wins, test_size=0.2)

print('Working on a 1 feature regression model using BreakPointsFaced to predict Wins')
lrm = LinearRegression()
lrm.fit(x_train, y_train)
y_predict = lrm.predict(x_test)
score_train = lrm.score(x_train, y_train)
score_test = lrm.score(x_test, y_test)
print(f'Train set score: {score_train}')
print(f'Test set score: {score_test}')
print(f'Coefficient: {lrm.coef_} (the larger the better)')

plt.scatter(x_train, y_train, label='Train set', alpha=0.4)
plt.scatter(x_test, y_test, label='Test set', alpha=0.4)
plt.plot(x_test, y_predict, 'C2', label='Model')

plt.xlabel('Break Points Faced')
plt.ylabel('Wins')
plt.legend()
plt.show()

# one feature regression model using BreakPointsFaced to predict Wins

ServiceGamesPlayed = np.array(df['ServiceGamesPlayed']).reshape(-1,1)
Wins = np.array(df['Wins']).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(ServiceGamesPlayed,Wins, test_size=0.2)

print('Working on a 1 feature regression model using ServiceGamesPlayed to predict Wins')
lrm = LinearRegression()
lrm.fit(x_train, y_train)
y_predict = lrm.predict(x_test)
score_train = lrm.score(x_train, y_train)
score_test = lrm.score(x_test, y_test)
print(f'Train set score: {score_train}')
print(f'Test set score: {score_test}')
print(f'Coefficient: {lrm.coef_} (the larger the better)')

plt.scatter(x_train, y_train, label='Train set', alpha=0.4)
plt.scatter(x_test, y_test, label='Test set', alpha=0.4)
plt.plot(x_test, y_predict, 'C2', label='Model')

plt.xlabel('Service Games Played')
plt.ylabel('Wins')
plt.legend()
plt.show()


# one feature regression model using BreakPointsOpportunities to predict Winings

BreakPointsOpportunities = np.array(df['BreakPointsOpportunities']).reshape(-1,1)
Winnings = np.array(df['Winnings']).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(BreakPointsOpportunities,Winnings, test_size=0.2)

print('Working on a 1 feature regression model using BreakPointsOpportunities to predict Winnings')
lrm = LinearRegression()
lrm.fit(x_train, y_train)
y_predict = lrm.predict(x_test)
score_train = lrm.score(x_train, y_train)
score_test = lrm.score(x_test, y_test)
print(f'Train set score: {score_train}')
print(f'Test set score: {score_test}')
print(f'Coefficient: {lrm.coef_} (the larger the better)')

plt.scatter(x_train, y_train, label='Train set', alpha=0.4)
plt.scatter(x_test, y_test, label='Test set', alpha=0.4)
plt.plot(x_test, y_predict, 'C2', label='Model')

plt.xlabel('Break Points Opportunities')
plt.ylabel('Winnings')
plt.legend()
plt.show()


## perform two feature linear regressions here:

# two feature regression model using BreakPointsOpportunities and BreakPointsFaced and to predict Winings

X = df[['BreakPointsOpportunities', 'BreakPointsFaced']]
Y = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

print('Working on a 2 feature regression model using BreakPointsOpportunities and BreakPointsFaced to predict Winnings')
lrm = LinearRegression()
lrm.fit(x_train, y_train)
y_predict = lrm.predict(x_test)
score_train = lrm.score(x_train, y_train)
score_test = lrm.score(x_test, y_test)
print(f'Train set score: {score_train}')
print(f'Test set score: {score_test}')
print(f'Coefficient: {lrm.coef_} (the larger the better)')

# two feature regression model using DoubleFaults and Aces and to predict Winings

X = df[['DoubleFaults', 'Aces']]
Y = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

print('Working on a 2 feature regression model using DoubleFaults and Aces to predict Winnings')
lrm = LinearRegression()
lrm.fit(x_train, y_train)
y_predict = lrm.predict(x_test)
score_train = lrm.score(x_train, y_train)
score_test = lrm.score(x_test, y_test)
print(f'Train set score: {score_train}')
print(f'Test set score: {score_test}')
print(f'Coefficient: {lrm.coef_} (the larger the better)')


# two feature regression model using ReturnGamesPlayed and Aces and to predict Winings

X = df[['ReturnGamesPlayed', 'ServiceGamesPlayed']]
Y = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

print('Working on a 2 feature regression model using ReturnGamesPlayed and ServiceGamesPlayed to predict Winnings')
lrm = LinearRegression()
lrm.fit(x_train, y_train)
y_predict = lrm.predict(x_test)
score_train = lrm.score(x_train, y_train)
score_test = lrm.score(x_test, y_test)
print(f'Train set score: {score_train}')
print(f'Test set score: {score_test}')
print(f'Coefficient: {lrm.coef_} (the larger the better)')

## perform multiple feature linear regressions here:

X = df[['BreakPointsOpportunities','DoubleFaults','Aces','ReturnGamesPlayed', 'ServiceGamesPlayed']]
Y = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

print('Working on a multiple feature regression model to predict Winnings')
lrm = LinearRegression()
lrm.fit(x_train, y_train)
y_predict = lrm.predict(x_test)
score_train = lrm.score(x_train, y_train)
score_test = lrm.score(x_test, y_test)
print(f'Train set score: {score_train}')
print(f'Test set score: {score_test}')
print(f'Coefficient: {lrm.coef_} (the larger the better)')