"""
Capturing patterns from data is called training/fitting.
Data used to train the model is called training data.
After model has been trained we can use it to predict the prices of additional houses.

Using the same model to test the data and train the data, the calculated score is called as the in-sample score.



"""

import pandas as pd

data = pd.read_csv('melb_data.csv')

print(data.describe())