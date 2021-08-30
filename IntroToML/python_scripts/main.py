import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def mae(train_x, val_x, train_y, val_y, max_leaf_nodes=10):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_x, train_y)
    prediction = model.predict(val_x)
    return mean_absolute_error(val_y, prediction)


def drop_columns_with_missing_data(x_train, x_val, y_train, y_val):
    columns_with_missing_data = [col for col in x_train.columns
                                 if x_train[col].isnull().any()]
    reduced_x_train = x_train.drop(columns_with_missing_data, axis=1)
    reduced_x_valid = x_train.drop(columns_with_missing_data, axis=1)
    print(mae(reduced_x_train, reduced_x_valid, y_train, y_val))


if __name__ == "__main__":
    path = os.getcwd()
    data = pd.read_csv(path + '/melb_data.csv')
    y = data.Price
    features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']  # features.
    x = data[features]
    train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)
    # print(mae(train_x,val_x,train_y,val_y))
    print(drop_columns_with_missing_data(train_x, val_x, train_y, val_y))
