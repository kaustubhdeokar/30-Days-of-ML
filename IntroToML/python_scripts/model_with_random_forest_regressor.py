import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def drop_missing_values(data):
    return data.dropna(axis=0)


def define_model(x, y):
    model = RandomForestRegressor(random_state=1)
    return model.fit(x, y)


def calculate_error(predicted_prices, y):
    return mean_absolute_error(y, predicted_prices)


def calculate_score(train_x, val_x, train_y, val_y):
    model = define_model(train_x, train_y)
    predicted_prices = model.predict(val_x)
    return calculate_error(predicted_prices, val_y)


def main():
    data = pd.read_csv('melb_data.csv')
    # print(data.describe())
    data = drop_missing_values(data)
    # print(data.describe())
    y = data.Price
    features = ['Rooms', 'Bathroom', 'Landsize', 'Latitude', 'Longitude']
    x = data[features]

    train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)
    print(calculate_score(train_x, val_x, train_y, val_y))


if __name__ == '__main__':
    main()
