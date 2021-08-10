import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def drop_missing_values(data):
    return data.dropna(axis=0)


def define_model(x, y):
    model = DecisionTreeRegressor(random_state=1)
    return model.fit(x, y)


def calculate_error(predicted_prices, y):
    return mean_absolute_error(y, predicted_prices)


def main():
    data = pd.read_csv('melb_data.csv')
    # print(data.describe())
    data = drop_missing_values(data)
    # print(data.describe())
    y = data.Price
    features = ['Rooms', 'Bathroom', 'Landsize', 'Latitude', 'Longitude']
    x = data[features]
    model = define_model(x, y)
    predicted_prices = model.predict(x)
    print(calculate_error(predicted_prices, y))


if __name__ == '__main__':
    main()
