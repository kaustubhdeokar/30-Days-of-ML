import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def drop_missing_values(data):
    return data.dropna(axis=0)


def define_model(max_leaf_nodes, x, y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    return model.fit(x, y)


def calculate_error(predicted_prices, y):
    return mean_absolute_error(y, predicted_prices)


def calculate_score(max_leaf_node, train_x, val_x, train_y, val_y):
    model = define_model(max_leaf_node, train_x, train_y)
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
    max_leaf_nodes = [5, 50, 100, 500]
    for leaf_nodes in max_leaf_nodes:
        print(calculate_score(leaf_nodes, train_x, val_x, train_y, val_y))


if __name__ == '__main__':
    main()
