import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('melb_data.csv')
data = data.dropna(axis=0)

y = data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'] # features.
x = data[features]

#predict

model = DecisionTreeRegressor(random_state=1)
model.fit(x,y)
predictions = model.predict(x)


#validation

print(mean_absolute_error(y,predictions))

#splitting test and training data

train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=0)
model2 = DecisionTreeRegressor(random_state=0)
model2.fit(train_x,train_y)
val_predictions = model2.predict(val_x)
print(mean_absolute_error(val_y,val_predictions))


# figuring out the best output using number of leaves

def get_mae(leaf_nodes,train_x,val_x,train_y,val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = leaf_nodes,random_state=0)
    model.fit(train_x,train_y)
    preds = model.predict(val_x)
    return mean_absolute_error(val_y,preds)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
results = []
for i in candidate_max_leaf_nodes:
    results.append(get_mae(i,train_x,val_x,train_y,val_y))
print(results)


#Using random forest instead of decision tree.

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_x,train_y)
preds = forest_model.predict(val_x)
print(mean_absolute_error(val_y,preds))
