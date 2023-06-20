import pandas as pd
import numpy as np

import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("car.data")
# print(data.head())

encode = preprocessing.LabelEncoder()

buying = encode.fit_transform(list(data["buying"]))
maint = encode.fit_transform(list(data["maint"]))
door = encode.fit_transform(list(data["door"]))
persons = encode.fit_transform(list(data["persons"]))
lug_boot = encode.fit_transform(list(data["lug_boot"]))
safety = encode.fit_transform(list(data["safety"]))
cls = encode.fit_transform(list(data["class"]))
# print(buying)

predict = "cls"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)

prediction = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(prediction)):
    print("Predicted: ", names[prediction[x]], "Actual: ", names[y_test[x]])
    # n = model.kneighbors([x_test[x]], 7, True)
    # print(n)