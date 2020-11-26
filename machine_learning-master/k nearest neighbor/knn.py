import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

LE = preprocessing.LabelEncoder()
buying = LE.fit_transform(list(data["buying"]))
maint = LE.fit_transform(list(data["maint"]))
door = LE.fit_transform(list(data["door"]))
persons = LE.fit_transform(list(data["persons"]))
lug_boot = LE.fit_transform(list(data["lug_boot"]))
safety = LE.fit_transform(list(data["safety"]))
clss = LE.fit_transform(list(data["class"]))

prediction = "class"

x = list(zip(buying,maint, door, persons, lug_boot, safety))
y = list(clss)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=10)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]
for x in range(len(x_test)):
    print("predicted:", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    m = model.kneighbors([x_test[x]], 9, True)
    print("M:", m)