import pickle 
import cv2
import os
import numpy as np

DATADIR = "files"
CATEGORIES = ["3"] #0=black/brown 1=black
tests = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_ANYCOLOR)
                img_array = cv2.resize(img_array,(60,60))
                tests.append([img_array, class_num])
            except Exception as e:
                pass

create_training_data()

with open("model_pickle", "rb") as f:
    model_loaded = pickle.load(f)

test_final = []

for features, label in tests:
    test_final.append(features)


test_final = np.array(test_final).reshape(-1,60,60,3)
test_final = np.array(test_final)
result = model_loaded.predict(test_final)
for x in result:
    for y in x:
        final_result = y
        print(final_result)

if final_result >= 0.500000:
    print("BLONDE")
elif final_result < 0.500000:
    print("BRUNETTE")

