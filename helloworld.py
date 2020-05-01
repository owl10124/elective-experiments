from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()
# print("Target names: {}".format(iris_dataset["target_names"]))
# print("Feature names: {}".format(iris_dataset["feature_names"]))
# print("Data type: {}".format(type(iris_dataset["data"])))
# print("Data shape: {}".format(iris_dataset["data"].shape))
# print("Data: {}".format(iris_dataset["data"]))

# print ("Target type: {}".format(type(iris_dataset["target"])))
# print ("Target shape: {}".format(iris_dataset["target"].shape))
# print ("Target: {}".format(iris_dataset["target"]))

x_train, x_test, y_train, y_test = train_test_split(iris_dataset["data"],iris_dataset["target"],random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])

print(knn.predict(x_new))
print(iris_dataset["target_names"][knn.predict(x_new)[0]])

y_pred = knn.predict(x_test)
print("Mean score: {:.2f}".format(np.mean(y_pred==y_test)))

print("Test set score: {:.2f}".format(knn.score(x_test,y_test)))

# taken from medium article: https://medium.com/gft-engineering/start-to-learn-machine-learning-with-the-iris-flower-classification-challenge-4859a920e5e3