from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.1)

for i in range(len(iris.target_names)):
    print("Label",i, "-", str(iris.target_names[i]))

classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print("Classification using KNN with K=2")

for i in range(0, len(x_test)):
    print('Sample: ',str(x_test[i]), 'Actual Label: ', str(y_test[i]), 'Predicted Label: ', str(y_pred[i]) )

print("Classification accuracy: ", classifier.score(x_test, y_test))
