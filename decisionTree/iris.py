# Source: https://en.wikipedia.org/wiki/Iris_flower_data_set
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
# Printing the iris target names
print(list(iris.target_names))

# Creating the classifier
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(iris.data, iris.target)
# You will notice the numerical output will switch between 1 and 0.
# This is because the decision tree is making the best comparison,
# thus resulting in probabilistic behavior.
print(classifier.predict([[5.1, 3.5, 1.4, 1.5]]))