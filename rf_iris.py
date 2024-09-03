import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

#load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

#train the random forest classifie
model = RandomForestClassifier()
model.fit(X,y)

# save the trained model
joblib.dump(model, "model.joblibs")