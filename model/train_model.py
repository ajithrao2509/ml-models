"""Train and save an iris classification model using Random Forest."""

import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/iris_model.pkl")
print("✅ Model saved to model/iris_model.pkl")
