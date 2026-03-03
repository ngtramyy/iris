from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

iris = load_iris()
X = iris.data
y = iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))

print("Model saved successfully!")