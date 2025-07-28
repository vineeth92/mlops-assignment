import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

model = joblib.load("models/sklearn_model.joblib")
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

y_pred = model.predict(X_test)
print("Prediction successful. R² Score:", r2_score(y_test, y_pred))

