import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1) 
Y = np.array([4, 8, 12, 16])

model = LinearRegression()
model.fit(X, Y)
slope_sklearn = model.coef_[0]
intercept_sklearn = model.intercept_
print(f"Slope (a): {slope_sklearn:.2f}")
print(f"Intercept (b): {intercept_sklearn:.2f}")
X_new = np.array([[2.5]])
Y_prediction = model.predict(X_new)
print(f"Prediction for 2.5 tons: {Y_prediction[0]:.2f} L/100 km")
