# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
-The dataset contains multiple features such as median income, house age, and number of rooms, which influence housing prices.

-Correlation analysis often shows that median income has the strongest positive impact on housing prices, while variables like latitude and longitude have weaker inverse relationships.

- among variables such as total rooms, total bedrooms, and households can affect model stability, requiring methods like ridge regression for improvement.

-The dataset is suitable for supervised learning models, especially multiple linear regression, decision trees, or random forest models to predict housing prices.

-Proper data preprocessing, including handling multicollinearity and verifying assumptions like independence, homoscedasticity, and normality of residuals, is essential for reliable regression models.



## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: VIJAYARAGHAVAN M
RegisterNumber:  25017872
*/
```

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the California housing data
data = fetch_california_housing()

# Select first 3 features as input
X = data.data[:, :3]

# Stack the target and 7th feature as output
Y = np.column_stack((data.target, data.data[:, 6]))

# Corrected train_test_split argument test_size (removed extra space)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale input and output
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# Initialize the SGDRegressor and wrap with MultiOutputRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(X_train, Y_train)

# Predict on test set
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform the scaled predictions and targets
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Calculate Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)

print("Mean Square Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
import matplotlib.pyplot as plt

# Assuming Y_test and Y_pred are the true and predicted values from the model

plt.figure(figsize=(12, 5))

# Plot for the first output (target housing price)
plt.subplot(1, 2, 1)
plt.scatter(Y_test[:, 0], Y_pred[:, 0], alpha=0.5)
plt.plot([Y_test[:, 0].min(), Y_test[:, 0].max()], [Y_test[:, 0].min(), Y_test[:, 0].max()], 'r--')
plt.xlabel('True Target Housing Price')
plt.ylabel('Predicted Target Housing Price')
plt.title('True vs Predicted Housing Prices')

# Plot for the second output (7th feature)
plt.subplot(1, 2, 2)
plt.scatter(Y_test[:, 1], Y_pred[:, 1], alpha=0.5)
plt.plot([Y_test[:, 1].min(), Y_test[:, 1].max()], [Y_test[:, 1].min(), Y_test[:, 1].max()], 'r--')
plt.xlabel('True 7th Feature Value')
plt.ylabel('Predicted 7th Feature Value')
plt.title('True vs Predicted 7th Feature')

plt.tight_layout()
plt.show()

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
<img width="1613" height="700" alt="Screenshot 2025-10-06 180211" src="https://github.com/user-attachments/assets/5d94c713-4013-4c65-8882-8f4384bbccfe" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
