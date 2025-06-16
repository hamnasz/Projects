import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("/workspaces/Projects/Regression/data.csv")

print(data.shape)
print(data.columns)
print(data.tail())
print(data.isnull().any())
print(data.describe())
print(data.info())

data_type = pd.DataFrame(data.dtypes).T.rename({0: 'Column Data Type'})
null_value = pd.DataFrame(data.isnull().sum()).T.rename({0: 'Null Values'})
data_info = pd.concat([data_type, null_value], axis=0)
print(data_info)

print(data.nunique())
print(data.head())

corr = data.select_dtypes(include=[np.number]).corr()  # Only numeric columns
plt.figure(figsize=(12, 10))
sns.heatmap(data=corr, annot=True, cmap='Spectral').set(title="Correlation Matrix")
plt.show()

corr_matrix = corr.round(2)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, center=0, vmin=-1, vmax=1, mask=mask, annot=True, cmap='BrBG')
plt.show()

X = data[['GrLivArea']]
Y = data[['SalePrice']]

print(X.head())
print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1)

print(X.shape)
print(Y.shape)
print()
print(X_train.shape)
print(Y_train.shape)
print()
print(X_test.shape)
print(Y_test.shape)

model = LinearRegression()
model.fit(X_train, Y_train)

model_coef = model.coef_
print("Model Coefficient:", model_coef.round(2))
model_intercept = model.intercept_
print("Model Intercept:", model_intercept.round(2))

new_GrLivArea = np.array([1500]).reshape(-1, 1)
prediction = model.predict(new_GrLivArea).round(2)
print("Predicted SalePrice for GrLivArea=1500:", prediction)

equation_predict = (model_coef * new_GrLivArea) + model_intercept
print("Equation Prediction:", equation_predict.round(2))

y_test_pred = model.predict(X_test)

print("Actual SalePrice (first 5):")
print(Y_test[:5].values)
print()
print("Predicted SalePrice (first 5):")
print(y_test_pred[:5].round(2))

plt.scatter(X_test, Y_test, label='Test Data', color='k')
plt.plot(X_test, y_test_pred, label='Predicted Data', color='b', linewidth=3)
plt.xlabel('GrLivArea (sq ft)')
plt.ylabel('SalePrice ($)')
plt.title('Model Evaluation: SalePrice vs GrLivArea')
plt.legend(loc='upper left')
plt.show()

mse = round(mean_squared_error(Y_test, y_test_pred), 2)
print("Mean Squared Error:", mse)

diff = (Y_test - Y_test.mean())
print("Difference from Mean (first 5):")
print(diff.head())