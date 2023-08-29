# adopted code from https://www.geeksforgeeks.org/multiple-linear-regression-with-scikit-learn/
# supported by ChatGPT and GitHub Copilot

# importing modules and packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

# importing data
df_input = pd.read_excel('input/YOUR_DATA.xlsx', sheet_name='Sheet1', header=0, engine='openpyxl')
df = df_input[['utilization', 'distance', 'europrotkm']]
# sort df descending by europrotkm

#df.drop('No', inplace=True, axis=1)

print(df.head())
print(df.columns)

# creating feature variables
X = df.drop('europrotkm', axis=1).drop('distance', axis=1)
y = df['europrotkm']

print(X)
print(y)

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

# Create polynomial regression model with degree 2
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)

# fitting the model
model.fit(X_train, y_train)

# making predictions
predictions = model.predict(X_test)

# model evaluation
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

# plotting the results
predict_all = model.predict(X)
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
ax.set_title("Utilization vs europertkm")
ax.scatter(X['utilization'], y, label="Actual")
ax.scatter(X['utilization'], predict_all, marker=".", c="red", label="Predicted")
ax.legend()
ax.set_xlabel('Utilization')
ax.set_ylabel('$€/tkm$')
fig.tight_layout()
plt.show()

# fitted function
# Getting the intercept and coefficients
coefficients = model.named_steps['linearregression'].coef_
intercept = model.named_steps['linearregression'].intercept_

# Printing the intercept and coefficients
print("Intercept:", intercept)
print("Coefficients:", coefficients)