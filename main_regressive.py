# adopted code from https://www.geeksforgeeks.org/multiple-linear-regression-with-scikit-learn/
# supported by ChatGPT and GitHub Copilot

# importing modules and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

# Importing data
df_input = pd.read_excel('input/YOUR_DATA.xlsx', sheet_name='Sheet1', header=0, engine='openpyxl')
df_dist = df_input[['distance', 'europrotkm']]
df = df_input[['utilization', 'europrotkm']]
df = df.sort_values(by='europrotkm', ascending=False)  # Sorting by europrotkm

# Creating feature variables
X = df[['utilization']].values
X_transformed = 1 / X  # Transforming X using the reciprocal

y = df['europrotkm'].values

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.3, random_state=101)

# Fitting a linear regression model to the transformed data
model = LinearRegression()
model.fit(X_train, y_train)

# making predictions
predictions = model.predict(X_test)

# model evaluation
mse = mean_squared_error(y_test, predictions)
print('mean_squared_error : ', mse)
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

# Getting the intercept and slope of the linear regression
intercept = model.intercept_
slope = model.coef_[0]
print("Intercept:", intercept)
print("Slope:", slope)

# Recovering the original parameters of the function
a = intercept
b = slope

# plotting the results
fitted_function = f"y(x) = {a:.4f} + {b:.4f} / x"
# sort X ascending
X = np.sort(X, axis=None)
predicted_y = a + b / X

plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].set_title("Cost per tkm vs Utilization")
ax[0].scatter(X, y, label="Actual")
ax[0].plot(X, predicted_y,  c="red", label=f"Predicted ${fitted_function}$, $R^2={mse:.2f}$")
ax[0].legend()
ax[0].set_xlabel('Utilization')
ax[0].set_ylabel('$€/tkm$')

ax[1].set_title("Cost per tkm vs Distance")
ax[1].scatter(df_dist['distance'], df_dist['europrotkm'], label="Actual")
ax[1].set_xlabel('Distance')
ax[1].set_ylabel('$€/tkm$')
ax[1].legend()

fig.tight_layout()
#plt.show()
plt.savefig('output/modeled_europertkm.svg')

# Writing the parameters to an output text file
output_filename = 'output/parameters.txt'
with open(output_filename, 'w') as output_file:
    output_file.write("Fitted Function Parameters:\n")
    output_file.write(f"Intercept: {intercept}\n")
    output_file.write(f"Slope: {slope}\n")
    output_file.write(f"Fitted Function: {fitted_function}\n")