# %%
# Import necessary packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# %%
# Load the data 

dunking_data = pd.read_csv('data/dunking-data.csv')
microscopy_data = pd.read_csv('data/microscopy-data.csv')
tr1_data = pd.read_csv('data/tr-1.csv')
tr2_data = pd.read_csv('data/tr-2.csv')
tr3_data = pd.read_csv('data/tr-3.csv')

microscopy_data

# %%
import numpy as np

def calculate_L(gamma, r, t, phi, eta):
    """
    Calculate the distance that the fluid travels into the solid.

    Parameters:
    gamma: the surface tension of the liquid
    r: the radius of the capillary pore
    t: the length of time for the capillary flow to occur
    phi: the contact angle between the solid and the liquid
    eta: the dynamic viscosity of the liquid

    Returns:
    L: the distance that the fluid travels into the solid
    """
    L = np.sqrt((gamma * r * t * np.cos(phi)) / (2 * eta))
    return L

# %%
dunking_data

# %%
microscopy_data["L_washburn"] = calculate_L(microscopy_data["gamma"], microscopy_data["r"], microscopy_data["t"], microscopy_data["phi"], microscopy_data["eta"])
microscopy_data["L_washburn_residuals"] = microscopy_data["L"] - microscopy_data["L_washburn"]
microscopy_data

# %%


# %%
microscopy_data.plot(x='L', y='L_washburn_residuals', kind='scatter')
plt.xlabel('L')
plt.ylabel('Residuals')
plt.title('Residuals L_washburn vs L')

plt.axhline(y=0.0, color='black', linestyle='dotted')

plt.show()


# %%

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = microscopy_data[['gamma', 'r', 't', 'phi', 'eta']]
y = microscopy_data['L']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Create a pipeline for linear regression
linear_pipeline = Pipeline([
    ('linear', LinearRegression())
])

# Create a pipeline for random forest regression
rf_pipeline = Pipeline([
    ('random_forest', RandomForestRegressor())
])

# Create a pipeline for ElasticNet regression
elasticnet_pipeline = Pipeline([
    ('elasticnet', ElasticNet(alpha=1.0, l1_ratio=0.5))
])

# Create a pipeline for Support Vector regression
svr_pipeline = Pipeline([
    ('svr', SVR(kernel='sigmoid', C=1.0, epsilon=0.1))
])

# Create a pipeline for Decision Tree regression
tree_pipeline = Pipeline([
    ('tree', DecisionTreeRegressor(max_depth=2))
])

# Create a pipeline for Gradient Boosting regression
gbr_pipeline = Pipeline([
    ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='squared_error'))
])

# Create a pipeline for Ridge regression
ridge_pipeline = Pipeline([
    ('ridge', Ridge())
])

# Create a pipeline for Lasso regression
lasso_pipeline = Pipeline([
    ('lasso', Lasso())
])

# Create a pipeline for MLPRegressor (a simple neural network)
nn_pipeline = Pipeline([
    ('scaling', StandardScaler()),  # It's important to scale inputs for neural networks
    ('nn', MLPRegressor(hidden_layer_sizes=(500, 500), shuffle=True, activation='tanh',learning_rate="adaptive", solver='lbfgs', max_iter=1000, random_state=42))
])

# List of pipelines for ease of iteration
pipelines = [ridge_pipeline, lasso_pipeline, elasticnet_pipeline, svr_pipeline, tree_pipeline, gbr_pipeline, linear_pipeline, rf_pipeline, nn_pipeline]

# Dictionary of pipelines and regressor types for ease of reference
pipe_dict = {0: 'Ridge', 1: 'Lasso', 2: 'ElasticNet', 3: 'Support Vector', 4: 'Decision Tree', 5: 'Gradient Boosting', 6: 'Linear', 7: 'Random Forest', 8: "Neural Network"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)

# Compare accuracies
for idx, val in enumerate(pipelines):
    print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], val.score(X_test, y_test)))


# %%
from sklearn.model_selection import GridSearchCV

# Define a dictionary of hyperparameters to optimize
parameters = {
    'nn__hidden_layer_sizes': [(50,), (100,), (50,50), (50,50,50,50), (100,100), (100,100,100,100), (100,100,100,100,100)],
    'nn__activation': ['tanh', 'relu', 'logistic'],
    'nn__solver': ['sgd', 'adam'],
    'nn__alpha': [0.0001, 0.05],
    'nn__learning_rate': ['constant','adaptive'],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(nn_pipeline, parameters, n_jobs=-1, cv=3)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_parameters = grid_search.best_params_

# Print the best parameters
print(best_parameters)

# Use the best parameters to create the best model
best_nn_pipeline = grid_search.best_estimator_

# Fit the best model to the data
best_nn_pipeline.fit(X_train, y_train)

# Print the score of the best model
print('Best Neural Network pipeline test score: %.3f' % best_nn_pipeline.score(X_test, y_test))

# %%
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
# Best initial models: RF and linear 
# Tuning models: RF 


# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [None, 2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Create the Random Forest regressor
rf = RandomForestRegressor()

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Print the best parameters
print("Best parameters for Random Forest:", best_params)

best_rf = grid_search.best_estimator_
# Print the score of the best model
print('Best Random Forest pipeline test score: %.3f' % best_rf.score(X_test, y_test))

# %%
# Continue with chosen model 

# Predict the L_washburn values using the best model

microscopy_data["L_washburn_predicted"] = best_rf.predict(microscopy_data[['gamma', 'r', 't', 'phi', 'eta']])
microscopy_data["L_washburn_residuals_predicted"] = microscopy_data["L"] - microscopy_data["L_washburn_predicted"]

fig, axs = plt.subplots(1, 2, 1)
                       
# Plot the residuals
microscopy_data.plot(ax=axs[0,0], x='L_washburn_predicted', y='L_washburn_residuals_predicted', kind='scatter')
plt.xlabel('L')
plt.axhline(y=0.0, color='black', linestyle='dotted')
plt.ylabel('Residuals')
plt.title('Residuals L_pred vs L_pred')
plt.show() 

microscopy_data.plot(ax= axs[0,1], x='L_washburn_predicted', y='L', kind='scatter')
# plot x=y
plt.plot([0, 0.02], [0, 0.02], color='black', linestyle='dotted')
plt.show()

# %%
microscopy_data.plot(x='L', y='L_washburn_residuals', kind='scatter')
plt.xlabel('L')
plt.ylabel('Residuals')
plt.title('Residuals L_washburn vs L')

plt.axhline(y=0.0, color='black', linestyle='dotted')

plt.show()


# %%
# Evaluate the accuracy with metrics
from sklearn.metrics import mean_squared_error, r2_score

# Calculate the mean squared error
mse = mean_squared_error(microscopy_data["L"], microscopy_data["L_washburn_predicted"])
print("Mean Squared Error:", mse)

# Calculate the R^2 score
r2 = r2_score(microscopy_data["L"], microscopy_data["L_washburn_predicted"])
print("R^2 Score:", r2)



# %%
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluation_metrics(model, y_true, y_pred):

    """
    Calculate the mean squared error and R^2 score of the model.

    Parameters:

    model: model name
    y_true: the true values
    y_pred: the predicted values

    """

    print("############# Evaluation metrics ###############")
    print("model: "+str(model))
    # Correlation coefficient
    corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
    print("Correlation coefficient:", corr_coef)

    # Calculate the mean squared error
    mse = mean_squared_error(y_true, y_pred)
    print("Mean Squared Error:", mse)

    # Calculate the R^2 score
    r2 = r2_score(y_true, y_pred)
    print("R^2 Score:", r2)


    # Mean absolute error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    print("Mean absolute error:", mae)

    # Root mean squared error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("Root mean squared error:", rmse)

    # Relative absolute error (RAE)
    rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
    print("Relative absolute error:", rae)

    # Root relative squared error (RRSE)
    rrse = np.sqrt(np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))))
    print("Root relative squared error:", rrse, "\n")

    return

evaluation_metrics("RF_pred", microscopy_data["L"], microscopy_data["L_washburn_predicted"])
evaluation_metrics("Washburn_pred", microscopy_data["L"], microscopy_data["L_washburn"])


# %%



