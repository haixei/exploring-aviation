from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from processing import data
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np
import plotly.express as pltx

# Select a subsample of data
data = data.head(5000)
y = data['Airfare(NZ$)']
X = data.drop(['Airfare(NZ$)'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create cross-validation
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=24)

# Define a scoring system
def mse(y, y_pred):
    return mean_squared_error(y, y_pred)


# Define models
lgbm = LGBMRegressor(
        learning_rate=0.1,
        max_bin=150,
        boosting_type='goss'
)

svr = SVR(kernel='rbf')

rf = RandomForestRegressor(n_jobs=-1,
                           oob_score=True)

# Grid-search
param_grid_lgbm = {
    'num_leaves': [80],
    'max_depth': [7, 10],
    'n_estimators': [200],
    'min_data_in_leaf': [100, 300]
}

param_grid_svr = {
    'C': [0.1, 1, 100],
    'epsilon': [0.001, 0.1, 1],
    'gamma': [0.1, 1, 3]
}

param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [5, 20]
}

# Fit the models into grid
lgbm_fit = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=cv).fit(X_train, y_train)
svr_fit = GridSearchCV(estimator=lgbm, param_grid=param_grid_lgbm, cv=cv).fit(X_train, y_train)
rf_fit = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=cv).fit(X_train, y_train)

# Stacking the models
stack = StackingCVRegressor(regressors=(lgbm_fit, rf_fit, svr_fit),
                            meta_regressor=rf_fit,
                            use_features_in_secondary=True)
stack.fit(X_train, y_train)

# Blend predictions
def blended(X):
    return ((0.15 * svr_fit.predict(X)) +
            (0.10 * lgbm_fit.predict(X)) +
            (0.25 * rf_fit.predict(X)) +
            (0.50 * stack.predict(np.array(X))))


acc = {'models': [], 'acc': []}
models = {'SVR': svr_fit, 'LGBM': lgbm_fit, 'Random Forest': rf_fit, 'Stack': stack}
for key, value in models.items():
    pred = mse(y_test, value.predict(X_test))
    print('Acc. for ' + key + ' --> ' , pred)
    acc['models'].append(key)
    acc['acc'].append(pred)

# Add the blended accuracy as an addition
acc['models'].append('Blended')
blend_pred = mse(y_test, blended(X_test))
acc['acc'].append(blend_pred)
print('Acc for Blended' + ' --> ' , blend_pred)

fig_model_acc = pltx.line(x=acc['models'], y=acc['acc'], title='Accuracy of the models (MSE)')
fig_model_acc.show()

train_score = mse(y_train, blended(X_train))
print('MSE score on train data:', train_score)

# Test the model
test_score = mse(y_test, blended(X_test))
print('MSE score on test data:', test_score)
