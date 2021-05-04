from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from processing import data
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np

# Select a subsample of data
data = data.head(5000)
y = data['Airfare(NZ$)']
X = data.drop(['Airfare(NZ$)'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create cross-validation
cv = RepeatedKFold(n_splits=12, n_repeats=4, random_state=24)

# Define a scoring system
def mse(y, y_pred):
    return mean_squared_error(y, y_pred)


def cv_mse(model, X, y):
    rmse = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    return rmse

# Define models
lgbm = LGBMRegressor(
        num_leaves=6,
        learning_rate=0.01,
        n_estimators=1000,
        max_bin=250
)

svr = SVR(kernel='rbf', epsilon=0.008)

rf = RandomForestRegressor(max_depth=15,
                           min_samples_split=5,
                           min_samples_leaf=5,
                           max_features=None,
                           oob_score=True)

# Grid-search
param_grid_lgbm = {
    'num_leaves': [15, 27],
    'max_bin': [200, 150],
    'n_estimators': [400, 1000]
}

param_grid_svr = {
    'C': [0.1, 1, 100],
    'epsilon': [0.001, 0.1, 1],
    'gamma': [0.1, 1, 3]
}

param_grid_rf = {
    'n_estimators': [200, 500],
    'max_depth': [4, 10]
}

# Fit the models into grid
svr_fit = GridSearchCV(estimator=lgbm, param_grid=param_grid_lgbm, cv=cv).fit(X_train, y_train)
rf_fit = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=cv).fit(X_train, y_train)
lgbm_fit = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=cv).fit(X_train, y_train)

# Stacking the models
stack = StackingCVRegressor(regressors=(lgbm_fit, rf_fit, svr_fit),
                            meta_regressor=lgbm_fit,
                            use_features_in_secondary=True)
stack.fit(X_train, y_train)

# Blend predictions
def blended(X):
    return ((0.15 * svr_fit.predict(X)) +
            (0.25 * lgbm_fit.predict(X)) +
            (0.25 * rf_fit.predict(X)) +
            (0.35 * stack.predict(np.array(X))))

train_score = mse(y_train, blended(X_train))
print('MSE score on train data:', train_score)

# Test the model
test_score = mse(y_test, blended(X_test))
print('MSE score on test data:', test_score)