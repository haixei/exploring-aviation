from eda import data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from lightgbm import LGBMRegressor
import pandas as pd
from math import sqrt

# Count unique values
print(data.nunique())

# Drop columns that don't contribute a lot of information or pollute the data set
data = data.drop(['ACTUAL_ELAPSED_TIME', 'DEP_DELAY', 'WHEELS_OFF', 'WHEELS_ON', 'ORIGIN', 'DEST'], axis=1)
data['DELAY_LEVEL'] = data['DELAY_LEVEL'].map({'SMALL': 1, 'MEDIUM': 2, 'BIG': 3})
print(data['DELAY_LEVEL'].head())

# Label encoding caterogical columns
data_without_target = data.drop(['DELAY_LEVEL'], axis=1)
delay_level = data['DELAY_LEVEL']
data_without_target = pd.get_dummies(data_without_target)
print('Encoded data:', data_without_target.head())

# Concat the dataframe back together, we need the month column to select right target
data = pd.concat([data_without_target, delay_level], axis=1)

# Scale the data
transformer = StandardScaler()
transformer.fit(data)
transformer.transform(data)

# Preparing the data's train and test variables
data_train = data[data['MONTH'] == 5]
y_train = data['DELAY_LEVEL']
X_train = data.drop(['DELAY_LEVEL'], axis=1)

data_test = data[data['MONTH'] == 6]
data_percentage = data_test.head(int(round(data_test.shape[0] * 0.5)))
y_test = data_percentage['DELAY_LEVEL']
X_test = data_percentage.drop(['DELAY_LEVEL'], axis=1)

# Droppig the month column since it won't contribute to the predictions
X_test = X_test.drop(['MONTH'], axis=1)
X_train = X_train.drop(['MONTH'], axis=1)


# Defining error metrics
def rmse(y, y_pred):
    return sqrt(mean_squared_error(y, y_pred))


def cv_rmse(model, X, y, cv):
    rmse = sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv))
    return rmse


# Define the model
lgb = LGBMRegressor(
    num_leaves=27,
    learning_rate=0.1,
    n_estimators=400,
    max_depth=5,
    max_bin=200,
    boosting_type='goss'
)

# Grid search for the best parameters
param_grid = {
    'num_leaves': [15, 27],
    'max_bin': [200, 150],
    'n_estimators': [400, 1000]
}

# Cross validation and fitting the model
cv = KFold(n_splits=5, shuffle=True, random_state=42).split(X=X_train, y=y_train)
gsearch = GridSearchCV(estimator=lgb, param_grid=param_grid, cv=cv)
lgb_model = gsearch.fit(X=X_train, y=y_train)

# Test the model
y_pred = lgb_model.predict(X_test)

print('RMSE:', rmse(y_test, y_pred))
print('Cross Val. RMSE:', cv_rmse(lgb_model, X_test, y_test, cv))
