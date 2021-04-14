from eda import data
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import pandas as pd
from math import sqrt

# Preparing the data
X = data.drop['ARR_DELAY']
y = data['ARR_DELAY']

# Dividing into categorical and numerical variables
cat_cols = X.select_dtypes(['object', 'category'])
num_cols = X.select_dtypes(['int64'])

# One hot encoding caterogical columns
encoded = OneHotEncoder().fit(cat_cols)
X = pd.concat([cat_cols, num_cols])

print('Encoded data:', X.head())

# Split data into test and train variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the data
transformer = RobustScaler()
transformer.fit(X)
transformer.transform(X)

# Define the models
# Define the models
lgbm = LGBMRegressor(
        num_leaves=6,
        learning_rate=0.01,
        n_estimators=1000,
        max_bin=250
)

# Defining error metrics
def rmse(y, y_pred):
    return sqrt(mean_squared_error(y, y_pred))


# Train the model
lgbm_model = lgbm.fit(X_train, y_train)

# Test the model
y_pred = lgbm_model.predict(X_test)
test_score = rmse(y_test, y_pred)
print('RMSE:', test_score)

