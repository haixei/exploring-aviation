import pandas as pd
from sklearn.preprocessing import PowerTransformer
from numpy import log2
import plotly.express as pltx

# Load and merge the data sets
m05 = pd.read_csv('../data/delays/2019/05.csv')
m06 = pd.read_csv('../data/delays/2019/06.csv')

data = pd.concat([m05, m06])

data.drop(data.filter(regex="Unname"), axis=1, inplace=True)
print(data.head())

# Exploring the data set more in depth
print(data.info())

# Making sure the names are correct
column_names = list(data.columns.values)
column_new_names = {}
for column in column_names:
    column_new_names[column] = column.strip()

data = data.rename(columns=column_new_names)
print(data.columns.values)

# Missing values in percentages
vals_missing = round(data.isnull().mean() * 100, 2)
print(vals_missing)

# Handling missing values
data.drop(['CANCELLATION_CODE'], axis=1, inplace=True)

# Filling missing delays with 0s since they simply didn't happen
delay_causes = []
for value in data.columns.values:
    if '_DELAY' in value:
        delay_causes.append(value)

data[delay_causes].fillna(0)

# Using approximation to fill other columns
for i, v in vals_missing.iteritems():
    if v > 0 and i != 'CANCELLATION_CODE':
        data[i] = data[i].interpolate(method='linear', limit_direction='both', axis=0)

print(round(data.isnull().mean() * 100, 2))

# Sanity check
sanity = True
# Early arrivals and dept. delay show negative numbers so we don't check for arrival time
# I'm only going trough the columns that should not have any negative values
scheck_cols = ['ARR_TIME', 'TAXI_IN', 'DEP_TIME', 'TAXI_OUT', 'DISTANCE']
for col in scheck_cols:
    sanity = (data[col] > 0).all()

print('Columns are sane: ', sanity)

# Check for typos in caterogical variables
num_cols = data._get_numeric_data().columns
cols = data.columns
cat_cols = list(set(cols) - set(num_cols))

for col in cat_cols:
    if col in ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']:
        print(data[col].value_counts())


# Parallel between the features
data['MONEY_LOST'] = data['ARR_DELAY']
data['MONEY_LOST'] = data.apply(lambda row: (abs(row['LATE_AIRCRAFT_DELAY'] * 25.26
                                             + row['ARR_DELAY'] * 24.55)),
                                axis=1)

data['MONEY_LOST'] = data['MONEY_LOST'].apply(log2)
data['ARR_DELAY_ONLY'] = [0 if arr <= 0 else arr for arr in data['ARR_DELAY']]
data['MONEY_LOST'] = [0 if money <= 0 else money for money in data['MONEY_LOST']]
data['ARR_DELAY_ONLY'] = data['ARR_DELAY_ONLY'].apply(log2)

fig_info = pltx.scatter(data, x="ARR_DELAY_ONLY", y="MONEY_LOST", color="OP_UNIQUE_CARRIER",
                              log_x=True)
fig_info.update_traces(marker=dict(size=3))
fig_info.show()

# Show density
fig_density = pltx.density_contour(data.head(60000), x="DISTANCE", y="MONEY_LOST", color="OP_UNIQUE_CARRIER", marginal_x="rug", marginal_y="histogram")
# >> fig_density.show()

# Show density heatmap for cities
fig_city = pltx.density_heatmap(data.head(30000), x="ORIGIN", y="DEST", marginal_y="histogram")
# >> fig_city.show()

# Explore the skewness
skew = data.skew()
print('Skewness:', skew)

# Fix using a yj transformation
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_cols = data.select_dtypes(include=numerics)

pt = PowerTransformer(method='yeo-johnson')
skewed_features = []
for feature, skew in skew.items():
    if skew >= 1.5 and feature in num_cols.columns.values and feature != 'YEAR':
        skewed_features.append(feature)

pt = PowerTransformer()
pt.fit(data[skewed_features])
data[skewed_features] = pt.transform(data[skewed_features])

print('Skewness after normalization:', data.skew())
