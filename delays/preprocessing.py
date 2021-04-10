import pandas as pd
import plotly.express as pltx
from plotly.subplots import make_subplots

# Load and merge the data sets
loaded_df = []
for i in range(11):
    i = i + 1
    if i < 10:
        i = str(i)
        i = '0' + i
    month = pd.read_csv(f'../data/delays/2019/{i}.csv')
    loaded_df.append(month)

data = pd.concat(loaded_df)

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
# >> print(data.columns.values)

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
# Early arrivals and dept. delay show negative numbers so we don't  check for arrival time
# I'm only going trough the columns that should not have any nergative values
scheck_cols = ['ARR_TIME', 'TAXI_IN', 'DEP_TIME', 'TAXI_OUT', 'DISTANCE']
for col in scheck_cols:
    sanity = (data[col] > 0).all()

print('Columns are sane: ', sanity)

# Correlation between features
corr = data.corr()
corr_fig = pltx.imshow(corr, color_continuous_scale='Teal')
corr_fig.update_layout(title='Correlation between features')
# >> corr_fig.show()

# Drop unnecessary columns, with high correlation or just not contributing
# a lot of useful information
data.drop(['YEAR', 'ACTUAL_ELAPSED_TIME', 'DISTANCE', 'DEP_DELAY'], axis=1)

# Check for typos in caterogical variables
num_cols = data._get_numeric_data().columns
cols = data.columns
cat_cols = list(set(cols) - set(num_cols))

for col in cat_cols:
    if col in ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']:
        print(data[col].value_counts())
        