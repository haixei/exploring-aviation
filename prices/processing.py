import pandas as pd
from sklearn.preprocessing import PowerTransformer
import plotly.express as pltx

# Load in the data
data = pd.read_csv('../data/prices/2019.csv')

# Exploring the data set properties
print(data.head())
print(data.info())
print(data.shape)

# Calculating missing data in percentages
vals_missing = round(data.isnull().mean() * 100, 2)
print(vals_missing)

# Transforming the columns

# Explore the skewness
skew = data.skew()
print('Skewness:', skew)

# Drop the unnamed columns (id)
data.drop(data.filter(regex="Unname"), axis=1, inplace=True)

# Transform the data
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_cols = data.select_dtypes(include=numerics)

pt = PowerTransformer(method='yeo-johnson')
skewed_features = []
for feature, skew in skew.items():
    if skew >= 1.5 and feature in num_cols.columns.values:
        skewed_features.append(feature)

pt = PowerTransformer()
#pt.fit(data[skewed_features])
#data[skewed_features] = pt.transform(data[skewed_features])

print('Skewness after normalization:', data.skew())

# Correlation between the features
corr = data.corr()
corr_fig = pltx.imshow(corr, color_continuous_scale='Teal')
corr_fig.update_layout(title='Correlation between features')

corr_fig.show()

# Binary encoding the features
