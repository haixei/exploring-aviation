import pandas as pd
from sklearn.preprocessing import PowerTransformer, LabelBinarizer, LabelEncoder, RobustScaler
import plotly.express as pltx
import re

# Load in the data
data = pd.read_csv('../data/prices/2019.csv')

# Exploring the data set properties
print(data.head())
print(data.info())
print(data.shape)

print(data['Transit'])

# Calculating missing data in percentages
print(round(data.isnull().mean() * 100, 2))

# Explore the skewness
print('Skewness:', data.skew())

# Drop the unnamed columns (id)
data.drop(data.filter(regex="Unname"), axis=1, inplace=True)

# Correlation between the features
corr = data.corr()
corr_fig = pltx.imshow(corr, color_continuous_scale='Teal')
corr_fig.update_layout(title='Correlation between features')

# corr_fig.show()

# Handling the missing values
data = data.drop(['Baggage'], axis=1)
data = data.dropna(subset=['Dep. airport', 'Arr. airport', 'Airline'])

# Dividing transit into two columns
data[['TransitTime', 'TransitPlace']] = data['Transit'].str.split(' in ', expand=True)
print(data.tail())

# Transform to transit time to minutes
data['TransitTime'] = data['TransitTime'].fillna(0)
data['TransitPlace'] = data['TransitPlace'].fillna('None')

def transformDuration(x):
    if x != 0:
        x_list = re.findall(r'\d+', x)
        # If we have more than two elements in the list, transform both hours and minutes
        # Otherwise only transform hours
        if len(x_list) > 1:
            x = (int(x_list[0]) * 60) + int(x_list[1])
        elif len(x_list) < 1 and 'h' in x:
            x = int(x_list[0]) * 60
        else:
            x = int(x_list[0])
    return x


data['TransitTime'] = data['TransitTime'].apply(transformDuration)
data['Duration'] = data['Duration'].apply(transformDuration)
data = data.drop(['Transit'], axis=1)

# Transform the other columns
def transformTime(x):
    x = x[:-3].split(':')
    return int(''.join(map(str, x)))

data['Dep. time'] = data['Dep. time'].apply(transformTime)
data['Arr. time'] = data['Arr. time'].apply(transformTime)

# Binary Encoding the multi-class columns

lb = LabelBinarizer()
cat_cols = ['Dep. airport', 'Arr. airport', 'Airline', 'TransitPlace']
for i in range(len(cat_cols)):
    cat = cat_cols[i]
    encoded_results = lb.fit_transform(data[cat].astype(str))

    # Change the name so the columns do not repeat
    classes = []
    for i in range(len(lb.classes_)):
        name = ''
        if cat == 'Dep. airport':
            name = ' Dep'
        elif cat == 'Airline':
            name = ' Air'
        elif cat == 'TransitPlace':
            name = ' Transit'
        elif cat == 'Arr. airport':
            name = ' Arr'

        classes.append(lb.classes_[i] + name)

    data_encoded = pd.DataFrame(encoded_results, columns=classes)
    data = pd.concat([data, data_encoded], axis=1).drop([cat], axis=1)

print(data.columns.values)

# Label Encoding the Direct column
le = LabelEncoder()
data['Direct'] = le.fit_transform(data['Direct'])

# Transform the date column to two columns
data[['Day', 'Month', 'Year']] = data['Travel Date'].str.split('/', expand=True)
data = data.drop(['nan Arr', 'Travel Date', 'nan Transit', 'nan Air', 'Year'], axis=1)
print(data.columns.values)
print(data.head())

# Save col names
col_names = list(data.columns)

# Normalizing the data
pt = PowerTransformer()
data = pd.DataFrame(pt.fit_transform(data))

# Scale the data
rs = RobustScaler()
data = pd.DataFrame(rs.fit_transform(data), columns=col_names)

# Exploring results
print('Skewness after normalization:', data.skew())
print(data.tail())
