from preprocessing import data
import plotly.express as pltx
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# Show the preprocessed data
print(data.head())

print('Skewness:', data.skew())

# Visualise the distribution of data using a histogram
def show_dist(data, columns):
    cols = data[columns]

    # Establish a canvas for the plots
    data_col_shape = cols.shape[1]
    col_num = int(data_col_shape/2)
    multiplot_fig = make_subplots(rows=2, cols=col_num,
                                  horizontal_spacing=0.05)
    row = 1
    col = 1

    for column in cols:
        # Create the histogram figure
        new_fig = pltx.histogram(data, x=column, histfunc="count", color_discrete_sequence=['LightSteelBlue'])
        new_fig_trace = new_fig['data'][0]

        # Add new figure to the canvas
        multiplot_fig.add_trace(new_fig_trace,
                                row=row, col=col)
        # Add titles
        multiplot_fig.update_xaxes(title_text=column, row=row, col=col)

        col += 1
        if col > col_num:
            col = 1
            row = 2

    # Show the plots
    multiplot_fig.update_layout(title_text="Distribution of data")
    multiplot_fig.show()


# Selected numerical columns
show_num_cols = ['ACTUAL_ELAPSED_TIME', 'DISTANCE', 'LATE_AIRCRAFT_DELAY', 'AIR_TIME']
# >> show_dist(data, show_num_cols)

# Selected caterogical columns
show_cat_cols = ['DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'DEST', 'DIVERTED']
# >> show_dist(data, show_cat_cols)

# Bivariate distribution of features
# First I'm going to visualise the correlation between months and different types of delays
def scatter_plot(df, x_name, y_name):
    scatter_fig = pltx.scatter(df, x=x_name, y=y_name, color_continuous_scale='Teal', color=y_name)
    scatter_fig.show()


# Arrival delay and the distance
# >> scatter_plot(data, 'ARR_DELAY', 'DISTANCE')
# >> scatter_plot(data, 'DAY_OF_WEEK', 'ARR_DELAY')
# >> scatter_plot(data, 'OP_UNIQUE_CARRIER', 'ARR_DELAY')

# Percentage of airport flights in the data set
names = data['OP_UNIQUE_CARRIER'].value_counts().index.values
vals = data['OP_UNIQUE_CARRIER'].value_counts().values

fig = pltx.pie(values=vals, names=names, title='Amount of flights by airlines', color_discrete_sequence=pltx.colors.sequential.Bluyl)
# >> fig.show()

# Correlation heatmap
# Drop the year since all the data is from the same one
data = data.drop(['YEAR'], axis=1)

# Create the map
corr = data.corr()
corr_fig = pltx.imshow(corr, color_continuous_scale='Teal')
corr_fig.update_layout(title='Correlation between features')
# >> corr_fig.show()

# Create the map
corr = data.corr()
corr_fig = pltx.imshow(corr, color_continuous_scale='Teal')
corr_fig.update_layout(title='Correlation between features')
# >> corr_fig.show()

# Show the difference in amount of delays by type (small - 1, medium - 2 and large - 3 delay)
def delay_type(x):
    if x <= 5:
        return 'SMALL'
    if 5 < x < 35:
        return 'MEDIUM'
    if x >= 35:
        return 'BIG'

data['DELAY_LEVEL'] = data['DEP_DELAY'].apply(delay_type)
delay_bar_fig = pltx.histogram(data, x='OP_UNIQUE_CARRIER', color="DELAY_LEVEL", barmode='group',
                               color_discrete_sequence=pltx.colors.diverging.Tropic)
# >> delay_bar_fig.show()

# Show the delays by airline
airline_delays = {'AIRLINES': [], 'MEAN_DELAYS': [], 'CANC_COUNT': [], 'DIV_COUNT': []}
for name in names:
    airline_delays['AIRLINES'].append(name)
    # Group by airline name
    get_airline = data[data['OP_UNIQUE_CARRIER'] == name]
    amount_of_flights = get_airline['OP_UNIQUE_CARRIER'].count()
    mean_airline_delay = get_airline['ARR_DELAY'].mean()
    airline_delays['MEAN_DELAYS'].append(mean_airline_delay)

    # Append the counted flight issues
    airline_canc_count = get_airline[get_airline['CANCELLED'] == 1]['CANCELLED'].count()
    airline_div_count = get_airline[get_airline['DIVERTED'] == 1]['DIVERTED'].count()
    airline_delays['CANC_COUNT'].append(airline_canc_count)
    airline_delays['DIV_COUNT'].append(airline_div_count)

# Show the mean delay
arr_delay_bar_fig = pltx.bar(airline_delays, x='AIRLINES', y="MEAN_DELAYS",
                             color_discrete_sequence=['LightSteelBlue'])
# >> arr_delay_bar_fig.show()

# Amount of flights, cancellations and divertions by airlines
flight_issues_fig = pltx.bar(airline_delays, x="AIRLINES", y=["CANC_COUNT", "DIV_COUNT"],
                             color_discrete_map={"CANC_COUNT": "LightSteelBlue", "DIV_COUNT": "#718abd"})
# >> flight_issues_fig.show()