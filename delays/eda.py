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
show_dist(data, show_num_cols)

# Selected caterogical columns
show_cat_cols = ['DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'DEST', 'DIVERTED']
show_dist(data, show_cat_cols)
