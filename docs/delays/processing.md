# 1.1 Pre-processing the data
Before going into any kind of data exploration we have to make sure that the information is structured, cleaned and processed properly. First I'm going to load and merge two quite sizable datasets from May and June (2019). The reason of why I decided to merge these two is that I want to have more data over time since my predictions will be ran over the data from July. Two months of information can provide more insight into patterns. I'm also filtering out the unnamed columns that I won't use in data analysis but which were still provided in the data set.

Before moving forward with the analysis I want to display a short info on the dataset that might help me save some time.

```
Int64Index: 1273081 entries, 0 to 636690
Data columns (total 25 columns):
 #   Column               Non-Null Count    Dtype  
---  ------               --------------    -----  
 0   YEAR                 1273081 non-null  int64  
 1   MONTH                1273081 non-null  int64  
 2   DAY_OF_WEEK          1273081 non-null  int64  
 3   OP_UNIQUE_CARRIER    1273081 non-null  object 
 4   ORIGIN               1273081 non-null  object 
 5   DEST                 1273081 non-null  object 
 6   DEP_TIME             1247955 non-null  float64
 7   DEP_DELAY            1247950 non-null  float64
 8   TAXI_OUT             1246980 non-null  float64
 9   WHEELS_OFF           1246980 non-null  float64
 10  WHEELS_ON            1246399 non-null  float64
 11  TAXI_IN              1246399 non-null  float64
 12  ARR_TIME             1246399 non-null  float64
 13  ARR_DELAY            1242226 non-null  float64
 14  CANCELLED            1273081 non-null  float64
 15  CANCELLATION_CODE    26239 non-null    object 
 16  DIVERTED             1273081 non-null  float64
 17  ACTUAL_ELAPSED_TIME  1242226 non-null  float64
 18  AIR_TIME             1242226 non-null  float64
 19  DISTANCE             1273081 non-null  float64
 20  CARRIER_DELAY        279641 non-null   float64
 21  WEATHER_DELAY        279641 non-null   float64
 22  NAS_DELAY            279641 non-null   float64
 23  SECURITY_DELAY       279641 non-null   float64
 24  LATE_AIRCRAFT_DELAY  279641 non-null   float64
dtypes: float64(18), int64(3), object(4)
```

From this information I can already tell that there's a few columns that should have grouped values and we should make sure that there's no misspells that could potentially create new groups and skew the results. There also seem to be a significant amount of null values in some places, it would be good to give that a more in-depth look. The last thing is that we work with both numerical and categorical values so before creating models we must make sure to transform it with label encoding.

## Handling missing values

Before filling anything I want to see if there's any kind of correlation between missing values and what percentages of the values are missing.

```
YEAR                    0.00
MONTH                   0.00
DAY_OF_WEEK             0.00
OP_UNIQUE_CARRIER       0.00
ORIGIN                  0.00
DEST                    0.00
DEP_TIME                1.97
DEP_DELAY               1.97
TAXI_OUT                2.05
WHEELS_OFF              2.05
WHEELS_ON               2.10
TAXI_IN                 2.10
ARR_TIME                2.10
ARR_DELAY               2.42
CANCELLED               0.00
CANCELLATION_CODE      97.94
DIVERTED                0.00
ACTUAL_ELAPSED_TIME     2.42
AIR_TIME                2.42
DISTANCE                0.00
CARRIER_DELAY          78.03
WEATHER_DELAY          78.03
NAS_DELAY              78.03
SECURITY_DELAY         78.03
LATE_AIRCRAFT_DELAY    78.03
```

It looks like there's a significant amount of missing values in the cancellation code and I'm afraid the column won't contribute in any way to the predictions so I'm simply going to drop it. ACTUAL_ELAPSED_TIME and AIR_TIME have the same amount of missing values which makes sense, they both describe how long the plane was in the air. In cases when a flight didn't happen, there's no reason for the row to have this value so it's safe to assume that filling these missing values with 0s (minutes) will make sense. I also want to bring your attention to the fact that the value of missing values in AIR_TIME seems to be almost equal to the amount of flights that had CANCELLATION_CODE which makes me think that these flights were in fact - cancelled. The rest of them might come from other places or be simply not filled, but I don't think the latter would influence the model a lot.

In _DELAY columns, the existing values add up to ~100% and that, first of all, lets us know that the dataset doesn't consist of flights that weren't delayed, and second, that the missing values are simply lack of delay so they are also safe to fill in with 0s (minutes).

The other columns have a small number of missing values and I will fill them in using interpolation. After this action, all the columns have 0% missing values and we can proceed to the next step.

## Sanity check

To make sure that our columns contain correct information (one that doesn't negate its purpose) I'm going to sanitize the numerical columns that should be positive. Let's keep in mind that some of the numerical columns should contain negative values like early arrivals and departments. 

```
Columns are sane:  True
```



## Check for typos in categorical features

As I said before, there's a few columns in the dataset of which the values inside should be grouped together by name. Knowing that we do not want to create new groups by accident since it could negatively impact our data exploration later and the prediction model.

```
Name: OP_UNIQUE_CARRIER, dtype: int64
ATL    68782
ORD    58964
DFW    53874
DEN    43512
CLT    39741
       ...  
```

I will only show a slice of the output here, but the returned information confirmed that there's no issues with the names.



## Skewness

When it comes to skewness, I have a very clear data processing direction in mind. First of all, I want to see how skewed the columns are. If the result is worrying enough I will try to transform the data using the Yeo Johnson method (since it works on both negative and positive values).

```
Skewness:
YEAR                     0.000000
MONTH                   -0.000473
DAY_OF_WEEK             -0.000456
DEP_TIME                 0.017437
DEP_DELAY                8.280913
TAXI_OUT                 3.606932
WHEELS_OFF              -0.020288
WHEELS_ON               -0.351076
TAXI_IN                  5.839490
ARR_TIME                -0.385604
ARR_DELAY                7.393070
CANCELLED                6.748319
DIVERTED                16.516710
ACTUAL_ELAPSED_TIME      1.420727
AIR_TIME                 1.450716
DISTANCE                 1.508514
CARRIER_DELAY            8.268453
WEATHER_DELAY           16.983710
NAS_DELAY                8.860583
SECURITY_DELAY         126.466440
LATE_AIRCRAFT_DELAY      4.786374

```

The data set seems to be fairly good with the skewness aside of a few selected features. SECURITY_DELAY has an absolutely ridiculous skewness, other _DELAY features seem to be a little bit over the top as well. Aside of these, I also have my eye on DIVERTED and CANCELLED flights. After analysing the output I came to the conclusion that it would be efficient to transform all columns with skewness >1.5. It's a quite large data set so I'm also looking at the efficiency vs time spend ratio. I think not transforming features where the skewness is <1.5 could save us some time and not interfere with the predictions too much.

After the normalization the result looks like the following:

```
Skewness after normalization:
YEAR                    0.000000
MONTH                  -0.000473
DAY_OF_WEEK            -0.000456
DEP_TIME                0.017437
DEP_DELAY              -0.346855
TAXI_OUT               -0.019163
WHEELS_OFF             -0.020288
WHEELS_ON              -0.351076
TAXI_IN                 0.003746
ARR_TIME               -0.385604
ARR_DELAY               0.045347
CANCELLED               6.748319
DIVERTED               16.516710
ACTUAL_ELAPSED_TIME     1.420727
AIR_TIME                1.450716
DISTANCE               -0.011056
CARRIER_DELAY           0.088386
WEATHER_DELAY           2.543276
NAS_DELAY               0.044775
SECURITY_DELAY         12.371745
LATE_AIRCRAFT_DELAY     0.001248
```



_After taking care of the data, let's start the [explorative analysis.](predictions.md)_

