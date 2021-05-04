# Understanding & Processing The Data
Since creating the previous part about delays I've learned about few things that might help me with past issues. In this document I will try to understand the data that I'm working with without a very in-depth analysis. I'm going to focus more on feature engineering and processing the data to fit my needs accordingly to the data set statistics. 



## Exploring the information

In this section we're going to explore the data we're working with and show some plots on distribution as well as the relationship between features and **our target - the price.** We will start by displaying some general information. As an introduction I will say that we're dealing with New Zealand's data from the year 2019. 

```
RangeIndex: 162833 entries, 0 to 162832
Data columns (total 11 columns):
 #   Column        Non-Null Count   Dtype 
---  ------        --------------   ----- 
 0   Travel Date   162833 non-null  object
 1   Dep. airport  162809 non-null  object
 2   Dep. time     162833 non-null  object
 3   Arr. airport  162809 non-null  object
 4   Arr. time     162828 non-null  object
 5   Duration      162833 non-null  object
 6   Direct        162833 non-null  object
 7   Transit       123077 non-null  object
 8   Baggage       2311 non-null    object
 9   Airline       162828 non-null  object
 10  Airfare(NZ$)  162833 non-null  int64 

```



### The schema

| Travel Date | Dep. airport | Dep. time | Arr. airport | Arr. time | Duration | Direct   | Transit       | Baggage                  | Airline         | Airfare(NZ$) |
| ----------- | ------------ | --------- | ------------ | --------- | -------- | -------- | ------------- | ------------------------ | --------------- | ------------ |
| 19/09/2019  | AKL          | 1:35 PM   | CHC          | 3:00 PM   | 1h 25m   | (Direct) | 4h 35m in NSN | Checked bag NOT included | Air New Zealand | 442          |

As we can see in the table I created above, there's some data that needs to be transformed to be useful and some columns to be cleaned. Before anything I will also see how many null values we have in case that we don't have to work on a column because it doesn't consist enough important information. For now the plan is to:

- Transform the duration column to: 1h 20min --> 80min
- Change the values in direct to following:
  - Direct --> 0
  - 1 Stop --> 1
  - 2 Stops --> 2
  - etc.

- Divide the transit into the transit time and type columns

  - 9h in NSN --> 540min(Transit Time) & NSN (Transit Type)

    

### Missing columns & sanity of the information

It looks like the baggage column won't be too useful so I think it's fair to just drop it. In the case of null values below 0.05% I'm going to do the same. The only column I will take care of is the transit. When it comes to sanity, it seems like all the features are alright in that department.

```
Travel Date      0.00
Dep. airport     0.01
Dep. time        0.00
Arr. airport     0.01
Arr. time        0.00
Duration         0.00
Direct           0.00
Transit         24.42
Baggage         98.58
Airline          0.00
Airfare(NZ$)     0.00
```

```
Columns are sane: True
```



### Skewness

One of the things that we should definitely check is skewness, we can then verify which is considered normal for this dataset and use transformation on columns that don't adhere to that idea. I noticed that most of the columns have 1.5 - 3, but there were a few with a big difference.

```
Skewness:
ItinID              -2.045501
MktID               -2.045501
MktCoupons           7.251936
Quarter             -0.045702
OriginWac           -0.017259
DestWac             -0.018266
Miles                0.942547
ContiguousUSA       -3.898311
NumTicketsOrdered    0.646620
PricePerTicket       0.001518
```



### Correlation between the features

We have a compact amount of solid features this time. As we can see, two of them are so correlated it doesn't even make sense to keep them around so I'm going to drop on of them. Miles and price per ticket have an obvious correlation and we can also notice some minor correlations between the features.

![Correlation heatmap](../../plots/prices/corr-heatmap.png)



## Processing the features

As I said before, in this document we won't too far into data exploration so I'm simply going to fix some common issues that the dataset could have and try to get some information. Our end goal there is to create an efficient model that can tackle a lot of categorical features and minimize their amount.



### Taking care of missing values

Before I try to encode anything, I'm going to take care of missing values so they don't get in our way while pre-processing the data. I'm going to process with the plan I explained before, in this section I'm only going to show the end result of what I did for the context purposes.

```python
data = data.drop(['Baggage'], axis=1)
data = data.dropna(subset=['Dep. airport', 'Arr. airport', 'Airline'])
```

First I dropped everything we didn't need, the whole Baggage column that had mostly missing values as well as a small percent of the other columns. The only column that had a lot of them was Transit but I will take care of them in the next section because these NaN's are actually useful for us.

_Note: While doing binary encoding later and changing the names of the columns so they don't repeat, I found out that there were some missing values left, but since a column got created for them in every category, I simply dropped them. I didn't look into what could cause it but it could have been a different encoding of a non defined value that dropna method didn't detect. It did not cause trouble so I won't go into it._



### Transforming dates and time

When it comes to processing time, we need to consider two things:

- The way in which the time is encoded
- If the time has any meaning useful for us

In this scenario we're dealing with a few columns: Transit, Travel Date, Duration, Arr. time and Dep. time. These columns all together contain three different types of time descriptions: AM/PM, our normal date type (DD/MM/YYYY) and hours/minutes. Let's start from encoding the Transit, and first, before doing anything, I want to look into the way the values in this column look. We deal with two different information, the time that was needed for the transit but also the duration of it. I decided to split it to two different categories:

```python
# Dividing transit into two columns
data[['TransitTime', 'TransitPlace']] = data['Transit'].str.split(' in ', expand=True)
print(data.tail())

# Transform to transit time to minutes
data['TransitTime'] = data['TransitTime'].fillna(0)
data['TransitPlace'] = data['TransitPlace'].fillna('None')
```



Next I took care of the values in the TransitTime column. I changed it so everything is turned to minutes and removed the h/m letters so we are only left with numerical data. Since the AM/PM time can be shortened to just the values since they increment till the day is over, I simply removed the ":" in-between hours and minutes. The last thing I did was splitting the date to month and day, at the same time removing the year since it's the same for every row in the data set.

```
# Transform the date column to two columns
data[['Day', 'Month', 'Year']] = data['Travel Date'].str.split('/', expand=True)
```



### Feature encoding

First of all, I want to encode out categorical data and for this purpose I'm going to use **Binary Encoding** that helps us a lot with the amount of columns we need to create. When it comes to situations like that, we want to keep columns like that but polluting our dataset by incorporating One Hot Encoding is simply not efficient, sometimes you might even end up with 400+ features. The way Binary Encoding is following: it converts multi-class labels into binary labels and creates a column for each binary digit. 

**Remember:** _If there are **n** unique categories, then binary encoding results in log(base 2)ⁿ features._

Let's imagine having 100 features. When using One Hot Encoding to distinguish between them in a numerical form we would need to create 100 new features. If we switch to binary we get only 7 features. For this purpose I'm going to use Scikit's build-in method called LabelBinarizer. Under I will present how the process will work.

![Binary Encoding](https://i.imgur.com/2E1OqMu.png)

After that, the only thing I need to encode to create a fully numerical data set is the Direct column. I used LabelEncoding for it because one category is more than another, it makes sense to use it here.

```python
le = LabelEncoder()
data['Direct'] = le.fit_transform(data['Direct'])
```



### Normalizing & scaling the data

The last thing I wanted to take care of is normalizing and scaling the data, in our case these steps are very short. When it comes to normalisation I tried both methods, yeo-yeo-johnson and box-cox, the latter turned out to provide a more satisfying result. I decided to scale the data since some models prefer it and also the RobustScaler is especially good with data that contains outliers. It would be really hard to visualise them with such a huge data set and clean by hand or at least the increase in accuracy in comparison to time needed to do it is not satisfying to me.



_After taking care of the data, let's head to the next section where I'm [building the prediction model.](predictions.md)_

