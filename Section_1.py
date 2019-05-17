#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from math import radians, pi

# loading data
df_raw = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv',parse_dates=[['DATE', 'TIME']])
df_raw.sort_values(by='DATE_TIME', ascending=False)
# consider data before Dec 31, 2018 
df = df_raw[df_raw.DATE_TIME < pd.datetime(2019,1,1,0,0)]

# Q1
print('Q: What is the total number of persons injured in the dataset (up to December 31, 2018?)')
print('A: ', df['NUMBER OF PERSONS INJURED'].sum())

# Q2
df2016 = df[(df.DATE_TIME < pd.datetime(2017,1,1)) & (df.DATE_TIME >= pd.datetime(2016,1,1))]
df_boro = df2016[df2016['BOROUGH'].notnull()]
print('\nQ: What proportion of all collisions in 2016 occured in Brooklyn? Only consider entries with a non-null value for BOROUGH.')
print('A: ', df_boro[df_boro['BOROUGH'] == 'BROOKLYN'].shape[0] / df_boro.shape[0])

# Q3

total_coll_2016 = df2016.shape[0]
cyc_coll_2016 = df2016[(df2016['NUMBER OF CYCLIST KILLED'] != 0) | (df2016['NUMBER OF CYCLIST INJURED'] != 0)].shape[0]
print('\nQ: What proportion of collisions in 2016 resulted in injury or death of a cyclist?')
print('A: ', cyc_coll_2016 / total_coll_2016)


# Q4


df2017 = df[(df.DATE_TIME < pd.datetime(2018,1,1)) & (df.DATE_TIME >= pd.datetime(2017,1,1))]
df_alco = df2017[df2017.values == 'Alcohol Involvement']
alco_boro = df_alco.groupby(by='BOROUGH').count()
alco_boro['POPULATION'] = pd.Series(dict({'BRONX':1471160, 'BROOKLYN':2648771, 'MANHATTAN':1664727, 'QUEENS':2358582, 'STATEN ISLAND':479458}))
alco_boro['RATE'] = alco_boro['DATE_TIME'] / alco_boro['POPULATION']
print('\nQ: For each borough, compute the number of accidents per capita involving alcohol in 2017. Report the highest rate among the 5 boroughs. Use populations as given by https://en.wikipedia.org/wiki/Demographics_of_New_York_City.')
print('A: ', alco_boro['RATE'].sort_values(ascending=False)[0])


# Q5

zip_veh = df2016.groupby(by='ZIP CODE').count().iloc[:,-5:]
zip_veh['TOTAL VEH'] = zip_veh.sum(axis=1)
print('\nQ: Obtain the number of vehicles involved in each collision in 2016. Group the collisions by zip code and compute the sum of all vehicles involved in collisions in each zip code, then report the maximum of these values.')
print('A: ', zip_veh['TOTAL VEH'].sort_values(ascending=False).iloc[0])


# Q6

df['YEAR'] = df['DATE_TIME'].dt.year
coll_year = df.groupby(by='YEAR').count()
coll_year.drop(2012, inplace=True)
x = np.array(coll_year['DATE_TIME'].index).reshape(-1,1)
y = coll_year['DATE_TIME'].values
lr = LinearRegression()
lr.fit(x, y)
print('\nQ: Obtain the number of vehicles involved in each collision in 2016. Group the collisions by zip code and compute the sum of all vehicles involved in collisions in each zip code, then report the maximum of these values.')
print('A: Slope of Linear Regression: ', lr.coef_)
#print('Variance score: {}'.format(lr.score(x, y)))

plt.scatter(x, y, color = 'red')
plt.plot(x, lr.predict(x), color = 'blue');
plt.title('Total Number of Collision vs Year');
plt.xlabel('Year');
plt.ylabel('Total Number of Collision');
plt.show()


# Q7

df2017['MONTH'] = df2017['DATE_TIME'].dt.month
coll_month = df2017.groupby(by='MONTH').count()
coll_month['MULTI COLLISION RATE'] = coll_month['VEHICLE TYPE CODE 3'] / coll_month['DATE_TIME']
print('\nQ: Compute the rate of multi car collisions for each month of 2017.Calculate the chi-square test statistic for testing whether a collision is more likely to involve 3 or more cars in January than in May.')
print('A: ',coll_month.loc[:,['MULTI COLLISION RATE']])
df2017.drop(columns=['MONTH'], inplace=True)

# Chi-Square Test
contingency_table = coll_month.loc[[1,5],['DATE_TIME', 'VEHICLE TYPE CODE 3']]
contingency_table['MULTI COLLISION'] = contingency_table['VEHICLE TYPE CODE 3']
contingency_table['NON MULTI COLLISION'] = contingency_table['DATE_TIME'] - contingency_table['MULTI COLLISION']
contingency_table.drop(columns=['DATE_TIME', 'VEHICLE TYPE CODE 3'], inplace=True)
contingency_table

stat, p, dof, expected = chi2_contingency(contingency_table)
#print('dof=%d' % dof)
#print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('\nH0: The rate of multi-car(> 3) collision and the month (Jan or May) is independent.')
print('H1: The rate of multi-car(> 3) collision and the month (Jan or May) is dependent.')
print('probability=%.6f, critical=%.6f, stat=%.6f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('A collision is more likely to involve 3 or more cars in January than in May. (reject H0)')
else:
    print('A collision is NOT more likely to involve 3 or more cars in January than in May. (fail to reject H0)')


# Q8

# drop rows with nulls in LATITUDE and LONGITUDE
valid_location = df2017[df2017['LOCATION'].notnull()]
#valid_location.info()
#valid_location.describe()
#valid_location.boxplot(column=['LATITUDE', 'LONGITUDE'])

# drop rows with outliers in LATITUDE and LONGITUDE (outlier defiend as being out of 10 std range)
valid_location = valid_location[np.abs(valid_location['LATITUDE'] - valid_location['LATITUDE'].mean()) <= (10 * valid_location['LATITUDE'].std())]
valid_location = valid_location[np.abs(valid_location['LONGITUDE'] - valid_location['LONGITUDE'].mean()) <= (10 * valid_location['LONGITUDE'].std())]

# calculate standard deviation of latitude and longitude for each zip code
location_std = valid_location.groupby('ZIP CODE')['LATITUDE', 'LONGITUDE'].agg(['std', 'count']).reset_index()
location_std['ZIP CODE'] = location_std['ZIP CODE'].astype(int)

# consider only zipcodes with total collision over 1000
location_std = location_std[location_std['LATITUDE']['count'] >= 1000]

# calculate the area of each zipcode region
earth_radius = 6371 # see wiki: https://en.wikipedia.org/wiki/Earth_radius
location_std['semi-axis a'] = location_std['LATITUDE']['std'].apply(radians)*earth_radius
location_std['semi-axis b'] = location_std['LONGITUDE']['std'].apply(radians)*earth_radius
location_std['AREA'] = pi * location_std['semi-axis a'] * location_std['semi-axis b']

# calculate collision number per square kilometer and print the greatest value.
location_std['collision per sq km'] = location_std['LONGITUDE']['count'] / location_std['AREA']
print('\nQ: The greatest value for collisions per square kilometer?\nA: ',
      location_std.sort_values(by='collision per sq km', ascending=False)['collision per sq km'].iloc[0])

