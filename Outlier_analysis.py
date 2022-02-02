#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib.pyplot import figure
import datetime
import seaborn as sns
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', None)

get_ipython().system('pip install awswrangler')

import sys
get_ipython().system('{sys.executable} -m pip install PyAthena')

import awswrangler as wr
pd.options.mode.chained_assignment = None  # default='warn'

from pyathena import connect
from pyathena.pandas.cursor import PandasCursor

cursor = connect(s3_staging_dir='s3://remsage/',
            region_name='us-east-1',
                 cursor_class=PandasCursor).cursor()

query = '''SELECT 
event_type, 
report_number, 
date_of_event, 
d."brand_name", 
d."manufacturer_d_name", 
d."device_report_product_code", 
d."openfda"."device_name", 
d."openfda"."device_class" 
FROM "fda-open-database"."event" 
cross join unnest(device) as t(d) WHERE UPPER(d."device_report_product_code") LIKE ('DZE') order by date_of_event desc'''
df_event = cursor.execute(query).as_pandas()

df_event['date_of_event']=pd.to_datetime(df_event['date_of_event'],format='%Y-%m-%d',errors='coerce')

df_event.head(3)

df = df_event[['event_type', 'manufacturer_d_name',"brand_name"]]

temp=pd.DataFrame(df.value_counts()).reset_index()
temp.rename(columns={'event_type': 'event_type', 'manufacturer_d_name': 'manufacturer_d_name',"brand_name":"brand_name",0: "count"}, inplace=True)
temp=temp.sort_values("manufacturer_d_name",ascending=True)

temp['normalized_percent'] = ''
for i in range(len(temp)):
    if temp['event_type'][i] == 'Injury':
        temp['normalized_percent'][i] = (temp['count'][i]/temp[temp.event_type=='Injury']['count'].sum())*100
    elif temp['event_type'][i] == 'Malfunction':
        temp['normalized_percent'][i] = (temp['count'][i]/temp[temp.event_type=='Malfunction']['count'].sum())*100
    elif temp['event_type'][i] == 'Other':
        temp['normalized_percent'][i] = (temp['count'][i]/temp[temp.event_type=='Other']['count'].sum())*100
    elif temp['event_type'][i] == 'No answer provided':
        temp['normalized_percent'][i] = (temp['count'][i]/temp[temp.event_type=='No answer provided']['count'].sum())*100
    else:
        temp['normalized_percent'][i] = (temp['count'][i]/temp[temp.event_type=='Death']['count'].sum())*100


temp = temp.sort_values("normalized_percent",ascending=False)

#Box Plot for Manufacture-Brand Name for normalized event count by event_type
fig = px.box(temp, x="event_type",y="normalized_percent",color="event_type",hover_name = "brand_name", title = "Box Plot of events for Manufacturer-Brand Name",log_y=True)
fig.show()


temp1=temp[['event_type',"normalized_percent"]]

#normalizing by event_type
#calculating lower and upper bounds for Death 
df_quantile=temp1[temp1.event_type=='Death']
q1 = np.quantile(df_quantile['normalized_percent'].values, 0.25)
q3 = np.quantile(df_quantile['normalized_percent'], 0.75)
med = np.median(df_quantile['normalized_percent'])

iqr = q3-q1
upper_bound_d = q3+(1.5*iqr)
lower_bound_d = q1-(1.5*iqr)
if lower_bound_d < 0:
    lower_bound_d = 0

#calculating lower and upper bounds for Malfunction 
df_quantile=temp1[temp1.event_type=='Malfunction']
q1 = np.quantile(df_quantile['normalized_percent'].values, 0.25)
q3 = np.quantile(df_quantile['normalized_percent'], 0.75)
med = np.median(df_quantile['normalized_percent'])

iqr = q3-q1
upper_bound_m = q3+(1.5*iqr)
lower_bound_m = q1-(1.5*iqr)
if lower_bound_m < 0:
    lower_bound_m = 0

#calculating lower and upper bounds for Injury
df_quantile=temp1[temp1.event_type=='Injury']
q1 = np.quantile(df_quantile['normalized_percent'].values, 0.25)
q3 = np.quantile(df_quantile['normalized_percent'], 0.75)
med = np.median(df_quantile['normalized_percent'])

iqr = q3-q1
upper_bound_i = q3+(1.5*iqr)
lower_bound_i = q1-(1.5*iqr)
if lower_bound_i < 0:
    lower_bound_i = 0

#calculating lower and upper bounds for Other 
df_quantile=temp1[temp1.event_type=='Other']
q1 = np.quantile(df_quantile['normalized_percent'].values, 0.25)
q3 = np.quantile(df_quantile['normalized_percent'], 0.75)
med = np.median(df_quantile['normalized_percent'])

iqr = q3-q1
upper_bound_o = q3+(1.5*iqr)
lower_bound_o = q1-(1.5*iqr)
if lower_bound_o < 0:
    lower_bound_o = 0

#calculating lower and upper bounds for No answer provided 
df_quantile=temp1[temp1.event_type=='No answer provided']
q1 = np.quantile(df_quantile['normalized_percent'].values, 0.25)
q3 = np.quantile(df_quantile['normalized_percent'], 0.75)
med = np.median(df_quantile['normalized_percent'])

iqr = q3-q1
upper_bound_n = q3+(1.5*iqr)
lower_bound_n = q1-(1.5*iqr)
if lower_bound_n < 0:
    lower_bound_n = 0
#print(iqr, upper_bound, lower_bound)


outliers = pd.DataFrame()
d = temp[temp.event_type == 'Death']
d = d[(d.normalized_percent <lower_bound_d) | (d.normalized_percent > upper_bound_d)]
outliers = outliers.append(d)
d = temp[temp.event_type == 'Injury']
d = d[(d.normalized_percent <lower_bound_i) | (d.normalized_percent > upper_bound_i)]
outliers = outliers.append(d)
d = temp[temp.event_type == 'Malfunction']
d = d[(d.normalized_percent <lower_bound_m) | (d.normalized_percent > upper_bound_m)]
outliers = outliers.append(d)
d = temp[temp.event_type == 'Other']
d = d[(d.normalized_percent <lower_bound_o) | (d.normalized_percent > upper_bound_o)]
outliers = outliers.append(d)
d = temp[temp.event_type == 'No answer provided']
d = d[(d.normalized_percent <lower_bound_n) | (d.normalized_percent > upper_bound_n)]
outliers = outliers.append(d)
outliers['outliers'] = True
outliers.sort_values("normalized_percent", ascending = False)


d1 = pd.DataFrame()
d = temp[temp.event_type == 'Death']
d = d[(d.normalized_percent >=lower_bound_d) & (d.normalized_percent <= upper_bound_d)]
d1 = d1.append(d)
d = temp[temp.event_type == 'Injury']
d = d[(d.normalized_percent >=lower_bound_i) & (d.normalized_percent <= upper_bound_i)]
d1 = d1.append(d)
d = temp[temp.event_type == 'Malfunction']
d = d[(d.normalized_percent >=lower_bound_m) & (d.normalized_percent <= upper_bound_m)]
d1 = d1.append(d)
d = temp[temp.event_type == 'Other']
d = d[(d.normalized_percent >= lower_bound_o) & (d.normalized_percent <= upper_bound_o)]
d1 = d1.append(d)
d = temp[temp.event_type == 'No answer provided']
d = d[(d.normalized_percent >= lower_bound_n) & (d.normalized_percent <= upper_bound_n)]
d1 = d1.append(d)
d1['outliers'] = False
d1.sort_values("normalized_percent",ascending=False)

event_norm = event_norm.append(outliers)
event_norm = event_norm.append(d1)
event_norm = event_norm.sort_values("normalized_percent", ascending = False)

#normalizing with respect to total adverse event counts for a product code
count_total = temp['count'].sum()
count_total

temp['normalized_percent'] = (temp['count']/count_total)*100

#calculating lower and upper bounds for the box plot 
temp1=temp[['event_type',"normalized_percent"]]
q1 = np.quantile(temp1['normalized_percent'].values, 0.25)
q3 = np.quantile(temp1['normalized_percent'], 0.75)
med = np.median(temp1['normalized_percent'])

iqr = q3-q1
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)
if lower_bound < 0:
    lower_bound = 0
print(lower_bound, upper_bound)

arr2 = temp[(temp[['normalized_percent']].values >= lower_bound) & (temp[['normalized_percent']].values <= upper_bound)]

outlier_df = temp[(temp[['normalized_percent']].values < lower_bound) | (temp[['normalized_percent']].values > upper_bound)]
outlier_df.sort_values("normalized_percent",ascending=False)
    
event_total = event_total.append(outlier_df)
event_total = event_total.append(arr2)
event_total = event_total.sort_values("normalized_percent", ascending = False)

