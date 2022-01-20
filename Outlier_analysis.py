#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib.pyplot import figure
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
pd.set_option('display.max_rows', None)


# In[2]:


get_ipython().system('pip install awswrangler')


# In[3]:


import sys
get_ipython().system('{sys.executable} -m pip install PyAthena')


# In[4]:


import awswrangler as wr
pd.options.mode.chained_assignment = None  # default='warn'


# In[5]:


from pyathena import connect
from pyathena.pandas.cursor import PandasCursor

cursor = connect(s3_staging_dir='s3://remsage/',
            region_name='us-east-1',
                 cursor_class=PandasCursor).cursor()


# In[6]:


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


# In[7]:


df_event = cursor.execute(query.format()).as_pandas()


# In[8]:


df_event['date_of_event']=pd.to_datetime(df_event['date_of_event'],format='%Y-%m-%d',errors='coerce')


# In[9]:


df_event.head(3)


# In[10]:


df = df_event[['event_type', 'manufacturer_d_name',"brand_name"]]


# In[35]:


temp=pd.DataFrame(df.value_counts()).reset_index()
temp.rename(columns={'event_type': 'event_type', 'manufacturer_d_name': 'manufacturer_d_name',"brand_name":"brand_name",0: "count"}, inplace=True)
temp=temp.sort_values("manufacturer_d_name",ascending=True)


# In[14]:


temp['normalized_percent'] = ''
for i in range(len(temp)):
    if temp['event_type'][i] == 'Injury':
        temp['normalized_percent'][i] = (temp['count'][i]/temp['event_type'].value_counts()[0])*100
    elif temp['event_type'][i] == 'Malfunction':
        temp['normalized_percent'][i] = (temp['count'][i]/temp['event_type'].value_counts()[1])*100
    elif temp['event_type'][i] == 'Other':
        temp['normalized_percent'][i] = (temp['count'][i]/temp['event_type'].value_counts()[2])*100
    elif temp['event_type'][i] == 'No answer provided':
        temp['normalized_percent'][i] = (temp['count'][i]/temp['event_type'].value_counts()[3])*100
    else:
        temp['normalized_percent'][i] = (temp['count'][i]/temp['event_type'].value_counts()[4])*100


# In[ ]:


temp = temp.sort_values("normalized_percent",ascending=False)
temp


# In[16]:


import seaborn as sns
import matplotlib as plt


# In[ ]:


import plotly.express as px
fig = px.box(temp, x="event_type",y="normalized_percent",color="event_type",hover_name = "brand_name", title = "Box Plot of events for Manufacturer-Brand Name",log_y=True)
fig.show()


# In[18]:


temp1=temp[['event_type',"normalized_percent"]]
import numpy as np


#calculating lower and upper bounds for Death 
df_quantile=temp1[temp.event_type=='Death']
q1 = np.quantile(df_quantile['normalized_percent'].values, 0.25)
q3 = np.quantile(df_quantile['normalized_percent'], 0.75)
med = np.median(df_quantile['normalized_percent'])

iqr = q3-q1
upper_bound_d = q3+(1.5*iqr)
lower_bound_d = q1-(1.5*iqr)
if lower_bound_d < 0:
    lower_bound_d = 0

#calculating lower and upper bounds for Malfunction 
df_quantile=temp1[temp.event_type=='Malfunction']
q1 = np.quantile(df_quantile['normalized_percent'].values, 0.25)
q3 = np.quantile(df_quantile['normalized_percent'], 0.75)
med = np.median(df_quantile['normalized_percent'])

iqr = q3-q1
upper_bound_m = q3+(1.5*iqr)
lower_bound_m = q1-(1.5*iqr)
if lower_bound_m < 0:
    lower_bound_m = 0

#calculating lower and upper bounds for Injury
df_quantile=temp1[temp.event_type=='Injury']
q1 = np.quantile(df_quantile['normalized_percent'].values, 0.25)
q3 = np.quantile(df_quantile['normalized_percent'], 0.75)
med = np.median(df_quantile['normalized_percent'])

iqr = q3-q1
upper_bound_i = q3+(1.5*iqr)
lower_bound_i = q1-(1.5*iqr)
if lower_bound_i < 0:
    lower_bound_i = 0

#calculating lower and upper bounds for Other 
df_quantile=temp1[temp.event_type=='Other']
q1 = np.quantile(df_quantile['normalized_percent'].values, 0.25)
q3 = np.quantile(df_quantile['normalized_percent'], 0.75)
med = np.median(df_quantile['normalized_percent'])

iqr = q3-q1
upper_bound_o = q3+(1.5*iqr)
lower_bound_o = q1-(1.5*iqr)
if lower_bound_o < 0:
    lower_bound_o = 0

#calculating lower and upper bounds for No answer provided 
df_quantile=temp1[temp.event_type=='No answer provided']
q1 = np.quantile(df_quantile['normalized_percent'].values, 0.25)
q3 = np.quantile(df_quantile['normalized_percent'], 0.75)
med = np.median(df_quantile['normalized_percent'])

iqr = q3-q1
upper_bound_n = q3+(1.5*iqr)
lower_bound_n = q1-(1.5*iqr)
if lower_bound_n < 0:
    lower_bound_n = 0
#print(iqr, upper_bound, lower_bound)


# In[19]:


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
outliers.sort_values("normalized_percent", ascending = False)


# In[ ]:


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
d1.sort_values("normalized_percent",ascending=False)


# In[33]:


# boxplot of data within the whisker
fig = px.box(d1, x="event_type",y="normalized_percent",color="event_type",hover_name = "brand_name", title = "Box Plot of normalized for each event type for Manufacturer-Brand Name",log_y=True)
fig.show()


# In[22]:


count_total = temp['count'].sum()
count_total


# In[24]:


temp1 = temp


# In[36]:


temp1['normalized_percent'] = (temp['count']/count_total)*100


# In[28]:


#calculating lower and upper bounds for the box plot 
temp2=temp1[['event_type',"normalized_percent"]]
q1 = np.quantile(temp2['normalized_percent'].values, 0.25)
q3 = np.quantile(temp2['normalized_percent'], 0.75)
med = np.median(temp2['normalized_percent'])

iqr = q3-q1
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)
if lower_bound < 0:
    lower_bound = 0
print(lower_bound, upper_bound)


# In[31]:


# boxplot of data within the whisker
arr2 = temp1[(temp1[['normalized_percent']].values >= lower_bound) & (temp1[['normalized_percent']].values <= upper_bound)]
fig = px.box(arr2, x="event_type",y="normalized_percent",color="event_type",hover_name = "brand_name", title = "Box Plot of events normalized for Manufacturer-Brand Name", log_y=True)
fig.show()


# In[ ]:


outlier_df = temp1[(temp1[['normalized_percent']].values < lower_bound) | (temp1[['normalized_percent']].values > upper_bound)]
outlier_df.sort_values("normalized_percent",ascending=False)
    

