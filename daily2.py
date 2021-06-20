# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 19:49:44 2021

@author: Mira priyadarshini
"""

import pandas as pd
import numpy as np
df=pd.read_csv('https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/denco.csv')
df
df.shape
df.columns
df.head()
df.dtypes
len(df)
df.describe()
df['region']=df['region'].astype('category')
df.region.value_counts()
df.region.value_counts().plot(kind='bar')
df['custname']=df['custname'].astype('category')
df.custname.value_counts()
df.custname.value_counts().head(5)
df.custname.value_counts().sort_values(ascending=False).tail(5)
#maximum revenue

df.groupby('custname').revenue.sum().sort_values(ascending=False).head(5)
df.groupby('custname')['revenue'].aggregate([np.sum,max,min,'size']).sort_values(by='sum').head(5)
df.groupby('partnum').revenue.sum()
df.groupby('partnum').revenue.sum().sort_values(ascending=False).head(5)

#top profit making items
df.groupby('partnum').margin.sum().sort_values(ascending=False).head(5)
#most sold items
df.groupby('partnum').size().sort_values(ascending=False).head(5)
#region having max revenue

df[['revenue','region']].groupby('region').sum().sort_values(by='revenue',ascending=False)
