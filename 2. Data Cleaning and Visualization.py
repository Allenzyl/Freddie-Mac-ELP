"""
@author: yilun
The cleaning process is based mostly on the loan data user guide. 
"""

import pandas as pd
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


Insample = pd.read_csv('InsampleFormatedData.txt',sep='|',header=None)
Outsample = pd.read_csv('OutsampleFormatedData.txt',sep='|',header=None)
data = Insample.append(Outsample)
del(Insample)
del(Outsample)

data.columns=['FICO','First Payment Date','First Time Homebuyer Flag','MSA','Units','Occupancy','CLTV','DTI',
                'OUPB','OLTV','OInterest rate','Channel','Property State','Property Type','Loanid','Purpose',
                'Number of borrowers','First Loan Age','Default Status']


''' Data Cleaning Based on User Guide'''
sample = data.head(100)
data.info()
describe = data.describe()
data.columns

# Remove Nan values (any nan in a row will be deleted)
for i in data.columns:
    print(len(data[data[i].isnull()]))

data['MSA'].fillna(0, inplace=True)
data = data.dropna(axis=0,how='any')

# Remove data with loan age lager than 12
Count = collections.Counter(data['First Loan Age'])
data = data[data['First Loan Age']<=12]

# Remove default status is 2
Count = collections.Counter(data['Default Status'])
data['Default Status'].unique()
data = data[data['Default Status']<2]

# FICO score
data = data[(data['FICO']>=600) & (data['FICO']<=850)]
data['FICO'].describe()

# First Payment Date (Do nothing)
Count = collections.Counter(data['First Payment Date'])

# First time Homebuye
Count = collections.Counter(data['First Time Homebuyer Flag'])
data = data[data['First Time Homebuyer Flag']!='9']

# Units
Count = collections.Counter(data['Units'])
data = data[data['Units']!=99]

# Occupency
Count = collections.Counter(data['Occupancy'])

# CLTV
sns.distplot(data['CLTV'])
data = data[(data['CLTV'] >= 0) & (data['CLTV'] <= 200)]

# DTI
sns.distplot(data['DTI'])
data = data[(data['DTI'] >= 0) & (data['DTI'] <= 65)]

# OUPB
sns.distplot(data['OUPB'])

# OLTV
sns.distplot(data['OLTV'])
data = data[(data['OLTV'] >= 6) & (data['OLTV'] <= 105)]

# OInterest rate
sns.distplot(data['OInterest rate'])

# Channel
Count = collections.Counter(data['Channel'])

# Property State: VI for Virgin Island, PR for Puerto Rico, GU for Guam
data = data[(data['Property State']!='VI') & (data['Property State']!='PR') & (data['Property State']!='GU')]
Count = collections.Counter(data['Property State'])

# Property Type
Count = collections.Counter(data['Property Type'])
data = data[data['Property Type'] != '99']

# Purpose
Count = collections.Counter(data['Purpose'])

# Number of borrowers
Count = collections.Counter(data['Number of borrowers'])
data = data[data['Number of borrowers'] != 99]

# MSA and ROUPB
data['ROUPB_mean'] = data.groupby(['Property State','MSA'])['OUPB'].transform(lambda x: x/x.mean())
data['ROUPB_median'] = data.groupby(['Property State','MSA'])['OUPB'].transform(lambda x: x/x.median())
data['ROUPB_standardized'] = data.groupby(['Property State','MSA'])['OUPB'].transform(lambda x: (x-x.mean()) / x.std())

# Add a column to notate origination year
data['OYear'] = data['Loanid'].apply(lambda x: x[2:4])
data['OYear'].unique()


# data.to_csv('cleaneddata.csv',index=False)

for i in ['First Time Homebuyer Flag','Units', 'Occupancy','Channel', 'Property Type', 'Purpose','Number of borrowers']:
    fig=plt.figure(figsize=(15,5))
    sns.countplot(x='OYear',hue=i,data=data,order=['99', '00', '01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12', '13', '14', '15', '16', '17'])
    fig.savefig('./Plot/%s.png' % i)

# Spliting data and taylor the data based on 17 loans
# data = pd.read_csv('cleaneddata.csv')
insample = data[(data['OYear'] %2 !=0 ) & (data['OYear'] != 17)]
outsample = data[data['OYear'] %2 ==0]
predict = data[data['OYear'] == 17]
del(data)

# define  names
dfname=[insample,outsample,predict]
dfyear=['odd years','even years','2017']
varname=list(insample.columns)

'''Categorical variables'''

# First time homebuyer
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.countplot(dfname[i][varname[2]],ax=axs[i],order=['N','Y']).set(title='%s distplot for %s loan' % (varname[2],dfyear[i]))
fig.savefig('.\Plot\%s distplot.png' % varname[2])

# Occupancy
dfname=[insample,outsample,predict]
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.countplot(dfname[i]['Occupancy'],ax=axs[i]).set(title='Occupancy barplot for %s loan' % (dfyear[i]))
fig.savefig('.\Plot\Occupancy barplot.png')
#insample = insample[insample['Occupancy']=='P']
#outsample = outsample[outsample['Occupancy']=='P']

# Channel
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.countplot(dfname[i]['Channel'],ax=axs[i],order=['R','B','C']).set(title='Channel barplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/Channel barplot.png')
insample = insample[insample['Channel']!='T']
outsample = outsample[outsample['Channel']!='T']

# Units
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.countplot(dfname[i]['Units_1'],ax=axs[i]).set(title='Units barplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/Units barplot.png')

# Property type
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.countplot(dfname[i]['Property Type'],ax=axs[i],order=['CO','PU','MH','SF','CP']).set(title='Property type barplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/Property Type barplot.png')

# Purpose
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.countplot(dfname[i]['Purpose'],ax=axs[i],order=['P','N','C']).set(title='Purpose barplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/Purpose barplot1.png')
insample = insample[insample['Purpose']=='P']

# Number of borrowers
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.countplot(dfname[i]['Number of borrowers'],ax=axs[i]).set(title='Number of borrowers barplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/Number of borrowers barplot.png')

'''Numerical variables'''
# FICO
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.distplot(dfname[i][varname[0]],bins = 50,ax=axs[i]).set(title='%s distplot for %s loan' % (varname[0],dfyear[i]))
fig.savefig('./Plot/FICO distplot1.png')
insample = insample[insample['FICO']>=600]
outsample = outsample[outsample['FICO']>=600]
dfname=[insample,outsample,predict]

# DTI
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.distplot(dfname[i]['DTI'],bins = 50,ax=axs[i]).set(title='DTI distplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/DTI distplot1.png')
insample = insample[insample['DTI']<=predict['DTI'].max()]
outsample = outsample[outsample['DTI']<=predict['DTI'].max()]

# OLTV Clean???
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.distplot(dfname[i]['OLTV'],bins = 50,ax=axs[i]).set(title='OLTV distplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/OLTV distplot1.png')

drop_indices = np.random.choice(insample[(insample['OLTV']==80) & (insample['OLTV']>=50)].index, 1000000, replace=False)
insample = insample.drop(drop_indices)

# OInterest rate
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.distplot(dfname[i]['OInterest rate'],bins = 50,ax=axs[i]).set(title='OInterest rate distplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/OInterest rate distplot.png')

# RUPB
fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.distplot(dfname[i]['ROUPB_standardized'],bins = 50,ax=axs[i]).set(title='ROUPB_standardized distplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/ROUPB_standardized distplot.png')

fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.distplot(dfname[i]['ROUPB_mean'],bins = 50,ax=axs[i]).set(title='ROUPB_mean distplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/ROUPB_mean distplot.png')

fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
for i in range(len(dfname)):
    sns.distplot(dfname[i]['ROUPB_median'],bins = 50,ax=axs[i]).set(title='ROUPB_median distplot for %s loan' % (dfyear[i]))
fig.savefig('./Plot/ROUPB_median distplot.png')

data = insample.append(outsample).append(predict)
data.to_csv('cleaneddata.csv',index=False)


'''Spline'''
FICOfreq=(insample[insample['Default Status']==1].groupby(['FICO'])['Default Status'].count()) / (insample.groupby(['FICO'])['Default Status'].count())
DTIfreq=(insample[insample['Default Status']==1].groupby(['DTI'])['Default Status'].count()) / (insample.groupby(['DTI'])['Default Status'].count())
OLTVfreq=(insample[insample['Default Status']==1].groupby(['OLTV'])['Default Status'].count()) / (insample.groupby(['OLTV'])['Default Status'].count())
#ROUPBfreq=(insample[insample['Default Status']==1].groupby(['ROUPB_standardized'])['Default Status'].count()) / (insample.groupby(['ROUPB_standardized'])['Default Status'].count())

fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
sns.scatterplot(x=FICOfreq.index, y=FICOfreq,ax=axs[0]).set(title='Default frequency by FICO for 1999-2013 loan')
sns.scatterplot(x=DTIfreq.index, y=DTIfreq,ax=axs[1]).set(title='Default frequency by DTI for 1999-2013 loan')
sns.scatterplot(x=OLTVfreq.index, y=OLTVfreq,ax=axs[2]).set(title='Default frequency by OLTV for 1999-2013 loan')
#sns.scatterplot(x=ROUPBfreq.index, y=ROUPBfreq,ax=axs[3]).set(title='Default frequency by ROUPB for 1999-2013 loan')
fig.savefig('./Plot/Default frequency for 1999-2013 loan.png')