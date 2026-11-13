# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 21:19:26 2025

@author: sh222
"""

import pandas as pd
df=pd.read_csv('dfIn.csv')

dfsel=df.loc[df.match==1,:].copy()
dfcomp=dfsel[['chid','tway']]
dfcomp.loc[:,'tway']=(dfcomp['tway'].values-1)**2
dfcomp=pd.merge(dfcomp,df.loc[df.match==0,:],how='inner',on=['chid','tway'])
dfcomp=dfcomp.loc[dfcomp.groupby('chid')['tt'].idxmin()]
dfsel=dfsel[dfsel['chid'].isin(dfcomp['chid'])]

df2=pd.concat([dfsel,dfcomp]).sort_values(['chid', 'tway'])

df2=df2.drop(columns=['alt','realDep','cost','matchiv','matchtt','matchxf','matchprop','compProp','PS','flag'])
df2=df2.rename(columns={'tway': 'alt'})
df2.to_csv('dfIn.csv',index=False)
13786/2
