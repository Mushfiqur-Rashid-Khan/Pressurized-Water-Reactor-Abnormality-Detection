import pandas as pd
df=pd.read_csv('Data_PWR.csv')

from sklearn.cluster import KMeans

df.drop(['Readings'], axis=1)

df.dropna()

df.fillna(0)

km=KMeans(n_clusters=5)

yp=km.fit_predict(df)

df['clusters']=yp

df1=df[df.clusters==1]
df2=df[df.clusters==2]
df3=df[df.clusters==3]
df4=df[df.clusters==4]
df5=df[df.clusters==5]

from matplotlib import pyplot as plt
plt.scatter(df1,df1, color='blue')
plt.scatter(df2,df2, color='green')
plt.scatter(df3,df3, color='orange')
plt.scatter(df4,df4, color='red')
plt.scatter(df5,df5, color='black')
