#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import  seaborn as sns


# In[72]:


ads=pd.read_csv('adsdata.csv')
#ads.head(500)


# In[151]:


#from  pandas.plotting import scatter_matrixe
######### train_test_split  ############
#from sklearn.model_selection import train_test_split
#x=final.t
#y=df_num
#y_train , y_test ,x_train, x_test= train_test_split(y,x,test_size=0.2, random_state=30)
#train_set.shap
#print("x_train _set:", y_train.shape)
#print("x_test_set:",y_test.shape)
#print("y_train _set:", x_train.shape)
#print("y_test_set:",x_test.shape)
########
df=train_set.copy()
df_lable=df["t"].copy()
df= ads.drop(("t"),axis=1)
df_num=df.drop("A", axis=1)
df_num=df_num.drop("B", axis=1)
df_num=df_num.drop("C", axis=1)
df_num=df_num.drop("D", axis=1)
#df_num


# In[164]:


#from  pandas.plotting import scatter_matrixe
######### train_test_split  ############
from sklearn.model_selection import train_test_split
y=final.t
y_train , y_test ,df_num_train, df_num_test= train_test_split(y,df_num,test_size=0.2, random_state=30)
#train_set.shap
print("df_num_train _set:", df_num_train.shape)
print("df_num_test_set:",df_num_test.shape)
print("y_train _set:", y_train.shape)
print("y_test_set:",y_test.shape)
########
df=train_set.copy()
df_lable=df["t"].copy()
df= ads.drop(("t"),axis=1)
df_num=df.drop("A", axis=1)
df_num=df_num.drop("B", axis=1)
df_num=df_num.drop("C", axis=1)
df_num=df_num.drop("D", axis=1)
df_num


# In[32]:


#from sklearn.preprocessing import OneHotEncoder 
#encoder_1hot =OneHotEncoder(sparse=False)
#data_cat_1hot_tmp=encoder_1hot.fit_transform(ads[["t"]])
#data_cat_1hot=pd.DataFrame(data_cat_1hot_tmp)
#data_cat_1hot.columns=encoder_1hot.get_feature_names(['test'])
#data_cat_1hot.head()
##############################final Data##########################
#final=pd.concat([     df_num      ,  data_cat_1hot   ],axis=1)
#final.head(10)                            


# In[58]:


#final.head(10)
#final.describe()
#import  seaborn as sns
#sns.countplot(x='test', data=final ,palette='hls')
#plt.show()


# In[127]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data_adc=ads["t"]
data_adc_encode=encoder.fit_transform(data_adc)
data_adc_encode= pd.DataFrame(data_adc_encode,columns=["t"])
#data_adc_encode.head(500)
##############################final Data##########################
final=pd.concat([     df_num      , data_adc_encode   ],axis=1)
final.head(1000) 
final.info()


# In[67]:


import matplotlib.pyplot as plt
import numpy as np 
#df_num.hist(bins=40, figsize=(40,40))
#plt.show()
import  seaborn as sns
sns.countplot(x='t', data=final ,palette='hls' )
#plt.bar(lable='o')
plt.legend('ad')
plt.show()
count_ads_sub=len(final[final['t']==0])
count_ads=len(final[final['t']==1])
pct_of_ad=count_ads_sub/(count_ads_sub+count_ads)
print("pct_of Data_ad:",pct_of_ad*100)
pct_of_noad=count_ads/(count_ads_sub+count_ads)
print("pct_of Data_nonad:",pct_of_noad*100)
#sns.countplot(x='test_ad.', data=final ,palette='hls')
#plt.show()


# In[70]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),cmap='viridis')


# In[169]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
dtc=DecisionTreeClassifier()
dtc=dtc.fit(df_num_train,y_train)
dt=dtc.predict(df_num_test)
from sklearn.metrics import classification_report
print (classification_report(y_test,dtc.predict(df_num_test)))
print('Accuacy of Desition Tree Classifire on test set:{:.2f}' .format(dtc.score(df_num_test,y_test)))
from sklearn import tree
plt.figure(figsize=(20,20))
temp=tree.plot_tree(dtc.fit(df_num,y),fontsize=5)
plt.show()


# In[192]:


#*************      Naive Bayes clasifier       *******************
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#gnb.fit(df_num_train, y_train())
y_pred =gnb.fit(df_num_train, y_train).predict(df_num_test)
#y_pred = gnb.predict(df_num_test)
print (classification_report(y_test,gnb.predict(df_num_test)))
print (' Naive Bayes test accuracy:  {:.2f} '.format(gnb.score(df_num_test, y_test)))


# In[197]:


#******************          Mlp Classifier       ********************
from sklearn.neural_network import MLPClassifier
mpc=MLPClassifier(hidden_layer_sizes=(5,2), max_iter=1000)
mpc.fit(df_num_train, y_train)
y_predm=mpc.predict(df_num_test)
print ("Mlp Classifier  test accuracy: ", mpc.score(df_num_test, y_test))
print (classification_report(y_test,adsk.predict(df_num_test)))
print("knn(k=5) test accuracy;", adsk.score(df_num_test  , y_test))


# In[167]:


#******************    <    logistic Regreession Classifier   >     ************************
from sklearn.linear_model import LogisticRegression
adsr=LogisticRegression(solver='1bfgs')
adsr.fit(y_train.ravel())
y_predr=adsr.predict(y_test)


# In[252]:


#**********************        knn          ***************************
from sklearn.neighbors import KNeighborsClassifier
k=5
adsk=KNeighborsClassifier(n_neighbors=k)
adsk.fit(df_num_train,y_train)
y_predk=adsk.predict(df_num_test)
print("when k={} neighnors , knn test accuracy:{}".format(k , adsk.score(df_num_test ,y_test)))
print("when k={} neighnors , knn test accuracy:{}".format(k , adsk.score(df_num_train ,y_train)))
print(classification_report(y_test,adsk.predict(df_num_test)))
print("knn(k=5) test accuracy;",adsk.score(df_num_test, y_test))
ran=np.arange(1,30)
train_list=[]
test_list=[]
for i  , each in enumerate(ran):
    adsk=KNeighborsClassifier(n_neighbors=each)
    # adsk.fit(df_num_train,y_train)
    adsk.fit(df_num_train,y_train)
    
    test_list.append(adsk.score(df_num_test, y_test))
    train_list.append(adsk.score(df_num_train , y_train))
print("when k={} neighnors , knn test accuracy:{}".format(np.max(test_list) ,test_list.index (np.max(test_list))+1))
print("when k={} neighnors , knn test accuracy:{}".format(np.max(train_list) ,train_list.index (np.max(train_list))+1))
plt.figure(figsize=[15,15])
plt.plot(ran,test_list,label='test score')
plt.plot(ran,train_list,label='train score')
plt.xlabel('number of neighbers')
plt.ylabel('numer/count')
plt.xticks(ran)
plt.legend()
plt.show()


# In[176]:


#****************              logistic Regreession Classifier              *****************
from sklearn.linear_model import LogisticRegression
#adsr=LogisticRegression(solver='1bfgs')
#adsr.fit(df_num_train,y_train())
#y_predr=adsr.predict(y_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#x_train, x_test, y_train, y_test = train_test_split(df_num, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(df_num_train, y_train)
y_pred = logreg.predict(df_num_test)
print(classification_report(y_test,logreg.predict(df_num_test)))
print('Accuracy of logistic regression classifier on test set:{:.2f}'.format(logreg.score(df_num_test, y_test)))

