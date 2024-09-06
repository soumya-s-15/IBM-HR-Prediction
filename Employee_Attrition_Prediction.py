#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\Ganesh prasad sahoo\OneDrive\Desktop\Soumya\Employee_Attrition_Classification_train.csv")
df.head(5)


# In[3]:


df.isna().sum()


# In[4]:


df.notna().sum()


# In[5]:


df.columns


# In[6]:


df.isna().sum().sum()


# In[7]:


df.describe()


# In[8]:


df.shape


# In[9]:


df["Job Role"].unique()


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[47]:


le= LabelEncoder()


# In[48]:


from sklearn.preprocessing import LabelEncoder

# Fit and transform the "Job Role" column
df["Job Role"] = le.fit_transform(df["Job Role"])

# Optionally, print the encoded data
print(df["Job Role"])


# In[36]:


df["Job Role"].unique()


# In[37]:


df["Job Role"].value_counts()


# In[ ]:





# In[38]:


from sklearn.preprocessing import LabelEncoder

# Fit and transform the "Job Role" column
df["Gender"] = le.fit_transform(df["Gender"])


# In[ ]:





# In[39]:


df.columns


# In[40]:


df.drop(columns=[
    "Employee ID", "Overtime", "Distance from Home", "Education Level", "Marital Status", 
    "Number of Dependents", "Company Size", "Company Tenure", "Remote Work", 
    "Leadership Opportunities", "Innovation Opportunities", "Company Reputation", 
    "Employee Recognition"
], inplace=True)


# In[41]:


df


# In[49]:


df["Work-Life Balance"].unique()


# In[50]:


# Fit and transform the "Job Role" column
df["Work-Life Balance"] = le.fit_transform(df["Work-Life Balance"])


# In[51]:


df["Job Satisfaction"]=le.fit_transform(df["Job Satisfaction"])


# In[52]:


df["Performance Rating"]=le.fit_transform(df["Performance Rating"])


# In[53]:


df["Job Level"]=le.fit_transform(df["Job Level"])


# In[54]:


df["Attrition"]=le.fit_transform(df["Attrition"])


# In[55]:


df


# In[56]:


#Scaling the data


# In[57]:


df.columns


# In[58]:


from sklearn.preprocessing import StandardScaler

# Define the numerical columns that need to be scaled
numerical_columns = ['Age', 'Years at Company', 'Monthly Income', 
                     'Work-Life Balance', 'Job Satisfaction', 
                     'Performance Rating', 'Number of Promotions', 'Job Level']

scaler = StandardScaler()

# Fit and transform the numerical columns and assign back to the DataFrame
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

df.head()


# In[ ]:





# In[27]:


sns.pairplot(data=df)


# In[ ]:





# In[59]:


x=df.iloc[:,:-1]
y=df["Attrition"]


# In[60]:


df["Attrition"].unique()


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[63]:


from sklearn.linear_model import LogisticRegression


# In[64]:


lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[66]:


score = lr.score(x_test, y_test) * 100


# In[67]:


score


# In[69]:


#Apply Decision Tree


# In[70]:


from sklearn.tree import DecisionTreeClassifier


# In[74]:


dt=DecisionTreeClassifier( criterion="entropy")
dt.fit(x_train,y_train)


# In[75]:


dt.score(x_train,y_train)*100


# In[76]:


dt.score(x_test,y_test)*100


# In[78]:


for i in range(1,100):
    dt2=DecisionTreeClassifier(max_depth=i)
    dt2.fit(x_train,y_train)
    print(dt2.score(x_train,y_train),dt2.score(x_test,y_test),i)


# In[79]:


from sklearn.neighbors import KNeighborsClassifier


# In[84]:


knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)


# In[85]:


knn.score(x_test,y_test)*100


# In[86]:


for i in range(1,30):
    knn1=KNeighborsClassifier(n_neighbors=i)
    knn1.fit(x_train,y_train)
    print(i,knn1.score(x_train,y_train)*100,knn1.score(x_test,y_test)*100)
    


# In[88]:


#svm


# In[89]:


from sklearn.svm import SVC


# In[94]:


sv=SVC(kernel="rbf")
sv.fit(x_train, y_train)


# In[96]:


sv.score(x_test,y_test)*100


# ### USE ANN

# In[99]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# In[129]:


from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(128, input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))


# In[130]:


model.summary()


# In[131]:


model.compile(loss="binary_crossentropy",optimizer="Adam",metrics=["accuracy"])


# In[132]:


history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)  


# In[115]:


y


# In[ ]:





# In[ ]:




