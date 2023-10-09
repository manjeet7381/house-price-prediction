#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("kc_house_data.csv")


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine


# In[6]:


plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
plt1 = plt()
sns.despine


# In[7]:


plt.scatter(data.price,data.sqft_living)
plt.title("Price vs Square Feet")


# In[8]:


plt.scatter(data.price,data.long)
plt.title("Price vs Location of the area")


# In[9]:


plt.scatter(data.price,data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")


# In[10]:


plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine


# In[11]:


plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])


# In[12]:


plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price ( 0= no waterfront)")


# In[13]:


train1 = data.drop(['id', 'price'],axis=1)


# In[14]:


train1.head()


# In[15]:


data.floors.value_counts().plot(kind='bar')


# In[16]:


plt.scatter(data.floors,data.price)


# In[17]:


plt.scatter(data.condition,data.price)


# In[18]:


plt.scatter(data.zipcode,data.price)
plt.title("Which is the pricey location by zipcode?")


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


reg = LinearRegression()


# In[21]:


labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)


# In[24]:


reg.fit(x_train,y_train)


# In[132]:


reg.score(x_test,y_test)


# In[98]:


from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'squared_error')


# In[67]:


clf.fit(x_train,y_train)


# In[68]:


clf.score(x_test,y_test)


# In[69]:


def params(str):
    print (str)
    return;


# In[70]:


t_sc = np.zeros((params('n_estimators')),dtype=np.float64)


# In[71]:


y_pred = reg.predict(x_test)


# In[92]:


testsc = np.arange(400,)+1


# In[128]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_kc_house_data
from sklearn.model_selection import train_test_split

# Load the Boston Housing dataset for illustration
data = load_kc_house_data()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
regressor.fit(X_train, y_train)

# Once the model is fitted, you can access train_score_
train_score = regressor.train_score_

# Now you can work with train_score or make predictions
y_pred = regressor.predict(X_test)


# In[129]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
plt.plot(testsc,t_sc,'r-',label = 'set dev test')


# In[94]:


from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# In[95]:


pca = PCA()


# In[96]:


pca.fit_transform(scale(train1))


# In[ ]:




