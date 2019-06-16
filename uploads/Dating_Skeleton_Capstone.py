#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn as sk

from sklearn.preprocessing import MinMaxScaler

import re

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# In[2]:


#Create your df here

df = pd.read_csv("profiles.csv")


# In[3]:


#Explore the data

df[{'drinks'}]


# In[4]:


#Visualize some of the data

plt.hist(df.age, bins=50)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(18, 65)
plt.show()


# In[5]:


#Formulate a question

df.smokes.unique()


# In[6]:


#Augment your data
drink_mapping = {"not at all":1, "rarely":2, "socially":3, "often":4, "very often":5, "desperately":6}
df["drinks_code"] = df.drinks.map(drink_mapping)

smokes_mapping = {"no":0, "trying to quit":1, "when drinking":2, "sometimes":3, "yes":4}
df["smokes_code"] = df.smokes.map(smokes_mapping)

drugs_mapping = {"never":0, "sometimes":1, "often":2}
df["drugs_code"] = df.drugs.map(drugs_mapping)

#Making essay list
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

#Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)

#Combine essays into array
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df["all_essays"] = all_essays

#Add essay_len column to dataframe
df["essay_len"] = all_essays.apply(lambda x: len(x))


# In[7]:


#Finding average word length in all_essays

#total word count
totalwords = df["all_essays"].str.split().str.len()

#total character count
characters = df["all_essays"].str.replace(' ','')
df['characters'] = characters


p = re.compile(r'[^\w\s]+')
df['characters'] = [p.sub('',x) for x in df['characters'].tolist()]

countcharacters = df["characters"].str.len()

average = countcharacters / totalwords

df["average_word_length"] = average


# In[8]:


#Normalize the data

#feature_data = df[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'average_word_length']]

#x = feature_data.values
#min_max_scaler = MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)

#feature_data = pd.DataFrame(x_scaled, columns = feature_data.columns)


# In[9]:


#Exploratory Analysis

print(df.age.unique())
print(df.drinks_code.unique())
print(df.body_type.unique())
print(df.status.unique())
print(df.average_word_length.unique())
print(df.essay_len.unique())


# In[10]:


#Create Variables - NEED ATLEAST TWO NEW COLUMNS

#New map for body_type and status
body_mapping = {'used up':1, 'thin':2, 'skinny':3, 'average':4, 'athletic':5, 'fit':6, 'jacked':7, 'a little extra':8, 'curvy':9, 'full figured':10, 'overweight':11, 'rather not say':12}
df['body_code'] = df.body_type.map(body_mapping)


status_mapping = {'single':1, 'available':2, 'seeing someone':3, 'married':4, 'unknown':5}
df['status_code'] = df.status.map(status_mapping)


#Check new columns of df
#print(df.body_code.unique())
#print(df.drinks_code.unique())
#print(df.status_code.unique())
#print(df.average_word_length.isnull().values.any())
#print(df.essay_len.isnull().values.any())
#print(df.age.isnull().values.any())

#Get rid of rows of Nan data points in dataframe
df.dropna(axis=0,subset=['body_code','drinks_code','status_code','average_word_length','essay_len','age'],inplace=True)

#Check Nan deletes
print(df.body_code.isnull().values.any())
print(df.drinks_code.isnull().values.any())
print(df.status_code.isnull().values.any())
print(df.average_word_length.isnull().values.any())
print(df.essay_len.isnull().values.any())
print(df.age.isnull().values.any())


# In[11]:


#Exploratory Analysis - NEED TWO GRAPHS
#Histograms to see frequency of various data set categories
plt.hist(df.body_code, bins=12)
plt.xlabel("Body Type")
plt.ylabel("Frequency")
plt.xlim(1, 12)
plt.show()

plt.hist(df.drinks_code, bins=6)
plt.xlabel("Drinking Amount")
plt.ylabel("Frequency")
plt.xlim(1, 6)
plt.show()

plt.hist(df.status_code, bins=5)
plt.xlabel("Status")
plt.ylabel("Frequency")
plt.xlim(1, 5)
plt.show()

plt.hist(df.average_word_length, bins=5000)
plt.xlabel("AVG Essay Word Length")
plt.ylabel("Frequency")
plt.xlim(1, 10)
plt.show()

plt.hist(df.essay_len, bins=2000)
plt.xlabel("Essay Length")
plt.ylabel("Frequency")
plt.xlim(1, 10000)
plt.show()


# In[12]:


#Scatter plot of drinks and weights by age
df['drink_weight'] = df.drinks_code * df.body_code

#print(df.body_code[:1])
#print(df.drinks_code[:1])
#print(df.drink_weight[:1])

x = df.drink_weight
y = df.age

plt.scatter(x,y,alpha=.2)
plt.xlabel("DrinkWeight")
plt.ylabel("Age")
plt.title("DrinkWeight by Age")
plt.show()

x = df.body_code
y = df.age
plt.scatter(x,y,alpha=0.2)
plt.xlabel("Body Type")
plt.ylabel("Age")
plt.title("BodyType by Age")
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['average_word_length'], df['essay_len'], df['age'], cmap=plt.cm.viridis, linewidth=0.2)
surf=ax.plot_trisurf(df['average_word_length'], df['essay_len'], df['age'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
ax.view_init(30, 45)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['average_word_length'], df['essay_len'], df['age'], c='skyblue', s=60)
ax.view_init(30, 45)
plt.show()


# In[13]:


#Use classification techniques - NEED TWO: 
#1.comparison between 2. qualitative discussion on simplicity, run time, accuracy/precison/recall

#Can we predict age based on body_type and drinkweight? (columns: age, body_type, drink_weight)
#Can we predict status based on avg word length and essay length? (columns: status, average_word_length, essay_len)

#Set feature data sets and labels
feature_data1 = df[['body_code','drink_weight']]
feature_labels1 = df['age']

feature_data2 = df[['average_word_length','essay_len']]
feature_labels2 = df['status_code']

#Normalize the data sets (also tried using min max scaling and precison and accuracy did not improve, tried Robust scaler)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x1 = feature_data1.values
x1_scaled = scaler.fit_transform(x1)
feature_data1 = pd.DataFrame(x1_scaled, columns = feature_data1.columns)

x2 = feature_data2.values
x2_scaled = scaler.fit_transform(x2)
feature_data2 = pd.DataFrame(x2_scaled, columns = feature_data2.columns)

#Create the training and test data sets
from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(feature_data1, feature_labels1, test_size = 0.20, random_state = 1)
x2_train, x2_test, y2_train, y2_test = train_test_split(feature_data2, feature_labels2, test_size = 0.20, random_state = 1)

print(feature_data1)
print(feature_data2)
print(feature_labels1)
print(feature_labels2)

print('x1train', x1_train)
print('x1test',x1_test)
print('y1train',y1_train)
print('y1test',y1_test)

print('x2train', x2_train)
print('x2test',x2_test)
print('y2train',y2_train)
print('y2test',y2_test)


# In[14]:


#Support Vector Machine Classification # NEED TO CHANGE THIS FOR SCALING BC IT IS WRONG
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


svclassifier1 = SVC(kernel = 'rbf', cache_size = 200, gamma='auto')
svclassifier2 = SVC(kernel = 'rbf', cache_size = 200, gamma='auto')

svclassifier1.fit(x1_train, y1_train)
svclassifier2.fit(x2_train, y2_train)

y1_predict = svclassifier1.predict(x1_test)
y2_predict = svclassifier2.predict(x2_test)

#y1_predict = scaler.inverse_transform(y1_predict)

print(y1_predict)
print(y2_predict)

from sklearn.metrics import classification_report, confusion_matrix  #(runs slowly using the cache_size set at 500)

print(confusion_matrix(y1_test,y1_predict))
print(classification_report(y1_test,y1_predict))

print(confusion_matrix(y2_test,y2_predict))
print(classification_report(y2_test,y2_predict))

print(f1_score(y1_test, y1_predict, average="macro"))
print(f1_score(y2_test, y2_predict, average="macro"))

print(precision_score(y1_test, y1_predict, average="macro"))
print(precision_score(y2_test, y2_predict, average="macro"))

print(recall_score(y1_test, y1_predict, average="macro"))
print(recall_score(y2_test, y2_predict, average="macro"))


# In[15]:


#K Nearest Neighbors

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 3) #(tried n at 3, 5, 7)
classifier.fit(x1_train, y1_train)
classifier.fit(x2_train, y2_train)

Ky1_predict = classifier.predict(x1_test)
Ky2_predict = classifier.predict(x2_test)


print(Ky1_predict)
print(Ky2_predict)

from sklearn.metrics import classification_report, confusion_matrix 

print(confusion_matrix(y1_test,Ky1_predict))
print(classification_report(y1_test,Ky1_predict))

print(confusion_matrix(y2_test,Ky2_predict))
print(classification_report(y2_test,Ky2_predict))

print(f1_score(y1_test, Ky1_predict, average="macro"))
print(f1_score(y2_test, Ky2_predict, average="macro"))

print(precision_score(y1_test, Ky1_predict, average="macro"))
print(precision_score(y2_test, Ky2_predict, average="macro"))

print(recall_score(y1_test, Ky1_predict, average="macro"))
print(recall_score(y2_test, Ky2_predict, average="macro"))


# In[16]:


#Use Regression techniques
#NEED TWO: 1.comparison between 2. qualitative discussion on simplicity, run time, accuracy/precison/recall

#Predict age based on body_type and drinkweight (columns: age, body_type, drink_weight)
#Predict status based on avg word length and essay length (columns: status, average_word_length, essay_len)

from sklearn.linear_model import LinearRegression

lineregr = LinearRegression()

lineregr_model1 = lineregr.fit(x1_train, y1_train)
lineregr_model2 = lineregr.fit(x2_train, y2_train)

r_sq1 = lineregr_model1.score(x1_train, y1_train)
r_sq2 = lineregr_model2.score(x2_train, y2_train)

print('coefficient of determination model 1:', r_sq1)
print('coefficient of determination model 2:', r_sq2)

print('intercept:', lineregr_model1.intercept_)
print('intercept:', lineregr_model2.intercept_)

print('slope:', lineregr_model1.coef_)
print('slope:', lineregr_model2.coef_)

y1_predict = lineregr_model1.predict(x1_test)
y2_predict = lineregr_model2.predict(x2_test)

print('predicted response:', y1_predict, sep='\n')
print('predicted response:', y2_predict, sep='\n')


# In[17]:


#Plot linear regression predictions

#Frustartion with array reshaping in order to inverse transform the predicted values. Unsure if this is necessary or not.

#y1_predict = y1_predict.reshape(1,-1)
#y1_predict = scaler.inverse_transform(y1_predict)

df1 = pd.DataFrame({'Actual': y1_test, 'Predicted': y1_predict})
print(df1[{'Actual', 'Predicted'}])

df2 = pd.DataFrame({'Actual': y2_test, 'Predicted': y2_predict})
print(df2[{'Actual', 'Predicted'}])

plt.scatter(y1_test, y1_predict)
plt.xlabel('Age')
plt.ylabel('Predicted Age')
plt.show()

plt.scatter(y2_test, y2_predict)
plt.xlabel('Status')
plt.ylabel('Predicted Status')
plt.show()


# In[18]:


#Use Regression techniques

from sklearn.neighbors import KNeighborsRegressor

Kregr = KNeighborsRegressor(n_neighbors=3, weights='distance')

Kregr1 = Kregr.fit(x1_train, y1_train)
Kregr2 = Kregr.fit(x2_train, y2_train)

print(Kregr1.score(x1_test, y1_test))
print(Kregr2.score(x2_test, y2_test))

y1_predict = Kregr1.predict(x1_test)
y2_predict = Kregr2.predict(x2_test)

print('predicted response:', y1_predict, sep='\n')
print('predicted response:', y2_predict, sep='\n')


# In[19]:


#Plot knearest regression predictions

df1 = pd.DataFrame({'Actual': y1_test, 'Predicted': y1_predict})
print(df1[{'Actual', 'Predicted'}])

df2 = pd.DataFrame({'Actual': y2_test, 'Predicted': y2_predict})
print(df2[{'Actual', 'Predicted'}])

plt.scatter(y1_test, y1_predict)
plt.xlabel('Age')
plt.ylabel('Predicted Age')
plt.show()

plt.scatter(y2_test, y2_predict)
plt.xlabel('Status')
plt.ylabel('Predicted Status')
plt.show()


# In[ ]:




