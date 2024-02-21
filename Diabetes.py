#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("./data/diabetes_binary_health_indicators_BRFSS2015.csv")


# In[3]:


df.head()


# ### About columns
# 
# **Diabetes_binary**: you have diabetes (0,1)
# 
# **HighBP**: Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional (0,1)
# 
# **HighChol**: Have you EVER been told by a doctor, nurse or other health professional that your blood cholesterol is high? (0,1)
# 
# **CholCheck**: Cholesterol check within past five years (0,1)
# 
# **BMI**: Body Mass Index (BMI)
# 
# **Smoker**: Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] (0,1)
# 
# **Stroke**: (Ever told) you had a stroke. (0,1)
# 
# **HeartDiseaseorAttack**: Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) (0,1)
# 
# **PhysActivity**: Adults who reported doing physical activity or exercise during the past 30 days other than their regular job (0,1)
# 
# **Fruits**: Consume Fruit 1 or more times per day (0,1)
# 
# **Veggies**: Consume Vegetables 1 or more times per day (0,1)
# 
# **HvyAlcoholConsump**: Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)(0,1)
# 
# **AnyHealthcare**: Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service? (0,1)
# 
# **NoDocbcCost**: Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? (0,1)
# 
# **GenHlth**: Would you say that in general your health is: rate (1 ~ 5)
# 
# **MentHlth**: Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? (0 ~ 30)
# 
# **PhysHlth**: Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0 ~ 30)
# 
# **DiffWalk**: Do you have serious difficulty walking or climbing stairs? (0,1)
# 
# **Sex**: Indicate sex of respondent (0,1) (Female or Male)
# 
# **Age**: Fourteen-level age category (1 ~ 14)
# 
# **Education**: What is the highest grade or year of school you completed? (1 ~ 6)
# 
# **Income**: Is your annual household income from all sources: (If respondent refuses at any income level, code "Refused.") (1 ~ 8)

# ### EDA

# In[4]:


df.shape


# In[5]:


df.isna().sum()


# In[6]:


df.info()


# In[7]:


df.describe().T


# #### All the values need to be converted to int

# In[8]:


df.rename({"Diabetes_binary": "HasDiabetes", "HeartDiseaseorAttack": "MI_CHD"}, axis=1, inplace=True)


# In[9]:


for col in df.columns:
    df[col] = df[col].astype("int8")


# In[10]:


df.info()


# ### Getting number of unique values in all the columns

# In[11]:


unique_vals = {}

for col in df.columns:
    unique_vals[col] = df[col].value_counts().shape[0]

print(pd.DataFrame(unique_vals, index=["n_unique"]).T)


# ### Get value counts of all columns

# In[12]:


for col in df.columns:
    print(df[col].value_counts())
    print("-" * 40)


# #### From the above info, we see many categories in values for "BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"

# ### Checking the outliers

# In[13]:


plt.figure(figsize = (15, 15))

for i, col in enumerate(["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]):
    plt.subplot(4, 2, i+1)
    sns.boxplot(x=col, data=df)
plt.show()


# #### We see that outliers are present in BMI, GenHlth, MentHlth, PhysHlth

# ### Dropping the duplicate records

# In[14]:


# number of duplicated records
df.duplicated().sum()


# In[15]:


df.drop_duplicates(inplace=True)


# In[16]:


df.duplicated().sum()


# In[17]:


df.shape


# In[18]:


df.columns


# ### Exploratory Data Analysis

# In[19]:


plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True, cmap="YlOrRd")
plt.title("Feature correlation");


# #### Fruits, Veggies, HvyAlcoholConsump, Education, Income show a negative coefficient and rest of the columns show a positive coefficient

# ### Getting to know the data better

# In[20]:


df.hist(figsize=(20,15));


# In[21]:


df.drop('HasDiabetes', axis=1) \
    .corrwith(df["HasDiabetes"]) \
    .plot(kind='bar', grid=True, figsize=(20, 8), title="Correlation with Diabetes_binary");


# #### Fruits, AnyHealthcare, NoDocbCost are lease correlated and the rest have a significant correlation with the target

# In[22]:


df.drop(["Fruits", "AnyHealthcare", "NoDocbcCost"], axis=1, inplace=True)


# ### Splitting the data

# In[23]:


X = df.drop(["HasDiabetes"], axis=1)
y = df["HasDiabetes"]


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[26]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[27]:


X_train.columns


# ### Modeling

# #### Logistic Regression

# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


log_reg = LogisticRegression(max_iter=1500).fit(X_train, y_train)


# In[30]:


y_pred = log_reg.predict(X_test)


# In[31]:


from sklearn.metrics import classification_report


# In[32]:


print(classification_report(y_test, y_pred))


# #### Decision Tree Classifier

# In[33]:


from sklearn.tree import DecisionTreeClassifier


# In[34]:


dt_clf = DecisionTreeClassifier().fit(X_train, y_train)


# In[35]:


y_pred = dt_clf.predict(X_test)


# In[36]:


print(classification_report(y_test, y_pred))


# #### KNearestNeighbors

# In[37]:


from sklearn.neighbors import KNeighborsClassifier


# In[38]:


knn = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)


# In[39]:


y_pred = knn.predict(X_test)


# In[40]:


print(classification_report(y_test, y_pred))


# #### RandomForestClassifier

# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


rf_clf = RandomForestClassifier().fit(X_train, y_train)


# In[43]:


y_pred = rf_clf.predict(X_test)


# In[44]:


print(classification_report(y_test, y_pred))


# #### XGBoost

# In[45]:


from xgboost import XGBClassifier


# In[46]:


xgb_clf = XGBClassifier().fit(X_train, y_train)


# In[47]:


y_pred = xgb_clf.predict(X_test)


# In[48]:


print(classification_report(y_test, y_pred))


# ### Choosing the final model

# We can get the baseline performance from Random Forest Classifier
# 
# - The baseline accuracy is 85%
# - Training of models like KNN is slower is are not easily interpretable.
# - XGBoost also gives about the same accuracy as Random Forest
# 
# ### Why Random Forest?
# - Ensemble technique
# - Gives a good accuracy as well as F1 score

# ### Saving the model

# In[49]:


import pickle


# In[50]:


with open("./model.pkl", "wb") as f:
    pickle.dump(rf_clf, f)


# In[ ]:




