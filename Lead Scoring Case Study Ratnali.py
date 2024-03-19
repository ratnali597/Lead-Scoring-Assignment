#!/usr/bin/env python
# coding: utf-8

# ##  Lead Scoring
#                                                                                         
#     
# * An education company named X Education sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses. 
# The company markets its courses on several websites and search engines like Google. Once these people land on the website, they might browse the courses or fill up a form for the course or watch some videos. When these people fill up a form providing their email address or phone number, they are classified to be a lead. Moreover, the company also gets leads through past referrals. Once these leads are acquired, employees from the sales team start making calls, writing emails, etc. Through this process, some of the leads get converted while most do not. The typical lead conversion rate at X education is around 30%. <br>
# * Now, although X Education gets a lot of leads, its lead conversion rate is very poor. For example, if, say, they acquire 100 leads in a day, only about 30 of them are converted. To make this process more efficient, the company wishes to identify the most potential leads, also known as ‘Hot Leads’. If they successfully identify this set of leads, the lead conversion rate should go up as the sales team will now be focusing more on communicating with the potential leads rather than making calls to everyone. A typical lead conversion process can be represented using the following funnel:<br>
# * As you can see, there are a lot of leads generated in the initial stage (top) but only a few of them come out as paying customers from the bottom. In the middle stage, you need to nurture the potential leads well (i.e. educating the leads about the product, constantly communicating etc. ) in order to get a higher lead conversion.<br>
# * X Education has appointed you to help them select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The company requires you to build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.

# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importing Important/required Liabraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",200)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reading data leads.csv

# In[2]:


Leads_Data=pd.read_csv("leads.csv")


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Checking size of the data

# In[3]:


print("Data Shape is : ",Leads_Data.shape[0],"Rows and",Leads_Data.shape[1],"Columns")


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Checking info of all available columns

# In[4]:


print(Leads_Data.info())


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Replacing "Select" with NaN values, as those are the values which are not filled in while filling the form

# In[5]:


Leads_Data = Leads_Data.replace('Select', np.nan)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Checking top rows of data

# In[6]:


Leads_Data.head()


# 
# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Chekcing Stats of dataset

# In[7]:


Leads_Data.describe()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Checking for any missing values in columns

# In[8]:


Null_values=Leads_Data.isnull().sum()/Leads_Data.shape[0]*100


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# % Summary of each columns having null values 

# In[9]:


Null_values.sort_values(ascending=False)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Dropping the columns having null values more than 40%

# In[10]:


null_40=(Null_values[Null_values>40]).index.to_list()
Leads_Data.drop(columns=null_40,inplace=True)
print("Application data new shape : ", Leads_Data.shape)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# In[11]:


Leads_Data.info()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# %Summary of the columns having null values pending after removing the high% null vlaue columns

# In[12]:


Null_values=Leads_Data.isnull().sum()/Leads_Data.shape[0]*100
Null_values=Null_values.sort_values(ascending=False)
Null_values


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# List of the columns having null values

# In[13]:


Null_values=Null_values[Null_values>0]
Null_values=Null_values.index.to_list()
Null_values


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Imputing mode in categorical columns and median in numerical columns at place of null values.

# In[14]:


for col in Null_values:
    if Leads_Data[col].dtype=="object":
        Leads_Data[col]=Leads_Data[col].fillna(Leads_Data[col].mode()[0])
    else:
        Leads_Data[col]=Leads_Data[col].fillna(Leads_Data[col].median())


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Checking final null values in each column

# In[15]:


Leads_Data.isna().sum()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Summary of Unique value count in each column

# In[16]:


Leads_Data.nunique().sort_values(ascending=False)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Columns having 1 unique value, means every value is same, so no use of keeping them, we may drop these

# In[17]:


Unique_Count=Leads_Data.nunique()
Single_unique=Unique_Count[Unique_Count==1]
Single_unique=Single_unique.index.to_list()
Single_unique


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Dropping these columns having single value in column.

# In[18]:


Leads_Data.drop(columns=Single_unique,axis=1,inplace=True)
Leads_Data.shape


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Creating summary having type and unique count to lookinto rest of the fields

# In[19]:


Col_Type_With_Unique_Value_Counts=Leads_Data.nunique().sort_values(ascending=False)
Col_Type_With_Unique_Value_Counts=Col_Type_With_Unique_Value_Counts.reset_index()
Col_Type_With_Unique_Value_Counts=Col_Type_With_Unique_Value_Counts.rename(columns={"index":"Col_name",0:"unique_val_count"})
Col_Type_With_Unique_Value_Counts["dtype"]=Col_Type_With_Unique_Value_Counts["Col_name"].apply(lambda x: Leads_Data[x].dtype)
Col_Type_With_Unique_Value_Counts


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Dropping columns having unique values, as every value is different and ID types, these may not help keeping in the database. we should remove these.

# In[20]:


Leads_Data.drop(columns=["Prospect ID","Lead Number"],inplace=True)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Lest see the count of values in every column to check imbalance

# In[21]:


for col in Leads_Data:
    print(Leads_Data[col].value_counts())
    print("---------------------------------------------------------------")


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# These columns have almost same values(except handful different), we may drop these columns

# In[22]:


Leads_Data.drop(columns=["Do Not Call","Through Recommendations","Digital Advertisement","Newspaper","X Education Forums","Newspaper Article","Search","What matters most to you in choosing a course"],inplace=True)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Creating list of Categorical and numerical columns, which will help us in EDA

# In[23]:


cat_cols=Leads_Data.select_dtypes(include="object").columns.to_list()
num_cols=Leads_Data.select_dtypes(include=["float64","int64"]).columns.to_list()
num_cols.remove("Converted")


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Summary of Leads converted and not converted

# In[24]:


Leads_Data.Converted.value_counts()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# In[25]:


Leads_Data.info()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Correlation between numerical features

# In[26]:


sns.heatmap(data=Leads_Data[num_cols].corr(),cmap='Greens',annot=True)
plt.show()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# In[27]:


sns.pairplot(data=Leads_Data,diag_kind='kde',hue='Converted')
plt.show()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Checking outliers in numerical features

# In[28]:


for col in num_cols:
    plt.figure(figsize=(9,2))
    sns.boxplot(x=Leads_Data[col])
    plt.show()
    print("-"*90)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Comparison of leads converted for each categorical feature

# In[29]:


for col in cat_cols:
    plt.figure(figsize=(12,6))
    ax=sns.countplot(data=Leads_Data,x=col,palette='Accent',hue=Leads_Data.Converted)
    for x in ax.containers:
        ax.bar_label(x,rotation=60)
    plt.xticks(rotation=60)
    plt.show()
    print("X"*120)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# ## Data Prepration for model building

# In[30]:


Leads_Data = pd.get_dummies(data=Leads_Data,columns=cat_cols,drop_first=True)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Converting dummy boolean column to int type

# In[31]:


cols=Leads_Data.columns.to_list()
for col in cols:
    if Leads_Data[col].dtype=="bool":
        Leads_Data[col]=Leads_Data[col].astype("uint8")
    


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# In[32]:


Leads_Data.head()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Total columns after creating dummy columns

# In[33]:


Leads_Data.columns.to_list()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importing important libraries required in predictive model building.

# In[34]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# New Shape of dataset

# In[35]:


Leads_Data.shape


# Seperating Dependent and non dependent columns for splitting the train and test data

# In[36]:


y =Leads_Data['Converted']
X=Leads_Data.drop('Converted', axis=1)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Spliting data into Train and Test

# In[37]:


#Train Test split with 70:30 ratio
X_train, X_test,y_train,y_test = train_test_split(X,y,train_size=0.7, random_state=100)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Shape of Train and Test data sets after split

# In[38]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Let us scale continuous variables and Fit and transform training set only

# In[39]:


scaler=StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_train.head()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Looking at X_train stats

# In[40]:


X_train.describe()


# We can clearly see all the values are sclaed between 0 and 1
# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Lets see sample data too

# In[41]:


X_train.head()


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Building Model with RFE

# In[42]:


import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE


# lets find the top 15 features which may has correlation with the target feature

# In[43]:


rfe = RFE(logreg, n_features_to_select=20)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# below are the fields selected or rejected for RFE

# In[44]:


#Columns selected by RFE and their weights
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# VIF Function definition of getting compatibility of the features in RFE model

# In[45]:


#Function to calculate VIFs and print them -Takes the columns for which VIF to be calcualted as a parameter
def get_vif(cols):
    df1 = X_train[cols]
    vif = pd.DataFrame()
    vif['Features'] = df1.columns
    vif['VIF'] = [variance_inflation_factor(df1.values, i) for i in range(df1.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    print(vif.sort_values(by='VIF',ascending=False))


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating list of columns selected for RFE model building process

# In[46]:


#Print Columns selected by RFE. We will manually eliminate for these columns
X_train_rfe=X_train.columns[rfe.support_].tolist()
print(X_train_rfe)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# List of features not selected for RFE model building

# In[47]:


# Features not selected by RFE
X_train_rfe_exlusion=X_train.columns[~rfe.support_].tolist()
print(X_train_rfe_exlusion)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# # Creating Models
# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Model 1

# In[48]:


#Selected columns for Model 1 - all columns selected by RFE
rfe_cols=['Lead Origin_Lead Add Form', 'Do Not Email_Yes', 
             'Last Activity_Converted to Lead', 
             'Last Activity_Olark Chat Conversation',
             'What is your current occupation_Unemployed', 
             'What is your current occupation_Working Professional',
             'Tags_Busy', 'Tags_Closed by Horizzon',
             'Tags_Interested in Next batch', 'Tags_Lateral student',
             'Tags_Lost to EINS', 'Tags_Ringing', 'Tags_Will revert after reading the email',
             'Tags_in touch with EINS', 'Tags_invalid number',
             'Tags_switched off', 'Tags_wrong number given',
             'Last Notable Activity_Email Bounced', 'Last Notable Activity_Had a Phone Conversation', 
             'Last Notable Activity_SMS Sent']
X_train_sm = sm.add_constant(X_train[rfe_cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
print(res.summary())
get_vif(rfe_cols)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Model 2

# Drop "Tags_Interested in Next batch " due to High P-Score

# In[49]:


rfe_cols=['Lead Origin_Lead Add Form', 'Do Not Email_Yes', 
             'Last Activity_Converted to Lead', 
             'Last Activity_Olark Chat Conversation',
             'What is your current occupation_Unemployed', 
             'What is your current occupation_Working Professional',
             'Tags_Busy', 'Tags_Closed by Horizzon',
              'Tags_Lateral student',
             'Tags_Lost to EINS', 'Tags_Ringing', 'Tags_Will revert after reading the email',
             'Tags_in touch with EINS', 'Tags_invalid number',
             'Tags_switched off', 'Tags_wrong number given',
             'Last Notable Activity_Email Bounced', 'Last Notable Activity_Had a Phone Conversation', 
             'Last Notable Activity_SMS Sent']
X_train_sm = sm.add_constant(X_train[rfe_cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
print(res.summary())
get_vif(rfe_cols)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Model 3

# Drop "Tags_Lateral student" due to High P-Score

# In[50]:


rfe_cols=['Lead Origin_Lead Add Form', 'Do Not Email_Yes', 
             'Last Activity_Converted to Lead', 
             'Last Activity_Olark Chat Conversation',
             'What is your current occupation_Unemployed', 
             'What is your current occupation_Working Professional',
             'Tags_Busy', 'Tags_Closed by Horizzon',
             'Tags_Lost to EINS', 'Tags_Ringing', 'Tags_Will revert after reading the email',
             'Tags_in touch with EINS', 'Tags_invalid number',
             'Tags_switched off', 'Tags_wrong number given',
             'Last Notable Activity_Email Bounced', 'Last Notable Activity_Had a Phone Conversation', 
             'Last Notable Activity_SMS Sent']
X_train_sm = sm.add_constant(X_train[rfe_cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
print(res.summary())
get_vif(rfe_cols)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Model 4

# Drop "Tags_wrong number given" due to High P-Score

# In[51]:


rfe_cols=['Lead Origin_Lead Add Form', 'Do Not Email_Yes', 
             'Last Activity_Converted to Lead', 
             'Last Activity_Olark Chat Conversation',
             'What is your current occupation_Unemployed', 
             'What is your current occupation_Working Professional',
             'Tags_Busy', 'Tags_Closed by Horizzon',
             'Tags_Lost to EINS', 'Tags_Ringing', 'Tags_Will revert after reading the email',
             'Tags_in touch with EINS', 'Tags_invalid number',
             'Tags_switched off',
             'Last Notable Activity_Email Bounced', 'Last Notable Activity_Had a Phone Conversation', 
             'Last Notable Activity_SMS Sent']
X_train_sm = sm.add_constant(X_train[rfe_cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
print(res.summary())
get_vif(rfe_cols)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Model 5

# Drop "Tags_invalid number" due to High P-Score

# In[52]:


rfe_cols=['Lead Origin_Lead Add Form', 'Do Not Email_Yes', 
             'Last Activity_Converted to Lead', 
             'Last Activity_Olark Chat Conversation',
             'What is your current occupation_Unemployed', 
             'What is your current occupation_Working Professional',
             'Tags_Busy', 'Tags_Closed by Horizzon',
             'Tags_Lost to EINS', 'Tags_Ringing', 'Tags_Will revert after reading the email',
             'Tags_in touch with EINS', 
             'Tags_switched off',
             'Last Notable Activity_Email Bounced', 'Last Notable Activity_Had a Phone Conversation', 
             'Last Notable Activity_SMS Sent']
X_train_sm = sm.add_constant(X_train[rfe_cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
print(res.summary())
get_vif(rfe_cols)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Model 6

# Drop "Last Notable Activity_Had a Phone Conversation" due to High P-Score

# In[53]:


rfe_cols=['Lead Origin_Lead Add Form', 'Do Not Email_Yes', 
             'Last Activity_Converted to Lead', 
             'Last Activity_Olark Chat Conversation',
             'What is your current occupation_Unemployed', 
             'What is your current occupation_Working Professional',
             'Tags_Busy', 'Tags_Closed by Horizzon',
             'Tags_Lost to EINS', 'Tags_Ringing', 'Tags_Will revert after reading the email',
             'Tags_in touch with EINS', 
             'Tags_switched off',
             'Last Notable Activity_Email Bounced', 
             'Last Notable Activity_SMS Sent']
X_train_sm = sm.add_constant(X_train[rfe_cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
print(res.summary())
get_vif(rfe_cols)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Model 7

# Drop "Tags_switched off" due to High P-Score

# In[54]:


rfe_cols=['Lead Origin_Lead Add Form', 'Do Not Email_Yes', 
             'Last Activity_Converted to Lead', 
             'Last Activity_Olark Chat Conversation',
             'What is your current occupation_Unemployed', 
             'What is your current occupation_Working Professional',
             'Tags_Busy', 'Tags_Closed by Horizzon',
             'Tags_Lost to EINS', 'Tags_Ringing', 'Tags_Will revert after reading the email',
             'Tags_in touch with EINS', 
             'Last Notable Activity_Email Bounced', 
             'Last Notable Activity_SMS Sent']
X_train_sm = sm.add_constant(X_train[rfe_cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
print(res.summary())
get_vif(rfe_cols)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Model 8

# Drop "Last Notable Activity_Email Bounced" due to High P-Score

# In[55]:


rfe_cols=['Lead Origin_Lead Add Form', 'Do Not Email_Yes', 
             'Last Activity_Converted to Lead', 
             'Last Activity_Olark Chat Conversation',
             'What is your current occupation_Unemployed', 
             'What is your current occupation_Working Professional',
             'Tags_Busy', 'Tags_Closed by Horizzon',
             'Tags_Lost to EINS', 'Tags_Ringing', 'Tags_Will revert after reading the email',
             'Tags_in touch with EINS', 
             'Last Notable Activity_SMS Sent']
X_train_sm = sm.add_constant(X_train[rfe_cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
print(res.summary())
get_vif(rfe_cols)


# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Model 9

# Drop "Tags_Ringing" due to High P-Score

# In[56]:


rfe_cols=['Lead Origin_Lead Add Form', 'Do Not Email_Yes', 
             'Last Activity_Converted to Lead', 
             'Last Activity_Olark Chat Conversation',
             'What is your current occupation_Unemployed', 
             'What is your current occupation_Working Professional',
             'Tags_Busy', 'Tags_Closed by Horizzon',
             'Tags_Lost to EINS', 'Tags_Will revert after reading the email',
             'Tags_in touch with EINS', 
             'Last Notable Activity_SMS Sent']
X_train_sm = sm.add_constant(X_train[rfe_cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
print(res.summary())
get_vif(rfe_cols)


# All p-scores and VIF are low now, we can proceed with these features now.

# ###### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# Predicting dependent variable on train data.

# In[57]:


y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# Predicting dependent variable on train data.

# In[58]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# Creating new dataframe having Converted status and predicted status

# In[59]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# Predicting dependent variable on train data.

# In[60]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# Generating Confusion matrix 

# In[61]:


from sklearn import metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# Let's check the overall accuracy.

# In[62]:


print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[63]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# Let's see the sensitivity of our logistic regression model

# In[64]:


TP / float(TP+FN)


# Let us calculate specificity

# In[65]:


TN / float(TN+FP)


# Calculate False Postive Rate - predicting conversion when customer does not have convert

# In[66]:


print(FP/ float(TN+FP))


# positive predictive value 

# In[67]:


print (TP / float(TP+FP))


# Negative predictive value

# In[68]:


print (TN / float(TN+ FN))


# ##### PLOTTING ROC CURVE

# In[69]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[70]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[71]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# The ROC Curve should be a value close to 1. We are getting a good value of 0.92 indicating a good predictive model.

# ##### Finding Optimal Cutoff Point

# Above we had chosen an arbitrary cut-off value of 0.5. We need to determine the best cut-off value and the below section deals with that:

# In[72]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

# In[73]:


cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# Let's plot accuracy sensitivity and specificity for various probabilities.

# In[74]:


cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# From the curve above, 0.35 is the optimum point to take it as a cutoff probability.

# In[75]:


y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.35 else 0)

y_train_pred_final.head()


# In[76]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# Let's check the overall accuracy.

# In[77]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[78]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[79]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# Let's see the sensitivity of our logistic regression model

# In[80]:


TP / float(TP+FN)


# Let us calculate specificity

# In[81]:


TN / float(TN+FP)


# Observation:
# 
# So as we can see above the model seems to be performing well. The ROC curve has a value of 0.92, which is very good. We have the following values for the Train Data:
# 
# 
# Accuracy : 79.17%
# 
# Sensitivity : 93.34%
# 
# Specificity : 70.43%
# 
# Some of the other Stats are derived below, indicating the False Positive Rate, Positive Predictive Value,Negative Predictive Values, Precision & Recall.

# Calculate False Postive Rate - predicting conversion when customer does not have convert

# In[82]:


print(FP/ float(TN+FP))


# Positive predictive value 

# In[83]:


print (TP / float(TP+FP))


# Negative predictive value

# In[84]:


print (TN / float(TN+ FN))


# Looking at the confusion matrix again

# In[85]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# Precision

# In[86]:


TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# Recall

# In[87]:


TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[88]:


from sklearn.metrics import precision_score, recall_score


# In[89]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[90]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[91]:


from sklearn.metrics import precision_recall_curve


# In[92]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[93]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# the cutoff is 0.38 from above precision and recall chart, lets check accuracy at cutoff 0.38

# In[94]:


y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.38 else 0)

y_train_pred_final.head()


# In[95]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# Let's check the overall accuracy.

# In[96]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[97]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[98]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# Let's see the sensitivity of our logistic regression model

# In[99]:


TP / float(TP+FN)


# Let us calculate specificity

# In[100]:


TN / float(TN+FP)


# There is no change at 0.35 and 0.38 cutoff, so we can go by either on them.

# Scaling test set

# In[101]:


num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[102]:


X_test = X_test[rfe_cols]
X_test.head()


# In[103]:


X_test_sm = sm.add_constant(X_test[rfe_cols])


# ##### PREDICTIONS ON TEST SET

# In[104]:


y_test_pred = res.predict(X_test_sm)


# In[105]:


y_test_pred[:10]


# Converting y_pred to a dataframe which is an array

# In[106]:


y_pred_1 = pd.DataFrame(y_test_pred)


# Let's see the head

# In[107]:


y_pred_1.head()


# Converting y_test to dataframe

# In[108]:


y_test_df = pd.DataFrame(y_test)


# Putting CustID to index

# In[109]:


y_test_df['Prospect ID'] = y_test_df.index


# Removing index for both dataframes to append them side by side 

# In[110]:


y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# Appending y_test_df and y_pred_1

# In[111]:


y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[112]:


y_pred_final.head()


# Renaming the column 

# In[113]:


y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[114]:


y_pred_final.head()


# Rearranging the columns

# In[115]:


y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# Let's see the head of y_pred_final

# In[116]:


y_pred_final.head()


# In[117]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.38 else 0)


# In[118]:


y_pred_final.head()


# Let's check the overall accuracy.

# In[119]:


metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[120]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[121]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# Let's see the sensitivity of our logistic regression model

# In[122]:


TP / float(TP+FN)


# Let us calculate specificity

# In[123]:


TN / float(TN+FP)


# In[124]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[125]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# Final Observation:
# Let us compare the values obtained for Train & Test:
#    
# Train Data:
#     
# Accuracy : 79.17%<br>
# Sensitivity : 93.34%<br>
# Specificity : 70.43%<br>
# 
# Test Data: 
# Accuracy : 79.43%<br>
# Sensitivity : 93.69%<br>
# Specificity : 70.12%<br>
#     
# The Model seems to predict the Conversion Rate very well and we should be able to give the CEO confidence in making good calls based on this model
# 
# Probabilty Threshold/cutoff 
# 
# Final Features:
# 
# Lead Origin_Lead Add Form                                
# Do Not Email_Yes                                        
# Last Activity_Converted to Lead                         
# Last Activity_Olark Chat Conversation                   
# What is your current occupation_Unemployed              
# What is your current occupation_Working Professional    
# Tags_Busy                                               
# Tags_Closed by Horizzon                                 
# Tags_Lost to EINS                                       
# Tags_Will revert after reading the email                
# Tags_in touch with EINS                                 
# Last Notable Activity_SMS Sent

# In[ ]:




