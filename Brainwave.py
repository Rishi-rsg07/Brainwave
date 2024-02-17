#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


#Relevant packages 
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score 

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[3]:


dataset = pd.read_csv("/kaggle/input/depression-anxiety-stress-scales-responses/data.csv", delimiter='\t')


# In[4]:


dataset.head(10)


# In[5]:


dataset.info()


# In[6]:


# Check for duplicates on each row.
dataset.duplicated().value_counts()


# In[7]:


# Check for unique values.
dataset.nunique()


# In[8]:


columns  = dataset.columns
print('Attributes',columns)


# In[9]:


for column in columns:
    print(column)


# # **Data Preprocessing**

# In[10]:


# Extract columns matching the pattern "Q<number>A"
pattern = r'^Q\d+A$'
scale_column = [column for column in dataset.columns if re.match(pattern, column)]
# Create a new DataFrame with the extracted columns
extracted_data = dataset[scale_column]


# In[11]:


extracted_data


# In[12]:


#Check if theres any missing or empty item
extracted_data.isnull().sum()


# In[13]:


#Subtract 1 from all the response to change the scale from 1 to 4 to 0 to 3
def sub(data):
    return data.subtract(1,axis=1)
regularized_dataset=sub(extracted_data) 

# Declare the question key to generate the dataset for each dataset
DASS_keys = {'Depression': [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
             'Anxiety': [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
             'Stress': [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]}
Depression_keys = []
for i in DASS_keys["Depression"]:
    Depression_keys.append('Q'+str(i)+'A')
Stress_keys = []
for i in DASS_keys["Stress"]:
    Stress_keys.append('Q'+str(i)+'A')
Anxiety_keys = []
for i in DASS_keys["Anxiety"]:
    Anxiety_keys.append('Q'+str(i)+'A')
depression_dataset= regularized_dataset.filter(Depression_keys)
stress_dataset = regularized_dataset.filter(Stress_keys)
anxiety_dataset = regularized_dataset.filter(Anxiety_keys)


# In[14]:


#Obtain the total score for each dataset here
def scores(data):
    col=list(data)
    data['Total_Count']=data[col].sum(axis=1)
    return data
depression_dataset=scores(depression_dataset)
stress_dataset=scores(stress_dataset)
anxiety_dataset=scores(anxiety_dataset)


# **Display the newly generated datasets**

# In[15]:


depression_dataset.head(10)


# In[16]:


stress_dataset.head(10)


# In[17]:


anxiety_dataset.head(10)


# # **Depression Dataset**

# In[18]:


#Declaring function to assign the label
def condition(x):
    if x<=9:
        return 'Normal'
    if  10<=x<=13:
        return 'Mild'
    if 14<=x<=20:
        return 'Moderate'
    if 21<=x<=27:
        return 'Severe'
    if x>=28:
        return 'Extremely Severe'

#Apply the condition and drop the "Total_Count" column
depression_dataset['Label']=depression_dataset['Total_Count'].apply(condition)
final_depression_dataset = depression_dataset.drop(columns=['Total_Count'])
final_depression_dataset.head(10)


# In[19]:


# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

# Count the frequency of each label
label_counts = final_depression_dataset['Label'].value_counts()
print(label_counts.reindex(desired_labels))

# Define the colors for each bar
colors = ['skyblue', 'green', 'yellow', 'orange', 'gray']

# Reorder the label counts based on the desired arrangement
label_counts_ordered = label_counts.reindex(desired_labels)

# Plot the bar chart
plt.bar(label_counts_ordered.index, label_counts_ordered.values, color=colors)

# Plot the bar chart
# plt.bar(label_counts.index, label_counts.values, color=colors)

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.title('Depression Dataset Distribution of Labels')

# Show the plot
plt.show()


# # **Stress Dataset**

# In[20]:


#Declaring function to assign the label
def condition(x):
    if x<=14:
        return 'Normal'
    if  15<=x<=18:
        return 'Mild'
    if 19<=x<=25:
        return 'Moderate'
    if 26<=x<=33:
        return 'Severe'
    if x>=34:
        return 'Extremely Severe'

#Apply the condition and drop the "Total_Count" column
stress_dataset['Label']=stress_dataset['Total_Count'].apply(condition)
final_stress_dataset = stress_dataset.drop(columns=['Total_Count'])
final_stress_dataset.head(10)


# In[21]:


# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

# Count the frequency of each label
label_counts = final_stress_dataset['Label'].value_counts()
print(label_counts.reindex(desired_labels))

# Define the colors for each bar
colors = ['skyblue', 'green', 'yellow', 'orange', 'gray']

# Reorder the label counts based on the desired arrangement
label_counts_ordered = label_counts.reindex(desired_labels)

# Plot the bar chart
plt.bar(label_counts_ordered.index, label_counts_ordered.values, color=colors)

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.title('Stress Dataset Distribution of Labels')

# Show the plot
plt.show()


# # **Anxiety Dataset**

# In[22]:


#Declaring function to assign the label
def condition(x):
    if x<=7:
        return 'Normal'
    if  8<=x<=9:
        return 'Mild'
    if 10<=x<=14:
        return 'Moderate'
    if 15<=x<=19:
        return 'Severe'
    if x>19:
        return 'Extremely Severe'

#Apply the condition and drop the "Total_Count" column
anxiety_dataset['Label']=anxiety_dataset['Total_Count'].apply(condition)
final_anxiety_dataset = anxiety_dataset.drop(columns=['Total_Count'])
final_anxiety_dataset.head(10)


# In[23]:


# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

# Count the frequency of each label
label_counts = final_anxiety_dataset['Label'].value_counts()
print(label_counts.reindex(desired_labels))

# Define the colors for each bar
colors = ['skyblue', 'green', 'yellow', 'orange', 'gray']

# Reorder the label counts based on the desired arrangement
label_counts_ordered = label_counts.reindex(desired_labels)

# Plot the bar chart
plt.bar(label_counts_ordered.index, label_counts_ordered.values, color=colors)

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.title('Anxiety Dataset Distribution of Labels')

# Show the plot
plt.show()


# # Generating the training and test set for each Dataset

# # **Depression Dataset**

# In[24]:


#Seperate the data and labels
depression_labels = final_depression_dataset["Label"]
depression_X = final_depression_dataset.drop(columns=["Label"])


# In[25]:


depression_labels


# In[26]:


depression_X


# In[27]:


#Encode the labels 
encoder = LabelEncoder()
encoded_depression_label = encoder.fit_transform(depression_labels)

# Define the desired label values
desired_label_values = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']
encoder.classes_ = desired_label_values


# In[28]:


dict(zip(encoder.classes_,range(len(encoder.classes_))))


# In[29]:


encoded_depression_label


# In[30]:


#Get the training and test set from the depression dataset
dp_X_Train, dp_X_Test, dp_Y_Train, dp_Y_Test = train_test_split(depression_X, encoded_depression_label, test_size=0.3, random_state= 30)


# In[31]:


dp_X_Train


# In[32]:


dp_Y_Train


# In[33]:


dp_X_Test


# In[34]:


dp_Y_Test


# In[35]:


# Calculate the count of each unique label
unique_labels, label_counts = np.unique(dp_Y_Test, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Test data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")


# In[36]:


# Calculate the count of each unique label
unique_labels, label_counts = np.unique(dp_Y_Train, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Training data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")


# # **Stress Dataset**

# In[37]:


#Seperate the data and labels
stress_labels = final_stress_dataset["Label"]
stress_X = final_stress_dataset.drop(columns=["Label"])


# In[38]:


stress_labels


# In[39]:


stress_X


# In[40]:


#Encode the labels 
encoder = LabelEncoder()
encoded_stress_label = encoder.fit_transform(stress_labels)

# Define the desired label values
desired_label_values = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']
encoder.classes_ = desired_label_values


# In[41]:


dict(zip(encoder.classes_,range(len(encoder.classes_))))


# In[42]:


encoded_stress_label


# In[43]:


#Get the training and test set from the stress dataset
st_X_Train, st_X_Test, st_Y_Train, st_Y_Test = train_test_split(stress_X, encoded_stress_label, test_size=0.3, random_state= 30)


# In[44]:


st_X_Train


# In[45]:


st_Y_Train


# In[46]:


st_X_Test


# In[47]:


st_Y_Test


# In[48]:


# Calculate the count of each unique label
unique_labels, label_counts = np.unique(st_Y_Test, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Test data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")


# In[49]:


# Calculate the count of each unique label
unique_labels, label_counts = np.unique(st_Y_Train, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Training data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")


# # **Anxiety Dataset**

# In[50]:


#Seperate the data and labels
anxiety_labels = final_anxiety_dataset["Label"]
anxiety_X = final_anxiety_dataset.drop(columns=["Label"])


# In[51]:


anxiety_labels


# In[52]:


anxiety_X


# In[53]:


#Encode the labels 
encoder = LabelEncoder()
encoded_anxiety_label = encoder.fit_transform(anxiety_labels)

# Define the desired label values
desired_label_values = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']
encoder.classes_ = desired_label_values


# In[54]:


dict(zip(encoder.classes_,range(len(encoder.classes_))))


# In[55]:


encoded_anxiety_label


# In[56]:


#Get the training and test set from the stress dataset
ax_X_Train, ax_X_Test, ax_Y_Train, ax_Y_Test = train_test_split(anxiety_X, encoded_anxiety_label, test_size=0.3, random_state= 30)


# In[57]:


ax_X_Train


# In[58]:


ax_Y_Train


# In[59]:


ax_X_Test


# In[60]:


ax_Y_Test


# In[61]:


# Calculate the count of each unique label
unique_labels, label_counts = np.unique(ax_Y_Test, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Test data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")


# In[62]:


# Calculate the count of each unique label
unique_labels, label_counts = np.unique(ax_Y_Train, return_counts=True)
# Define the desired label arrangement
desired_labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

print("Training data distribution")
# Print the label counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")


# # **Model Training**

# # KNN Model

# **Model training on Depression dataset**

# In[63]:


#Training the model on depression dataset
k_model = KNeighborsClassifier(n_neighbors=10)
k_model.fit(dp_X_Train, dp_Y_Train)


# In[64]:


dp_predictions = k_model.predict(dp_X_Test)


# In[65]:


dp_predictions


# In[66]:


#Confusion matrix
cm = confusion_matrix(dp_Y_Test, dp_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[67]:


#Classification report
print(classification_report(dp_Y_Test, dp_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[68]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(dp_Y_Test, dp_predictions)
precision = precision_score(dp_Y_Test, dp_predictions, average='macro')
recall = recall_score(dp_Y_Test, dp_predictions, average='macro')
f1 = f1_score(dp_Y_Test, dp_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# **Model training on Stress dataset**

# In[69]:


#Training the model on depression dataset
k_model = KNeighborsClassifier(n_neighbors=10)
k_model.fit(st_X_Train, st_Y_Train)


# In[70]:


st_predictions = k_model.predict(st_X_Test)


# In[71]:


st_predictions


# In[72]:


#Confusion matrix
cm = confusion_matrix(st_Y_Test, st_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[73]:


#Classification report
print(classification_report(st_Y_Test, st_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[74]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(st_Y_Test, st_predictions)
precision = precision_score(st_Y_Test, st_predictions, average='macro')
recall = recall_score(st_Y_Test, st_predictions, average='macro')
f1 = f1_score(st_Y_Test, st_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# **Model training on Anxiety dataset**

# In[75]:


#Training the model on depression dataset
k_model = KNeighborsClassifier(n_neighbors=10)
k_model.fit(ax_X_Train, ax_Y_Train)


# In[76]:


ax_predictions = k_model.predict(ax_X_Test)


# In[77]:


ax_predictions


# In[78]:


#Confusion matrix
cm = confusion_matrix(ax_Y_Test, ax_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[79]:


#Classification report
print(classification_report(ax_Y_Test, ax_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[80]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(ax_Y_Test, ax_predictions)
precision = precision_score(ax_Y_Test, ax_predictions, average='macro')
recall = recall_score(ax_Y_Test, ax_predictions, average='macro')
f1 = f1_score(ax_Y_Test, ax_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# # SVM Model

# **Model training on Depression dataset**

# In[81]:


#Training the model on depression dataset
svm_model = SVC(C = 10, kernel = 'rbf', gamma= 0.2, random_state=24)
svm_model.fit(dp_X_Train, dp_Y_Train)


# In[82]:


dp_predictions = svm_model.predict(dp_X_Test)


# In[83]:


dp_predictions


# In[84]:


#Confusion matrix
cm = confusion_matrix(dp_Y_Test, dp_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[85]:


#Classification report
print(classification_report(dp_Y_Test, dp_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[86]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(dp_Y_Test, dp_predictions)
precision = precision_score(dp_Y_Test, dp_predictions, average='macro')
recall = recall_score(dp_Y_Test, dp_predictions, average='macro')
f1 = f1_score(dp_Y_Test, dp_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# **Model training on Stress dataset**

# In[87]:


#Training the model on stress dataset
svm_model = SVC(C = 10, kernel = 'rbf', gamma= 0.2, random_state=24)
svm_model.fit(st_X_Train, st_Y_Train)


# In[88]:


st_predictions = svm_model.predict(st_X_Test)


# In[89]:


st_predictions


# In[90]:


#Confusion matrix
cm = confusion_matrix(st_Y_Test, st_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[91]:


#Classification report
print(classification_report(st_Y_Test, st_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[92]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(st_Y_Test, st_predictions)
precision = precision_score(st_Y_Test, st_predictions, average='macro')
recall = recall_score(st_Y_Test, st_predictions, average='macro')
f1 = f1_score(st_Y_Test, st_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# **Model training on Anxiety dataset**

# In[93]:


#Training the model on depression dataset
svm_model = SVC(C = 10, kernel = 'rbf', gamma= 0.2, random_state=24)
svm_model.fit(ax_X_Train, ax_Y_Train)


# In[94]:


ax_predictions = svm_model.predict(ax_X_Test)


# In[95]:


ax_predictions


# In[96]:


#Confusion matrix
cm = confusion_matrix(ax_Y_Test, ax_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[97]:


#Classification report
print(classification_report(ax_Y_Test, ax_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[98]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(ax_Y_Test, ax_predictions)
precision = precision_score(ax_Y_Test, ax_predictions, average='macro')
recall = recall_score(ax_Y_Test, ax_predictions, average='macro')
f1 = f1_score(ax_Y_Test, ax_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# # Random Forest Model

# **Model training on Depression dataset**

# In[99]:


#Training the model on depression dataset
rf_model = RandomForestClassifier(random_state = 30)
rf_model.fit(dp_X_Train, dp_Y_Train)


# In[100]:


dp_predictions = rf_model.predict(dp_X_Test)


# In[101]:


dp_predictions


# In[102]:


#Confusion matrix
cm = confusion_matrix(dp_Y_Test, dp_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[103]:


#Classification report
print(classification_report(dp_Y_Test, dp_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[104]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(dp_Y_Test, dp_predictions)
precision = precision_score(dp_Y_Test, dp_predictions, average='macro')
recall = recall_score(dp_Y_Test, dp_predictions, average='macro')
f1 = f1_score(dp_Y_Test, dp_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# **Model training on Stress dataset**

# In[105]:


#Training the model on stress dataset
rf_model = RandomForestClassifier(random_state=24)
rf_model.fit(st_X_Train, st_Y_Train)


# In[106]:


st_predictions = rf_model.predict(st_X_Test)


# In[107]:


st_predictions


# In[108]:


#Confusion matrix
cm = confusion_matrix(st_Y_Test, st_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[109]:


#Classification report
print(classification_report(st_Y_Test, st_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[110]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(st_Y_Test, st_predictions)
precision = precision_score(st_Y_Test, st_predictions, average='macro')
recall = recall_score(st_Y_Test, st_predictions, average='macro')
f1 = f1_score(st_Y_Test, st_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# **Model training on Anxiety dataset**

# In[111]:


#Training the model on depression dataset
rf_model = RandomForestClassifier(random_state=24)
rf_model.fit(ax_X_Train, ax_Y_Train)


# In[112]:


ax_predictions = rf_model.predict(ax_X_Test)


# In[113]:


ax_predictions


# In[114]:


#Confusion matrix
cm = confusion_matrix(ax_Y_Test, ax_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[115]:


#Classification report
print(classification_report(ax_Y_Test, ax_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[116]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(ax_Y_Test, ax_predictions)
precision = precision_score(ax_Y_Test, ax_predictions, average='macro')
recall = recall_score(ax_Y_Test, ax_predictions, average='macro')
f1 = f1_score(ax_Y_Test, ax_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# # Decision Tree Model

# **Model training on Depression dataset**

# In[117]:


#Training the model on depression dataset
dt_model = DecisionTreeClassifier(random_state = 30, max_depth=20)
dt_model.fit(dp_X_Train, dp_Y_Train)


# In[118]:


dp_predictions = dt_model.predict(dp_X_Test)


# In[119]:


dp_predictions


# In[120]:


#Confusion matrix
cm = confusion_matrix(dp_Y_Test, dp_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[121]:


#Classification report
print(classification_report(dp_Y_Test, dp_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[122]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(dp_Y_Test, dp_predictions)
precision = precision_score(dp_Y_Test, dp_predictions, average='macro')
recall = recall_score(dp_Y_Test, dp_predictions, average='macro')
f1 = f1_score(dp_Y_Test, dp_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# **Model training on Stress dataset**

# In[123]:


#Training the model on stress dataset
dt_model = DecisionTreeClassifier(random_state = 30, max_depth=20)
dt_model.fit(st_X_Train, st_Y_Train)


# In[124]:


st_predictions = dt_model.predict(st_X_Test)


# In[125]:


st_predictions


# In[126]:


#Confusion matrix
cm = confusion_matrix(st_Y_Test, st_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[127]:


#Classification report
print(classification_report(st_Y_Test, st_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[128]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(st_Y_Test, st_predictions)
precision = precision_score(st_Y_Test, st_predictions, average='macro')
recall = recall_score(st_Y_Test, st_predictions, average='macro')
f1 = f1_score(st_Y_Test, st_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# **Model training on Anxiety dataset**

# In[129]:


#Training the model on depression dataset
dt_model = DecisionTreeClassifier(random_state = 30, max_depth=20)
dt_model.fit(ax_X_Train, ax_Y_Train)


# In[130]:


ax_predictions = dt_model.predict(ax_X_Test)


# In[131]:


ax_predictions


# In[132]:


#Confusion matrix
cm = confusion_matrix(ax_Y_Test, ax_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[133]:


#Classification report
print(classification_report(ax_Y_Test, ax_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[134]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(ax_Y_Test, ax_predictions)
precision = precision_score(ax_Y_Test, ax_predictions, average='macro')
recall = recall_score(ax_Y_Test, ax_predictions, average='macro')
f1 = f1_score(ax_Y_Test, ax_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# # Naive Bayes Model

# **Model training on Depression dataset**

# In[135]:


#Training the model on depression dataset
nb_model = GaussianNB()
nb_model.fit(dp_X_Train, dp_Y_Train)


# In[136]:


dp_predictions = nb_model.predict(dp_X_Test)


# In[137]:


dp_predictions


# In[138]:


#Confusion matrix
cm = confusion_matrix(dp_Y_Test, dp_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[139]:


#Classification report
print(classification_report(dp_Y_Test, dp_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[140]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(dp_Y_Test, dp_predictions)
precision = precision_score(dp_Y_Test, dp_predictions, average='macro')
recall = recall_score(dp_Y_Test, dp_predictions, average='macro')
f1 = f1_score(dp_Y_Test, dp_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# **Model training on Stress dataset**

# In[141]:


#Training the model on stress dataset
nb_model = GaussianNB()
nb_model.fit(st_X_Train, st_Y_Train)


# In[142]:


st_predictions = nb_model.predict(st_X_Test)


# In[143]:


st_predictions


# In[144]:


#Confusion matrix
cm = confusion_matrix(st_Y_Test, st_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[145]:


#Classification report
print(classification_report(st_Y_Test, st_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[146]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(st_Y_Test, st_predictions)
precision = precision_score(st_Y_Test, st_predictions, average='macro')
recall = recall_score(st_Y_Test, st_predictions, average='macro')
f1 = f1_score(st_Y_Test, st_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# **Model training on Anxiety dataset**

# In[147]:


#Training the model on depression dataset
nb_model = GaussianNB()
nb_model.fit(ax_X_Train, ax_Y_Train)


# In[148]:


ax_predictions = nb_model.predict(ax_X_Test)


# In[149]:


ax_predictions


# In[150]:


#Confusion matrix
cm = confusion_matrix(ax_Y_Test, ax_predictions)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'Normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('KNN Depression.png')
plt.show()


# In[151]:


#Classification report
print(classification_report(ax_Y_Test, ax_predictions, target_names = ['Extremely Severe(0)', 'Severe(1)', 'Moderate(2)', 'Mild(3)', 'Normal(4)']))


# In[152]:


# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(ax_Y_Test, ax_predictions)
precision = precision_score(ax_Y_Test, ax_predictions, average='macro')
recall = recall_score(ax_Y_Test, ax_predictions, average='macro')
f1 = f1_score(ax_Y_Test, ax_predictions, average='macro')


# Print the evaluation metrics
print("Accuracy: %.f" %(accuracy*100))
print("Precision: %.f" %(precision*100))
print("Recall: %.f" %(recall*100))
print("F1-score: %.f" %(f1*100))


# # **Save SVM Model**

# In[153]:


# Save the model
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)


# In[154]:


# Load the saved model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

