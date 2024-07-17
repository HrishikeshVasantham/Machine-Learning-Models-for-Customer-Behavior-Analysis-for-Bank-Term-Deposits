# Importing the libraries
import numpy as np
import pandas as pd
import klib
from numpy import math
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#file path reading or csv upload to coolab from drive (seperated = ;)
file_path='bank-full.csv'     
data=pd.read_csv(file_path)
data.head()
print(data.columns)
#Checking shape of data.
print("Rows - ", data.shape[0])
print("Columns - ", data.shape[1])
data.isnull().sum()
data.duplicated().sum()
data.describe(include='all')
#checking info of data
data.info()
klib.dist_plot(data)

import seaborn as sns
import matplotlib.pyplot as plt

categorcial_variables = ['job', 'marital', 'education', 'default', 'loan', 'contact', 'month', 'day', 'poutcome','y']
for col in categorcial_variables:
    plt.figure(figsize=(10,4))    
    sns.barplot(x=data[col].value_counts().index, y=data[col].value_counts().values)
    plt.title(col)
    plt.tight_layout()
    plt.show()
    

plt.figure(figsize=(7,5))
sns.boxplot(y='age',x="y", data= data)
plt.show()

plt.figure(figsize=(7,7))
plt.pie(data.job.value_counts(),labels=data.job.value_counts().index,shadow = True,autopct='%1.1f%%')
plt.title('Jobs')
plt.show()

data.job.value_counts()

#Lets explore the count of accept and reject term deposit on the basis of different type of job 
fig, ax = plt.subplots(figsize=(15,5))
#sns.countplot(data.job, ax=ax, palette='pastel')
#sns.countplot(data.job, hue=data.y,ax=ax, palette='pastel')

sns.countplot(x=data.job, hue=data.y, palette='pastel')


#Lets see pie plot of Martial status 
plt.figure(figsize=(6,6))
plt.pie(data.marital.value_counts(),labels=data.marital.value_counts().index,autopct='%1.2f%%',shadow = True)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Marital Status')
plt.show()

data.marital.value_counts()


#Lets explore the count of accept and reject term deposit on the basis of different type of marital status
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=data.marital, hue=data.y, palette='YlGnBu')
plt.xlabel('marital status',fontsize=10)
plt.ylabel('percentage',fontsize=10)
plt.title('Count of yes and no for different category of marital status')

# deposit term on the basis of different type of educated person
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x=data.education, palette='pastel')
sns.countplot(x=data.education, hue=data.y, palette='bright')
plt.xlabel('Education ')
plt.title('Count of yes and no for different education level')

#Lets see pie plot of default 
plt.figure(figsize=(6,6))
colors = ['#ff9990','#81b3ff']
plt.pie(data.default.value_counts(),labels=data.default.value_counts().index,autopct='%1.2f%%',colors=colors)

plt.title('default')
plt.show()

#Lets explore the count of accept and reject term deposit on the basis of different type of educated person
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=data.default, hue=data.y, palette='pastel')
plt.xlabel('Default')
plt.title('Count of yes and no for default category')

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=data.housing, palette='pastel')
sns.countplot(x=data.housing, hue=data.y, palette='hot')
plt.xlabel('Housing')
plt.title('Count of yes and no for house loan category')

#Lets see pie plot for loan
plt.figure(figsize=(6,6))
plt.pie(data.loan.value_counts(),labels=data.loan.value_counts().index,autopct='%1.2f%%',shadow= True, colors='bright')
plt.title('loan')
plt.show()


#Lets explore the count of accept and reject term deposit on the basis of different type of educated person
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=data.loan,  palette='pastel')
sns.countplot(x=data.loan, hue=data.y, palette='bright')
plt.xlabel('Personal Loan')
plt.title('Count of yes and no for personal loan category')

#Lets explore the count of accept and reject term deposit on contact type
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=data.contact, palette='pastel')
sns.countplot(x=data.contact, hue=data.y, palette='bright')
plt.xlabel('Contact Type')
plt.title('Count of yes and no for contact category')

data['month'].value_counts()

#Lets explore the count of accept and reject term deposit every month
fig, ax = plt.subplots(figsize=(16, 5))
sns.countplot(x=data.month,palette='coolwarm')
sns.countplot(x=data.month, hue=data.y, palette='bright')
plt.xlabel('Month')
plt.title('Count of yes and no for every month')

sns.boxplot(y='duration',x="y", data= data)

#Countplot of various number of contact were perfomed to how many people
plt.figure(figsize=(22,6))
sns.countplot(x=data['campaign'])
plt.show()

sns.boxplot(y='pdays',x="y", data= data)



#Ploting countplot of previous feature
plt.figure(figsize=(15,6))
sns.countplot(x=data['previous'] ,palette="Accent")
plt.show()

plt.figure(figsize=(6,6))
plt.pie(data.poutcome.value_counts(),labels=data.poutcome.value_counts().index,autopct='%1.2f%%',shadow = True)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Poutcome')
plt.show()

#Lets explore the count of accept and reject term deposit on the basis of poutcome
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x=data.poutcome,palette='pastel')
sns.countplot(x=data.poutcome, hue=data.y, palette='bright')

# pie plot 
plt.figure(figsize=(6,6))
colors = ['#ff9999','#66b3ff']
plt.pie(data.y.value_counts(),labels=data.y.value_counts().index,autopct='%1.2f%%',shadow = True,colors = colors)
plt.title('Deposite')
plt.show()

data['y'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x=data.y, palette='pastel')

# Convert target variable into numeric
data.y = data.y.map({'no':0, 'yes':1}).astype('uint8')

# Plotting correlation matrix
plt.subplots(figsize=(15,8))
sns.heatmap(data.corr().abs(), annot=True,cmap="YlGnBu")

#dropping unnescessary column
data.drop(['marital'],axis=1, inplace=True)
data.drop(['contact'],axis=1, inplace=True)
data.head()

data[['default','housing','loan']]=data[['default','housing','loan']].replace(["yes","no"],["1","0"])
data['month']=data['month'].replace(["jan","feb","mar","apr","may","jun","jul", "aug","sep","oct","nov","dec"],["1","2","3","4","5","6","7","8","9","10","11","12"])
data['job']=data['job'].replace(['unknown'],['other'])

data.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['job']=le.fit_transform(data['job'])
data['education']=le.fit_transform(data['education'])
data['poutcome']=le.fit_transform(data['poutcome'])

data.head()

col=data[['age','balance','day','campaign','duration','pdays','previous']]
for i in col:
  n=1
  plt.figure(figsize=(20,20))
  plt.subplot(4,3,1)
  sns.boxplot(data[i])
  plt.title(i)
  plt.show()
  n=n+1

from scipy import stats
z = np.abs(stats.zscore(data[['age','balance','duration','campaign','pdays','previous']]))
print(z)
data=data[(z<3).all(axis=1)]
data.shape

#checking outliers removed ot not
for i in col:
  plt.figure(figsize=(20,10))
  plt.subplot(3,3,1)
  sns.boxplot(x=data[i])
  plt.title(i)
plt.show()

#contain all  independent variabl
x=data.drop(['y'],axis=1)

#dependent variable
y=data['y'] 

sns.countplot(x='y',data=data)
print(data['y'].value_counts())

# Using random over sampling
from imblearn.over_sampling import RandomOverSampler
os =  RandomOverSampler()
x_new,y_new=os.fit_resample(x,y)

sns.countplot(x=y_new)

from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_new)))

#loading required libraries 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score
from sklearn.model_selection import cross_val_score,ShuffleSplit,cross_val_predict
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score, roc_curve, log_loss

#dividing the dataset into training and testing
x_train,x_test,y_train,y_test=train_test_split(x_new,y_new,test_size=.3,random_state=0)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# Lets define a function for Evaluation metrics 
def print_metrics(actual,prediction,model=''):
  print(f'{model} Test accuracy Score', accuracy_score(actual,prediction))
  print(classification_report(actual,prediction))
  
  return confusion_matrix(actual,prediction)

#1.Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 4000)
lr.fit(x_train,y_train)
cv_score = cross_val_score(lr,x_train,y_train,cv=5)

y_pred_lr=lr.predict(x_test)

print('Cross_validation score',cv_score)
print_metrics(y_test,y_pred_lr,'LogisticRegression')

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

rf_reg = RandomForestClassifier(max_depth = 8, n_estimators = 200)
rf_reg.fit(x_train,y_train)
cv_score = cross_val_score(rf_reg,x_train,y_train,cv=5)

y_pred_rf=rf_reg.predict(x_test)

print('Cross_validation score',cv_score)
print_metrics(y_test,y_pred_rf,'RandomForest')

#3.KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
cv_score = cross_val_score(knn,x_train,y_train,cv=5)

y_pred_knn=knn.predict(x_test)

print('Cross_validation score',cv_score)
print_metrics(y_test,y_pred_knn,'KNN')

from sklearn.svm import SVC

svc = SVC(random_state=0)

svc.fit(x_train,y_train)
cv_score = cross_val_score(svc,x_train,y_train,cv=5)

y_pred_svc = svc.predict(x_test)

print('Cross_validation score',cv_score)
print_metrics(y_test,y_pred_svc,'SVC')

#loading libraries
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(learning_rate=0.1, max_depth= 25, n_estimators= 50)

lgbm.fit(x_train,y_train)
cv_score = cross_val_score(lgbm,x_train,y_train,cv=5)

y_pred_lgbm = lgbm.predict(x_test)

print('Cross_validation score',cv_score)
print_metrics(y_test,y_pred_lgbm,'LGBM')

y_probs_train = lgbm.predict_proba(x_train)
y_probs_test = lgbm.predict_proba(x_test)
y_predicted_train = lgbm.predict(x_train)
y_predicted_test = lgbm.predict(x_test)

# keep probabilities for the positive outcome only
y_probs_train = y_probs_train[:, 1]
y_probs_test = y_probs_test[:, 1]

# calculate AUC and Accuracy
plt.figure(figsize=(9,7))
train_auc = roc_auc_score(y_train, y_probs_train)
test_auc = roc_auc_score(y_test, y_probs_test)
train_acc = accuracy_score(y_train, y_predicted_train)
test_acc = accuracy_score(y_test, y_predicted_test)
f1_s=f1_score(y_test,y_predicted_test)
p_score=precision_score(y_test,y_predicted_test)

print('*'*50)
print('Train AUC: %.3f' % train_auc)
print('Test AUC: %.3f' % test_auc)
print('*'*50)
print('Train Accuracy: %.3f' % train_acc)
print('Test Accuracy: %.3f' % test_acc)

#score['KNN (Over sampling)'] = [test_auc, test_acc,f1_s,p_score]

# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_probs_train)
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_probs_test)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr, marker='.', label='Train AUC')
plt.plot(test_fpr, test_tpr, marker='.', label='Test AUC')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LGBM-ROC Curve")
plt.show()

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,ShuffleSplit
lgbm_wt = LGBMClassifier(learning_rate=0.1, max_depth= 5, n_estimators= 150)


parameters=[{'learning_rate': [0.05, 0.1, 0.2 ,0.3 ,0.4], 'max_depth': range(3,8), 'n_estimators': range(30, 150, 10)}] 

lgbm_grid=RandomizedSearchCV(lgbm_wt,parameters,scoring='precision',cv=5,verbose=2)

lgbm_grid.fit(x_train,y_train)

y_pred_lgbm=lgbm_grid.predict(x_test)

print('Cross_validation score',cv_score)
print_metrics(y_test,y_pred_lgbm,'LGBM_Hypertunning')

from prettytable import PrettyTable

print('**** Comparison of  models for Class 1(Yes)  (oversampled train data)  ****')
table = PrettyTable(['Model', 'Test Accuracy', 'Precision', 'Recall', 'F1_score'])
table.add_row(['Logistic regression', 0.7921, 0.81, 0.77, 0.79])
table.add_row(['Random Forest', 0.8597, 0.84, 0.88, 0.86])
table.add_row(['KNN', 0.9264, 0.87, 0.98, 0.92])
table.add_row(['SVC', 0.8493, 0.83, 0.87, 0.85])
table.add_row(['LGBM', 0.8723, 0.86, 0.90, 0.88])
table.add_row(['LGBM (Hyperparameter Tunned)', 0.9293, 0.90, 0.95, 0.92])


print(table)

