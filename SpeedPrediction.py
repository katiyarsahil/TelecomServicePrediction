
#Import Relevant packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
get_ipython().magic('matplotlib inline')
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
#Read inout data
sp=pd.read_csv('FIA_Speed_Predict_Ready_Input_1129.csv',low_memory=False)
#Replace target variables 10000 and 5000 with 2000
sp['HH_Bucketed_Speed'].replace(10000,2000,inplace=True)
sp['HH_Bucketed_Speed'].replace(5000,2000 ,inplace=True)
# filtering data to only include year 2017
sp=sp.loc[(sp.HH_Year == 2017)]
# Changing data types for categorical variables
sp['NAICS2']=sp['NAICS2'].astype('category')
sp['NAICS4']=sp['NAICS4'].astype('category')
sp['NAICS_CD']=sp['NAICS_CD'].astype('category')
sp['Restricted_Vertical']=sp['Restricted_Vertical'].astype('category')
sp['LCTN_TYP_VAL']=sp['LCTN_TYP_VAL'].astype('category')
sp['srvc_five_dgt_zip']=sp['srvc_five_dgt_zip'].astype('category')
sp['data_srvc_rnge_6_flg']=sp['data_srvc_rnge_6_flg'].astype('category')
sp['max_data_srvc_rnge_cls_val']=sp['max_data_srvc_rnge_cls_val'].astype('category')
sp['CITY_NM']=sp['CITY_NM'].astype('category')
sp['ST_CD']=sp['ST_CD'].astype('category')
sp['ONE_TWC_CMRCL_IND']=sp['ONE_TWC_CMRCL_IND'].astype('category')
sp['DMA_Code']=sp['DMA_Code'].astype('category')
sp['NO_MKT_DATA_FLG']=sp['NO_MKT_DATA_FLG'].astype('category')
sp['Employee_Bucket_Sales']=sp['Employee_Bucket_Sales'].astype('category')
sp['SEV_Employee_Bucket']=sp['SEV_Employee_Bucket'].astype('category')
sp['DUNS_Employee_Bucket']=sp['DUNS_Employee_Bucket'].astype('category')
sp['TWC_FOOTPRINT_FLG']=sp['TWC_FOOTPRINT_FLG'].astype('category')
sp['Fiber_Splice_Bucket']=sp['Fiber_Splice_Bucket'].astype('category')
sp['New_Region']=sp['New_Region'].astype('category')
sp['IN_DUNS']=sp['IN_DUNS'].astype('category')
sp['IN_INFO_USA']=sp['IN_INFO_USA'].astype('category')
sp['New_Division']=sp['New_Division'].astype('category')
sp['Location_Type']=sp['Location_Type'].astype('category')
sp['HH_HQ']=sp['HH_HQ'].astype('category')
sp['HH_Year']=sp['HH_Year'].astype('category')
sp['HH_Bucketed_Speed']=sp['HH_Bucketed_Speed'].astype('category')
# Checking for correlated variables
sp.corr()
sns.heatmap(sp.corr(),cmap='viridis')
# Check and plot Distribution of the target variables
count=sp.HH_Bucketed_Speed.value_counts()
count.columns=["Target", "Count"]
sns.countplot(x='HH_Bucketed_Speed',data=sp,order = sp['HH_Bucketed_Speed'].value_counts().index)
#Separating independent and target variables 
X=sp.iloc[:,0:33]
X.columns
y=sp['HH_Bucketed_Speed']
# Splitting training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=101)
#Checking for data imbalance
from collections import Counter
print(sorted(Counter(y).items()))
# Importing Smote packages
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
# Initiating SMOTE and fitting on the training data set. checking new traget variable distribution
smote=SMOTE('minority')
X_train_sm1,y_train_sm1=smote.fit_sample(X_train,y_train)
print(sorted(Counter(y_train_sm1).items()))
## Initiating SMOTEENN and fitting on the training data set. checking new traget variable distribution
smote_enn = SMOTEENN(random_state=101)
X_train_sme1, y_train_sme1 = smote_enn.fit_sample(X_train, y_train)
print(sorted(Counter(y_train_sme1).items()))
## Initiating SMOTE TOMEK and fitting on the training data set. checking new traget variable distribution
smote_tomek = SMOTETomek(random_state=101)
X_train_smt1, y_train_smt1= smote_tomek.fit_sample(X_train, y_train)
print(sorted(Counter(y_train_smt1).items()))
# IMporting Decision Tree classifier and metrics packages
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
# Initiating and fitting Decision tree
dt=DecisionTreeClassifier(criterion = "gini", splitter = 'random', min_samples_leaf = 15, max_depth=32,random_state=101)
dt.fit(X_train_sm1,y_train_sm1)
#Predicting on test set
predict=dt.predict(X_test)
# checking performacne of the decision tree classifier
print(confusion_matrix(y_test,predict))
print(classification_report(y_test,predict))
# Importing graphviz
from sklearn.tree import export_graphviz
import graphviz
features=X.columns
#Plotting and exporting decision tree
dot_data = export_graphviz(dt, out_file=None,feature_names=features) 
graph = graphviz.Source(dot_data) 
graph.render("decision tree",view=True) 
# Importing,intitiating and fitting Random forest 
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=1000,min_samples_split=30,max_features='auto',random_state=101)
rf.fit(X_train_sm1,y_train_sm1)
# predicitng on test data
rf_predict=rf.predict(X_test)
# checking performacne of the Random forest classifier
print(confusion_matrix(y_test,rf_predict))
print(classification_report(y_test,rf_predict))
#getting feature importance
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Plotting featre importance
plt.figure(figsize=(10,5))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), features,rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()    
# Importing,intitiating and fitting Extra trees classifier
from sklearn.tree import ExtraTreeClassifier
extree = ExtraTreeClassifier(max_features=11,min_samples_split=21,
                             random_state=101,max_depth =28)
extree.fit(X_train_sm1,y_train_sm1)
extree_predict=extree.predict(X_test)
#checking performacne of the extra trees classifier
print(confusion_matrix(y_test,extree_predict))
print(classification_report(y_test,extree_predict))
#Importing test data
test=pd.read_csv('FIA_predictions.csv')
# getting columns same as training data 
test=test.iloc[:,0:33]
#converting data type for categorical variables
test['NAICS2']=test['NAICS2'].astype('category')
test['NAICS4']=test['NAICS4'].astype('category')
test['NAICS_CD']=test['NAICS_CD'].astype('category')
test['Restricted_Vertical']=test['Restricted_Vertical'].astype('category')
test['LCTN_TYP_VAL']=test['LCTN_TYP_VAL'].astype('category')
test['srvc_five_dgt_zip']=test['srvc_five_dgt_zip'].astype('category')
test['data_srvc_rnge_6_flg']=test['data_srvc_rnge_6_flg'].astype('category')
test['max_data_srvc_rnge_cls_val']=test['max_data_srvc_rnge_cls_val'].astype('category')
test['CITY_NM']=test['CITY_NM'].astype('category')
test['ST_CD']=test['ST_CD'].astype('category')
test['ONE_TWC_CMRCL_IND']=test['ONE_TWC_CMRCL_IND'].astype('category')
test['DMA_Code']=test['DMA_Code'].astype('category')
test['NO_MKT_DATA_FLG']=test['NO_MKT_DATA_FLG'].astype('category')
test['Employee_Bucket_Sales']=test['Employee_Bucket_Sales'].astype('category')
test['SEV_Employee_Bucket']=test['SEV_Employee_Bucket'].astype('category')
test['DUNS_Employee_Bucket']=test['DUNS_Employee_Bucket'].astype('category')
test['TWC_FOOTPRINT_FLG']=test['TWC_FOOTPRINT_FLG'].astype('category')
test['Fiber_Splice_Bucket']=test['Fiber_Splice_Bucket'].astype('category')
test['New_Region']=test['New_Region'].astype('category')
test['IN_DUNS']=test['IN_DUNS'].astype('category')
test['IN_INFO_USA']=test['IN_INFO_USA'].astype('category')
test['New_Division']=test['New_Division'].astype('category')
test['Location_Type']=test['Location_Type'].astype('category')
test['HH_HQ']=test['HH_HQ'].astype('category')
test['HH_Year']=test['HH_Year'].astype('category')
# predicting speed for prediction data set
speed_pred=dt.predict(test)
print(sorted(Counter(speed_pred).items()))
ax=plt.axes()
sns.countplot(speed_pred,ax=ax)
# Adding predicted spee to predict dataset
output=test
test['Predicted Speed']=speed_pred
#Exporting results to csv
test.to_csv('New_Predicted_Speed_11302.csv')
