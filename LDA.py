
#  The Linear Discriminant analysis 

# import package
import pandas as pd
#from pydataset import data
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics

# Reading Dataset - @ I got the Dataset from website(http://www-eio.upc.edu/~pau/cms/rdata/datasets.html)
path='F://Faisal//bioChemists.csv'
df=pd.read_csv(path)
df

# Dummy vaiable
dummy=pd.get_dummies(df['fem'])
df=pd.concat([df,dummy] ,axis=1)
dummy=pd.get_dummies(df['mar'])
df=pd.concat([df,dummy] , axis=1)
df.head()

#  Independent and Dependent Variables
X=df[['Men', 'kid5', 'ment', 'art']]
y=df['Married']

#  Train and Test Dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3 , random_state=0)

#  Model of  training - testing 
clf=LDA()
clf.fit(X_train,y_train)
clf.score(X_train,y_train)

#  Model of Prediction
y_pred=clf.predict(X_test)
y_pred

#  Evauation the performance of model
print(classification_report(y_test,y_pred))

#  Visualization
fpr, tpr, thresholds=metrics.roc_curve(y_test,y_pred)
roc_auc=metrics.auc(fpr,tpr)
roc_auc
plt.clf()
plt.plot(fpr,tpr,label="ROC curve (are=%0.2f)" % roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.legend(loc="lower right")

tpr
thresholds
roc_auc



