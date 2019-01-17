

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from scripts import prgm1
import pandas as pd
import xgboost as xgb
# Here we perform stacking with k fold evaluation
# %%

dataset = np.loadtxt('/home/arjun/PycharmProjects/ML_proj/dataset/train.csv', dtype=str, delimiter=",")
dataset,outcome = prgm1.preprocessing(dataset)
np.save("/home/arjun/PycharmProjects/ML_proj/scripts/dataset.npy", dataset)
np.save("/home/arjun/PycharmProjects/ML_proj/scripts/outcome.npy", outcome)
# %%
dataset = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/dataset.npy")
outcome = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/outcome.npy")
# %%

partition = np.int(np.round(0.8*dataset.shape[0]))
x_train = dataset[0:partition,:]
x_test = dataset[partition:,:]
y_test = np.array(outcome[partition:])
y_train = np.array(outcome[0:partition])

# %%
def Stacking(model,train,y,test,n_fold):
   folds=StratifiedKFold(n_splits=n_fold,random_state=1)
   test_pred=np.empty((0,1),float)
   train_pred=np.empty((0,1),float)
   train_label = np.empty((0, 1), float)
   for train_indices,val_indices in folds.split(train,y):
        # x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        x_train, x_val = train[train_indices], train[val_indices]
        y_train,y_val=y[train_indices],y[val_indices]

        model.fit(X=x_train,y=y_train)
        train_pred=np.append(train_pred,model.predict(x_val))
        train_label = np.append(train_label,y_val)
   model.fit(X=train,y=y)
   test_pred=np.append(test_pred,model.predict(test))
   return test_pred.reshape(-1,1),train_pred,train_label

# Model 1
# model1 = DecisionTreeClassifier(random_state=1)

model1 = RandomForestClassifier()
test_pred1 ,train_pred1,train_label1=Stacking(model=model1,n_fold=5, train=x_train,test=x_test,y=y_train)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)
print("Scores from Model1::",model1.score(x_test,y_test))

# %%
# Model 2
model2 = svm.SVC(kernel="rbf",gamma="auto")

test_pred2 ,train_pred2,train_label2=Stacking(model=model2,n_fold=5,train=x_train,test=x_test,y=y_train)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)
print("Scores from Model2::",model2.score(x_test,y_test))


# %% Model 3
model3 =  DecisionTreeClassifier()
test_pred3 ,train_pred3,train_label3=Stacking(model=model3,n_fold=5,train=x_train,test=x_test,y=y_train)

train_pred3=pd.DataFrame(train_pred3)
test_pred3=pd.DataFrame(test_pred3)
print("Scores from Model3::",model3.score(x_test,y_test))

# %% model 4
model4 = KNeighborsClassifier()
test_pred4 ,train_pred4,train_label4=Stacking(model=model4,n_fold=5,train=x_train,test=x_test,y=y_train)

train_pred4=pd.DataFrame(train_pred4)
test_pred4=pd.DataFrame(test_pred4)
print("Scores from Model4::",model4.score(x_test,y_test))
# %%
# Layer2
df = pd.concat([train_pred1, train_pred2, train_pred3, train_pred4], axis=1)
df_test = pd.concat([test_pred1, test_pred2, test_pred3, test_pred4], axis=1)
train_label2 = pd.DataFrame(train_label2)
y_test = pd.DataFrame(y_test)
# df_label = pd.concat([train_label1,train_label2,train_label3,train_label4], axis=1)
#
# model = GradientBoostingClassifier()
# model.fit(df,train_label1)
# # predictions = model.predict(df_test)
# # predictions = predictions.reshape(-1,1)
# print(model.score(df_test,y_test))

gbm = xgb.XGBClassifier(
 learning_rate = 0.38146109888951574,
 n_estimators= 90,
 max_depth= 13,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,
 subsample= 0.7335344541230151,
 colsample_bytree=0.23691477683290396,
 objective= 'multi:softprob',
 nthread= 8,
 reg_alpha=17.7827941,
 scale_pos_weight=1)
gbm.fit(df.loc[:,~df.columns.duplicated()], train_label2.loc[:,~train_label2.columns.duplicated()])
print(gbm.score(df_test.loc[:,~df_test.columns.duplicated()],y_test.loc[:,~y_test.columns.duplicated()]))


