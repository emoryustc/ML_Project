import numpy as np
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
from catboost import CatBoostClassifier
from sklearn import discriminant_analysis

dataset = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/dataset1.npy")
outcome = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/outcome1.npy")
x_train, x_test, y_train, y_test = train_test_split(dataset, outcome, test_size=0.33, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)


model1 = xgb.XGBClassifier(colsample_bytree= 0.38080923653441978,
                           learning_rate= 0.13766708758980395,
                           max_depth=13,
                           n_estimators= 72,
                           nthread=8,
                           objective='multi:softprob',
                           reg_alpha=1.77827941,
                           subsample= 0.98148897078432862)
model1.fit(x_train, y_train)
val_pred1=model1.predict(x_val)
test_pred1=model1.predict(x_test)
val_pred1=pd.DataFrame(val_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = CatBoostClassifier(iterations=500, depth=10,
                            learning_rate=0.1,
                            loss_function='MultiClass')
model2.fit(x_train,y_train)
val_pred2=model2.predict(x_val)
test_pred2=model2.predict(x_test)
val_pred2=pd.DataFrame(val_pred2)
test_pred2=pd.DataFrame(test_pred2)

model3 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model3.fit(x_train,y_train)
val_pred3=model3.predict(x_val)
test_pred3=model3.predict(x_test)
val_pred3=pd.DataFrame(val_pred3)
test_pred3=pd.DataFrame(test_pred3)

model4 = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr',n_components=0)
model4.fit(x_train,y_train)
val_pred4=model4.predict(x_val)
test_pred4=model4.predict(x_test)
val_pred4=pd.DataFrame(val_pred4)
test_pred4=pd.DataFrame(test_pred4)

x_val = pd.DataFrame(x_val)
x_test= pd.DataFrame(x_test)
df_val=pd.concat([x_val, val_pred1,val_pred2],axis=1)
df_test=pd.concat([x_test, test_pred1,test_pred2],axis=1)

model = LogisticRegression()
model.fit(df_val,y_val)
print (model.score(df_test,y_test))