import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd

dataset = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/dataset1.npy")
outcome = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/outcome1.npy")
x_train, x_test, y_train, y_test = train_test_split(dataset, outcome, test_size=0.33, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)


model1 = xgb.XGBClassifier(colsample_bytree= 0.39258925829065067,
 learning_rate= 0.49227635408236803, max_depth= 65, n_estimators=168, nthread=8,
objective= 'multi:softprob', reg_alpha= 17.7827941, subsample= 0.93124726513080813)
model1.fit(x_train, y_train)
val_pred1=model1.predict(x_val)
test_pred1=model1.predict(x_test)
val_pred1=pd.DataFrame(val_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = RandomForestClassifier(n_estimators=225,max_features="auto",
                                min_weight_fraction_leaf= 0.0, min_samples_split=0.10000000000000001, min_samples_leaf=4, max_depth=467)
model2.fit(x_train,y_train)
val_pred2=model2.predict(x_val)
test_pred2=model2.predict(x_test)
val_pred2=pd.DataFrame(val_pred2)
test_pred2=pd.DataFrame(test_pred2)

x_val = pd.DataFrame(x_val)
x_test= pd.DataFrame(x_test)
df_val=pd.concat([x_val, val_pred1,val_pred2],axis=1)
df_test=pd.concat([x_test, test_pred1,test_pred2],axis=1)

model = LogisticRegression()
model.fit(df_val,y_val)
print (model.score(df_test,y_test))