import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

import xgboost as xgb

dataset = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/dataset.npy")
outcome = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/outcome.npy")


model1 = BaggingClassifier(RandomForestClassifier(n_estimators=225,max_features="auto",
                                min_weight_fraction_leaf= 0.0, min_samples_split=0.10000000000000001, min_samples_leaf=4, max_depth=467))
model1.fit(dataset,outcome)
score1 = model1.score(dataset,outcome)
print ("Accuracy from Random forest::",score1)

model2 = BaggingClassifier(xgb.XGBClassifier(colsample_bytree= 0.39258925829065067,
 learning_rate= 0.49227635408236803, max_depth= 65, n_estimators=168, nthread=8,
objective= 'multi:softprob', reg_alpha= 17.7827941, subsample= 0.93124726513080813))

model2.fit(dataset,outcome)
score2 = model2.score(dataset,outcome)
print ("Accuracy from XGB::",score2)



