
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import pandas as pd
from catboost import CatBoostClassifier

dataset = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/dataset1.npy")
outcome = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/outcome1.npy")

x_train, x_test, y_train, y_test = train_test_split(dataset, outcome, test_size=0.25, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
# %% GBM
model= BaggingClassifier(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))
model.fit(x_train, y_train)
print(model.score(x_test,y_test))

# %% catboost

model = CatBoostClassifier(iterations=600, depth=11, learning_rate=0.1,loss_function='MultiClass')
model.fit(pd.DataFrame(x_train),pd.DataFrame(y_train),
          eval_set=(pd.DataFrame(x_val),pd.DataFrame(y_val)))
print(model.score(x_test,y_test))