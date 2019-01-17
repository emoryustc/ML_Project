from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import random
from sklearn.model_selection import StratifiedKFold
import pandas as pd

x = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/dataset.npy")
y = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/outcome.npy")
model = RandomForestClassifier()
n_iter = 10
k_fold = 5
cv = StratifiedKFold(n_splits=k_fold,shuffle=True)
param = np.arange(500)
param_grid = {'n_estimators': param,
              'max_depth':param,
              'min_samples_split':np.arange(0.1,1.0),
              'min_samples_leaf':np.arange(1,5),
              'min_weight_fraction_leaf':np.arange(0.5),
              }
result = RandomizedSearchCV(model,param_grid,n_iter=n_iter
                            ,cv=cv,verbose=True).fit(x,y)

print (pd.DataFrame(result.cv_results_))
print (' ', result.best_score_)
print(' ', result.best_params_)
