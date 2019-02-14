import numpy as np
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier


#Assumed you have training and test data set as train and test
dataset = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/dataset1.npy")
outcome = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/outcome1.npy")
X_train, X_test, y_train, y_test = train_test_split(dataset, outcome, test_size=0.33, random_state=42)
# Create PCA obeject
# pca= decomposition.PCA()
#default value of k =min(n_sample, n_features)catboost
# For Factor analysis
#fa= decomposition.FactorAnalysis()
# Reduced the dimension of training dataset using PCA
# train_reduced = pca.fit_transform(X_train)
# test_reduced = pca.transform(X_test)

model1 = BaggingClassifier(xgb.XGBClassifier(colsample_bytree= 0.39258925829065067,
 learning_rate= 0.49227635408236803, max_depth= 65, n_estimators=168, nthread=8,
objective= 'multi:softprob', reg_alpha= 17.7827941, subsample= 0.93124726513080813))

model1.fit(X_train,y_train)
score1 = model1.score(X_test,y_test)
print ("Accuracy from XGB::",score1)


model2 = model3 = BaggingClassifier(CatBoostClassifier(iterations=500, depth=10,
                           learning_rate=0.1,
                           loss_function='MultiClass'))
model2.fit(X_train,y_train)
score2 = model2.score(X_test,y_test)
print ("Accuracy from catboost::",score2)

model4 = BaggingClassifier(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))


model4.fit(X_train,y_train)
score4 = model4.score(X_test,y_test)
print ("Accuracy from GBM::",score4)

model6 = BaggingClassifier(LinearDiscriminantAnalysis(solver='lsqr',n_components=0))

model6.fit(X_train,y_train)
score6 = model6.score(X_test,y_test)
print ("Accuracy from LDA::",score6)

model4 = BaggingClassifier(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))

model4.fit(X_train,y_train)
score4 = model4.score(X_test,y_test)
print ("Accuracy from catboost::",score4)

model5 = BaggingClassifier(LogisticRegression(penalty='l1'))

model5.fit(X_train,y_train)
score5 = model5.score(X_test,y_test)
print ("Accuracy from LG::",score5)

model = VotingClassifier(estimators=[('xgb', model1), ('cb', model2),('gbm',model4),('lda',model6),('lg',model5)], voting='hard')
model.fit(X_train,y_train)
print (model.score(X_test,y_test))

