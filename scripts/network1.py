import numpy as np
from scripts import prgm1
from sklearn.metrics import accuracy_score
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# %% Preparing the dataset and the putput label
dataset = np.loadtxt('../dataset/train.csv', dtype=str, delimiter=",")
dataset, outcome = prgm1.pre_processing(dataset)
partition = np.round(0.8 * dataset.shape[0]).__int__()
train_set = dataset[0:partition, :]
test_set = dataset[partition:, :]

# %% Training

test_outcome = np.array(outcome[partition:]).astype(int)
train_outcome = np.array(outcome[0:partition]).astype(int)

seed = 2017
np.random.seed(seed)
ensemble = SuperLearner(scorer=accuracy_score, random_state=seed, verbose=2)
# Build the first layer
ensemble.add([RandomForestClassifier(random_state=seed), SVC()])
# # Attach the final meta estimator
ensemble.add_meta(LogisticRegression())
# # Fit ensemble
ensemble.fit(train_set, train_outcome, gamma="auto")
# # Predict
preds = ensemble.predict(test_set)
print("Fit data:\n%r" % ensemble.data)
print("Prediction score: %.3f" % accuracy_score(preds, test_outcome))
