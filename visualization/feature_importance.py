import pandas as pd
from matplotlib import pylab as plt
import operator
import numpy as np
import xgboost as xgb

dataset = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/dataset.npy")
outcome = np.load("/home/arjun/PycharmProjects/ML_proj/scripts/outcome.npy")
# %%
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
features = ["Male/Female",
            "Neutered/Intact",
            "Single/Multi-color",
            "Named/un-named",
            "name_length",
            "name_frequency",
            "age",
            "breed1",
            "breed2",
            "breed3",
            "breed4",
            "breed5",
            "breed6",
            "breed7",
            "breed8",
            "breed9",
            "breed10",
            "year",
            "month",
            "day",
            "hour",
            "d_o_w"]
ceate_feature_map(features)
dtrain = xgb.DMatrix(dataset, label=outcome)
num_rounds = 50
xgb_params={"colsample_bytree": 0.39258925829065067,
 "learning_rate": 0.49227635408236803, "max_depth": 65, "n_estimators":168, "nthread":8,
 "reg_alpha": 17.7827941, "subsample":0.93124726513080813}
gbdt = xgb.train(xgb_params, dtrain, num_rounds)

importance = gbdt.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')
plt.show()