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


features = ["Name_length",
            "Name_frequency",
            "Named/un-named",
            "Year",
            "Month",
            "Day",
            "Hour",
            "Day_of_week"
            "Male/Female",
            "Neutered/Intact",
            "Age",
            "Breed1",
            "Breed2",
            "Breed3",
            "Breed4",
            "Breed5",
            "Breed6",
            "Breed7",
            "Breed8",
            "Breed9",
            "Breed10",
            "Color-light",
            "Color-medium",
            "Color-dark",
            "Color-warm",
            "Color-medium",
            "Color-cold",
            "Color_feature1",
            "Color_feature2",
            ]
ceate_feature_map(features)
dtrain = xgb.DMatrix(dataset, label=outcome)
num_rounds = 50
xgb_params = {"colsample_bytree": 0.39258925829065067,
              "learning_rate": 0.49227635408236803, "max_depth": 65, "n_estimators": 168, "nthread": 8,
              "reg_alpha": 17.7827941, "subsample": 0.93124726513080813}
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
plt.gcf().savefig('feature_importance_xgb.png', bbox_inches='tight')
plt.show()
