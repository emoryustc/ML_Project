
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from catboost import CatBoostClassifier

dataset = np.load("/Users/jieyab/ML_Project/scripts/dataset1.npy")
outcome = np.load("/Users/jieyab/ML_Project/scripts/outcome1.npy")

x_train, x_test, y_train, y_test = train_test_split(dataset, outcome, test_size=0.25, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

model = CatBoostClassifier(
    iterations=500, depth=10,
                           learning_rate=0.1,
                           loss_function='MultiClass',
                           # bagging_temperature=2,
                           # l2_leaf_reg=4)
                           )
model.fit(pd.DataFrame(x_train),pd.DataFrame(y_train),
          eval_set=(pd.DataFrame(x_val),pd.DataFrame(y_val)),plot=True)

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

print(model.score(x_test,y_test))
plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.bar(range(len(model.get_feature_importance(prettified=False))), model.get_feature_importance(prettified=False))
plt.title("Cat Feature Importance")
plt.xticks(range(len(model.get_feature_importance(prettified=False))), features, rotation='vertical');
plt.gcf().savefig('feature_importance_catboost.png')
plt.show()
