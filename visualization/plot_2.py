import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mla = pd.read_pickle('./comparison.pkl')
mla = mla.drop(
    ['MLA Time', 'MLA Test Accuracy 3*STD', 'MLA Parameters', 'MLA Test Accuracy Mean', 'MLA Train Accuracy Mean'],
    axis=1)
mla = mla.rename(index=str, columns={'MLA Name': 'Name'})
mla = mla.iloc[0:0]

mla = mla.append(
    {'Name': 'Stacking', 'Model3': 0.576, 'Result': 0.575, 'Model2': 0.573, 'Model1': 0.553, 'Model4': 0.550},
    ignore_index=True)
mla = mla.append(
    {'Name': 'Bagging', 'CatBoost': 0.579, 'XGB': 0.577, 'XBM': 0.571, 'LogisticRegression': 0.544, 'LDA': 0.256},
    ignore_index=True)
mla = mla.append(
    {'Name': 'Blending', 'Result': 0.572},
    ignore_index=True)
mla = mla.append(
    {'Name': 'Max voting', 'Result': 0.581},
    ignore_index=True)
mla = mla.sort_values(by=['Result'])
print(mla)

ax = mla.plot.barh(x='Name', rot=0, width=1.5)
# plt.legend(loc='top left', bbox_to_anchor=(1.0, 0.5))
ax.set_xlim(0, 1)
plt.grid(zorder=0)
plt.savefig('./comparison2.png', format='png', dpi=600, bbox_inches='tight')
# plt.show()
# print(mla.dtypes)

# # mla['Accuracy Mean'] = [['MLA Train Accuracy Mean', 'MLA Test Accuracy Mean']]
#
# sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=mla, color='grey')
#
# # prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
# plt.title('Machine Learning Algorithm Accuracy Score \n')
# plt.xlabel('Accuracy Score (%)')
# plt.ylabel('Algorithm')
# plt.show()
