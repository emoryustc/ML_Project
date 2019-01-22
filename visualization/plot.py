import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mla = pd.read_pickle('./comparison.pkl')
mla = mla.drop(['MLA Time', 'MLA Test Accuracy 3*STD', 'MLA Parameters'], axis=1)
mla = mla.rename(index=str, columns={'MLA Name': 'Name', 'MLA Test Accuracy Mean': 'Test Accuracy',
                                     'MLA Train Accuracy Mean': 'Train Accuracy'})

mla = mla.append({'Name': 'NeuralNetwork(1 layer)', 'Test Accuracy': 0.516, 'Train Accuracy': 0.513},
                 ignore_index=True)
mla = mla.append({'Name': 'NeuralNetwork(2 layers)', 'Test Accuracy': 0.540, 'Train Accuracy': 0.550},
                 ignore_index=True)
mla = mla.append({'Name': 'NeuralNetwork(3 layers)', 'Test Accuracy': 0.551, 'Train Accuracy': 0.555},
                 ignore_index=True)
mla = mla.append({'Name': 'NeuralNetwork(4 layers)', 'Test Accuracy': 0.551, 'Train Accuracy': 0.552},
                 ignore_index=True)
mla = mla.append({'Name': 'NeuralNetwork(5 layers)', 'Test Accuracy': 0.562, 'Train Accuracy': 0.559},
                 ignore_index=True)
mla = mla.append({'Name': 'NeuralNetwork(7 layers)', 'Test Accuracy': 0.565, 'Train Accuracy': 0.557},
                 ignore_index=True)
mla = mla.append({'Name': 'NeuralNetwork(8 layers)', 'Test Accuracy': 0.459, 'Train Accuracy': 0.547},
                 ignore_index=True)
mla = mla.append({'Name': 'NeuralNetwork(4 layers, auto-encoder)', 'Test Accuracy': 0.545, 'Train Accuracy': 0.582},
                 ignore_index=True)
mla = mla.sort_values(by=['Test Accuracy'])
print(mla)

mla.plot.barh(x='Name', rot=0)
plt.grid(zorder=0)
plt.savefig('./comparison.png', format='png', dpi=600, bbox_inches='tight')
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
