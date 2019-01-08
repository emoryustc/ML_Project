import numpy as np
import re


seed = 2018
np.random.seed(seed)
dataset = np.loadtxt('/home/arjun/PycharmProjects/ML_proj/dataset/train.csv',dtype=str, delimiter=",")
print(dataset[:,3])

# Creating the required fields in the dataset
data = dataset[:,0:3]
data = np.hstack((data,dataset[:,4::]))
# print (data)
#%% Creating the target set, which is a one hot vector of the 5 possible classes

target = dataset[1:,3]
print (target)
vector = {'Adoption':np.array([1,0,0,0,0])
         ,'Died':np.array([0,1,0,0,0])
         ,'Euthanasia':np.array([0,0,1,0,0])
         ,'Return_to_owner':np.array([0,0,0,1,0])
         ,'Transfer':np.array([0,0,0,0,1])}
integer_encoded = [vector[str] for str in target]
print (integer_encoded)
# %%

# Data Preprocessing
# Color field : First we count the occurence of all the unique lables that are present,
# then we find the mean of the counts and the label associated with this mean count.
# Then we replace all the labels with single occurences to the label with the mean occurence.
# This is known as imputation.

color = dataset[:,9]
print (color)
unique,pos = np.unique(color,return_inverse=True)
counts = np.bincount(pos)
print("counts",counts)
mean = np.where(counts == np.int(np.round(np.mean(counts))))
# min = np.where(counts == counts.min())
min = np.where(counts<=50)
for s in np.nditer(unique[min]):
        color[color == s] = color[color == unique[mean]][0]

# Now the unique number of classes in color = 60
# Need to think of furthur ways to reduce the number of elements
# Need to create a onehot vector possibly for all the results.

# %%
# Preprocessing for the breeds section.
# For the breeds,
# the entire section can be subdivided into mixed and not mixed.

# %%
# Preprocessing for the naming of the animal.
# is name present :1 else 0

names = []
naming_list = dataset[1:,1]
for i in naming_list:
    if i == "":
        names = np.append(names,0)
    else:
        names = np.append(names,1)

print (names)

# %%

AgeUponOutcome = dataset[1:,7]
day_set = []
for entry in AgeUponOutcome:
    if entry == "":
        days = 0
        day_set = np.append(day_set, days)
        continue
    first = np.int(entry.split(" ")[0])
    second = entry.split(" ")[1]
    if re.match(second,("years|year")):
        days = 365 * first
    if re.match(second,("months|month")):
        days = 30 * first
    if re.match(second,("weeks|week")):
        days = 7 * first
    if re.match(second,("days|day")):
        days = first
    day_set = np.append(day_set,days)

print (day_set)



# %%


