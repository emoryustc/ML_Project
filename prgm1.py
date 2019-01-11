import re

import numpy as np

seed = 2018
np.random.seed(seed)
dataset = np.loadtxt('/home/arjun/PycharmProjects/ML_proj/dataset/train.csv', dtype=str, delimiter=",")
# print(dataset[:,3])
# %%
# Creating the required fields in the dataset
data = dataset[1:, 0:4]
data = np.hstack((data, dataset[1:, 5:]))
# print (data)
dog_data = np.empty((9,), str)
cat_data = np.empty((9,), str)
# dog dataset
for row in data[1:, :]:
    # print (row)
    if row[4] == "Dog":
        dog_data = np.vstack((dog_data, row))
    elif row[4] == "Cat":
        cat_data = np.vstack((cat_data, row))
# %%
dog_data = dog_data[1:, :]
cat_data = cat_data[1:, :]
# Adding gender field and neutered field
dog_data = np.hstack((dog_data, np.ones((dog_data.shape[0], 2), int)))
cat_data = np.hstack((cat_data, np.ones((cat_data.shape[0], 2), int)))

# %%
counter = 0
for i in dog_data[:, :]:
    if "Female" in i[5]:
        dog_data[counter, 9] = 0
    if "Neutered" in i[5]:
        dog_data[counter, 10] = 0
    counter += 1
counter = 0
for i in cat_data[:, :-1]:
    if "Female" in i[5]:
        cat_data[counter, 9] = 0
    if "Neutered" in i[5]:
        cat_data[counter, 10] = 0
    counter += 1

# %% Creating the target set, which is a one hot vector of the 5 possible classes

target_dog = dog_data[:, 3]
print(target_dog)
vector = {'Adoption': np.array([1, 0, 0, 0, 0])
    , 'Died': np.array([0, 1, 0, 0, 0])
    , 'Euthanasia': np.array([0, 0, 1, 0, 0])
    , 'Return_to_owner': np.array([0, 0, 0, 1, 0])
    , 'Transfer': np.array([0, 0, 0, 0, 1])}
integer_encoded_dog = [vector[str] for str in target_dog]
print(integer_encoded_dog)
# Cat encoding
target_cat = cat_data[:, 3]
print(target_cat)
vector = {'Adoption': np.array([1, 0, 0, 0, 0])
    , 'Died': np.array([0, 1, 0, 0, 0])
    , 'Euthanasia': np.array([0, 0, 1, 0, 0])
    , 'Return_to_owner': np.array([0, 0, 0, 1, 0])
    , 'Transfer': np.array([0, 0, 0, 0, 1])}
integer_encoded_cat = [vector[str] for str in target_cat]
print(integer_encoded_cat)

# %%

# Data Preprocessing
# Color field : First we count the occurence of all the unique lables that are present,
# then we find the mean of the counts and the label associated with this mean count.
# Then we replace all the labels with single  treat dogs and cats separately in all caseoccurences to the label with the mean occurence.
# This is known as imputation.

# color = dataset[1:,9]
# print (color)
# unique,pos = np.unique(color,return_inverse=True)
# counts = np.bincount(pos)
# print("counts",counts)
# mean = np.where(counts == np.int(np.round(np.mean(counts))))
# # min = np.where(counts == counts.min())
# min = np.where(counts<=50)
# for s in np.nditer(unique[min]):
#         color[color == s] = color[color == unique[mean]][0]


# Now the unique number of classes in color = 60
# Need to think of furthur ways to reduce the number of elements
# Need to create a onehot vector possibly for all the results.

counter = 0
dog_color = dog_data[:, 8]
for color in dog_color:
    if "/" in color:
        print("multi")
        #     one for multi
        dog_color[counter] = 1
    else:
        print("uni")
        #   zero for uni
        dog_color[counter] = 0
    counter += 1
# %%
# Preprocessing for the breeds section.
# For the breeds,
# the entire section can be subdivided into mixed and not mixed.

# %%
# Preprocessing for the naming of the animal.
# is name present :1 else 0

counter = 0
dog_names = dog_data[:, 1]
for i in dog_data[:, 1]:
    if i == "":
        dog_names[counter] = 0
    else:
        dog_names[counter] = 1
    counter += 1
print(dog_names)

# %%
# Calculating the age of the animal in days.

AgeUponOutcome = dataset[1:, 7]
day_set = []
for entry in AgeUponOutcome:
    if entry == "":
        days = 0
        day_set = np.append(day_set, days)
        continue
    first = np.int(entry.split(" ")[0])
    second = entry.split(" ")[1]
    if re.match(second, ("years|year")):
        days = 365 * first
    if re.match(second, ("months|month")):
        days = 30 * first
    if re.match(second, ("weeks|week")):
        days = 7 * first
    if re.match(second, ("days|day")):
        days = first
    day_set = np.append(day_set, days)

print(day_set)

# %% Sex upon outcome is a one hot vector with 5 possibilities, neuterd male, neutered female,intact female, intact male and unknown.
# It is a one hot vector of these possibilities.
#
# sexuponoutcome = dataset[1:,6]
# count = 0
# for data in sexuponoutcome:
#     if data == "":
#         target[count] = "Unknown"
#     if "Male" in data:
#     count += 1
#
#
#
# animal_gender = []
