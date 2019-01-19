import re
from scripts import breed_processing
import numpy as np
import datetime


def pre_processing(dataset):
    """
    Preprocess data

    :param dataset:
    :return:
    """
    dog_count = np.count_nonzero(dataset[:, 5] == 'Dog')
    new_dataset = np.zeros((dog_count, 22))

    unique, counts = np.unique(dataset[:, 1], return_counts=True)
    frequency = dict(zip(unique, counts))
    frequency[''] = 0
    max_occurrence_times = max(frequency.values())
    max_length_of_names = len(max(frequency.keys(), key=len))

    index_for_new_dataset_row = 0

    for i in range(dataset.shape[0]):
        index_for_new_dataset_column = 0

        # Process dog data
        if dataset[i][5] == 'Dog':
            # Name processing
            # 0 offset
            new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = len(
                dataset[i][1]) / max_length_of_names
            index_for_new_dataset_column += 1
            # 1 offset
            new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = frequency[dataset[i][
                1]] / max_occurrence_times
            index_for_new_dataset_column += 1
            # 2 offset
            if dataset[i][1] == '':
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 0
            else:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1

            # Datatime processing
            # 3 offset



            index_for_new_dataset_row += 1
            pass


    print(new_dataset[:10])
    return None

    counter = 0
    for i in dog_data[:, :]:
        if "Female" in i[5]:
            dog_data[counter, 9] = 0
        if "Neutered" in i[5]:
            dog_data[counter, 10] = 0
        counter += 1

    # Creating the target set, which is a one hot vector of the 5 possible classes
    target_dog = dog_data[:, 3]
    print(target_dog)
    vector = {'Adoption': 16
        , 'Died': 8
        , 'Euthanasia': 4
        , 'Return_to_owner': 2
        , 'Transfer': 1}
    integer_encoded_dog = [vector[str] for str in target_dog]
    print(integer_encoded_dog)

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

    # %%
    # Preprocessing for the naming of the animal.
    # is name present :1 else 0

    # Now the unique number of classes in color = 60
    # Need to think of furthur ways to reduce the number of elements
    # Need to create a onehot vector possibly for all the results.
    dog_data = np.hstack((dog_data, np.ones((dog_data.shape[0], 1), int)))
    counter = 0

    for color in dog_data[:, 8]:
        if "/" in color:
            print("uni")
            #     one for multi
            dog_data[counter, (dog_data.shape[1] - 1)] = 0
        counter += 1
        # %%
        # Preprocessing for the breeds section.
        # For the breeds,
        # the entire section can be subdivided into mixed and not mixed.

        # %%
        # Preprocessing for the naming of the animal.
        # is name present :1 else 0
    dog_data = np.hstack((dog_data, np.ones((dog_data.shape[0], 3), int)))
    # %%
    counter = 0
    unique, counts = np.unique(dog_data[:, 1], return_counts=True)
    frequency = dict(zip(unique, counts))
    for i in dog_data[:, 1]:
        if i == "":
            dog_data[counter, dog_data.shape[1] - 3] = 0
            dog_data[counter, dog_data.shape[1] - 2] = 0
            dog_data[counter, dog_data.shape[1] - 1] = 0
        else:
            dog_data[counter, dog_data.shape[1] - 2] = len(i)
            dog_data[counter, dog_data.shape[1] - 1] = frequency[i]
        counter += 1
    temp = dog_data[:, 13]
    temp = np.array(temp).astype(np.float32)
    temp /= np.max(temp)
    dog_data[:, 13] = temp
    temp = dog_data[:, 14]
    temp = np.array(temp).astype(np.float32)
    temp /= np.max(temp)
    dog_data[:, 14] = temp
    # dog_data = np.delete(dog_data,1,1)
    # %%
    # Calculating the age of the animal in days.
    dog_data = np.hstack((dog_data, np.ones((dog_data.shape[0], 1), int)))
    AgeUponOutcome = dog_data[:, 6]
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

    counter = 0
    for i in day_set:
        day_set[counter] /= np.max(day_set)
        counter += 1
    counter = 0
    for i in day_set:
        dog_data[counter, dog_data.shape[1] - 1] = day_set[counter]
        counter += 1
    print(day_set)

    # %% Section for integrating the breed

    dog_data = np.hstack((dog_data, np.ones((dog_data.shape[0], 10), int)))
    # %%
    indices = {"Herding": 16, 'Hound': 17, 'Mix': 18, 'Non-Sporting': 19,
               'Pit Bull': 20, 'Sporting': 21, 'Terrier': 22, 'Toy': 23,
               'Unknown': 24, 'Working': 25}
    dog_breeds = breed_processing.breeder()
    counter = 0
    for breed in dog_breeds[:-1]:
        if "Herding" not in breed:
            dog_data[counter, indices["Herding"]] = 0
        if "Hound" not in breed:
            dog_data[counter, indices["Hound"]] = 0
        if "Mix" not in breed:
            dog_data[counter, indices["Mix"]] = 0
        if "Non-Sporting" not in breed:
            dog_data[counter, indices["Non-Sporting"]] = 0
        if "Pit Bull" not in breed:
            dog_data[counter, indices["Pit Bull"]] = 0
        if "Sporting" not in breed:
            dog_data[counter, indices["Sporting"]] = 0
        if "Terrier" not in breed:
            dog_data[counter, indices["Terrier"]] = 0
        if "Toy" not in breed:
            dog_data[counter, indices["Toy"]] = 0
        if "Unknown" not in breed:
            dog_data[counter, indices["Unknown"]] = 0
        if "Working" not in breed:
            dog_data[counter, indices["Working"]] = 0

        counter += 1
    # %% Intake date processing
    dog_data = np.hstack((dog_data, np.ones((dog_data.shape[0], 5), int)))
    # %%
    date_array = np.core.defchararray.split(dog_data[:, 2], "-")
    counter = 0
    for date in date_array:
        dog_data[counter, dog_data.shape[1] - 5] = date[0]  # Year
        dog_data[counter, dog_data.shape[1] - 4] = date[1]  # month
        dog_data[counter, dog_data.shape[1] - 3] = date[2].split(" ")[0]  # day
        dog_data[counter, dog_data.shape[1] - 2] = date[2].split(" ")[1].split(":")[0]  # hour
        dog_data[counter, dog_data.shape[1] - 1] = datetime.datetime(
            np.int(date[0]), np.int(date[1]), np.int(date[2].split(" ")[0])
        ).weekday()
        counter += 1
    for i in range(26, 31):
        temp = dog_data[:, i]
        temp = np.array(temp).astype(np.float32)
        temp /= np.max(temp)
        dog_data[:, i] = temp

    # %%Deleting unwanted collums from the dataset,
    # and preparing the final dataset to be fed to the network.

    dog_data = np.delete(dog_data, slice(0, 9), 1)

    return dog_data, integer_encoded_dog


if __name__ == '__main__':
    dataset = np.loadtxt('../dataset/train.csv', dtype=str, delimiter=",")
    dataset, outcome = pre_processing(dataset)
    np.save("dataset1.npy", dataset)
    np.save("outcome1.npy", outcome)
