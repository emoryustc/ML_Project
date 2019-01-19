import re
from scripts import breed_processing, color_processing
import numpy as np
import datetime


def pre_processing(dataset):
    """
    Preprocess data

    :param dataset:
    :return:
    """
    dog_count = np.count_nonzero(dataset[:, 5] == 'Dog')
    new_dataset = np.zeros((dog_count, 36))
    new_outcome = np.zeros((dog_count, 1))

    outcome_vector = {'Adoption': 16, 'Died': 8, 'Euthanasia': 4, 'Return_to_owner': 2, 'Transfer': 1}

    unique, counts = np.unique(dataset[:, 1], return_counts=True)
    frequency = dict(zip(unique, counts))
    frequency[''] = 0
    max_occurrence_times = max(frequency.values())
    max_length_of_names = len(max(frequency.keys(), key=len))

    set_years = set()
    for i in range(dataset.shape[0]):
        if 'years' in dataset[i, 7]:
            set_years.add(int(dataset[i, 7].split(' ')[0]))
    max_age_in_year = max(set_years)

    dog_breeds = breed_processing.breeder()
    indices_breeds = {"Herding": 0, 'Hound': 1, 'Mix': 2, 'Non-Sporting': 3,
                      'Pit Bull': 4, 'Sporting': 5, 'Terrier': 6, 'Toy': 7,
                      'Unknown': 8, 'Working': 9}
    indices_pattern = {'Merle': 0, 'Tabby': 1, 'Tick': 2, 'Brindle': 3, 'Tricolor': 4, 'Tiger': 5, 'Smoke': 6}

    index_for_new_dataset_row = 0

    for i in range(dataset.shape[0]):
        index_for_new_dataset_column = 0
        if i % 100 == 0:
            print('Processing No.' + str(i) + ' record...')

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
            # print('Name processing finished')

            # Datatime processing
            a_data = dataset[i, 2].split(' ')[0].split('-')
            a_time = dataset[i, 2].split(' ')[-1].split(':')
            # 3 offset
            new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = (int(a_data[0]) - 2013) / 3.0  # Year
            index_for_new_dataset_column += 1
            # 4 offset
            new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = int(a_data[1]) / 12.0  # Month
            index_for_new_dataset_column += 1
            # 5 offset
            new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = int(a_data[2]) / 31.0  # Day
            index_for_new_dataset_column += 1

            # 6 offset
            new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = int(a_time[0]) / 24.0  # Hour
            index_for_new_dataset_column += 1
            # 7 offset
            new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = datetime.date(int(a_data[0]),
                                                                                                 int(a_data[1]),
                                                                                                 int(a_data[2])) \
                                                                                       .weekday() / 6.0
            index_for_new_dataset_column += 1
            # print('Date time processing finished')

            # Sex processing
            # 8 offset
            if "Female" in dataset[i][6]:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1

            # 9 offset
            if "Neutered" in dataset[i][6]:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1
            # print('Gender processing finished')

            # Age processing
            # Calculating the age of the animal in days.
            if dataset[i, 7] == '':
                days = 0
            else:
                first = np.int(dataset[i, 7].split(' ')[0])
                second = dataset[i, 7].split(' ')[1]
                if re.match(second, "years|year"):
                    days = 365 * first
                if re.match(second, "months|month"):
                    days = 30 * first
                if re.match(second, "weeks|week"):
                    days = 7 * first
                if re.match(second, "days|day"):
                    days = first
            # 10 offset
            new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = np.sqrt(days) / np.sqrt(
                max_age_in_year * 365)
            index_for_new_dataset_column += 1
            # print('Age processing finished')

            # Breed processing
            # 11-20 offset
            for breed in dog_breeds[index_for_new_dataset_row]:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column + indices_breeds[breed]] = 1
            index_for_new_dataset_column += 10
            # print('Breed processing finished')

            # Color processing
            features = color_processing.get_and_set_param_by_color_theory(dataset[i, 9])

            # 21 offset
            if 'l-light' in features:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1
            # 22 offset
            if 'l-medium' in features:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1
            # 23 offset
            if 'l-dark' in features:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1
            # 24 offset
            if 'warm' in features:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1
            # 25 offset
            if 'medium' in features:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1
            # 26 offset
            if 'cold' in features:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1

            # 27 offset
            count = 0
            for feature in features:
                if 'l-' in feature:
                    count += 1
            if count > 1:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1
            # 28 offset
            count = 0
            for feature in features:
                if 'l-' not in feature:
                    count += 1
            if count > 1:
                new_dataset[index_for_new_dataset_row, index_for_new_dataset_column] = 1
            index_for_new_dataset_column += 1

            # 29-35 offset
            for feature in features:
                if feature in indices_pattern.keys():
                    new_dataset[index_for_new_dataset_row, index_for_new_dataset_column + indices_pattern[feature]] = 1
            index_for_new_dataset_column += 7

            # Create Arjun's outcome file
            new_outcome[index_for_new_dataset_row] = outcome_vector[dataset[i, 3]]

            # Row count ++
            index_for_new_dataset_row += 1

    # print(new_dataset[:, 30:36][30:60])
    # print(len(new_outcome))
    return new_dataset, new_outcome


if __name__ == '__main__':
    dataset = np.loadtxt('../dataset/train.csv', dtype=str, delimiter=",")
    new_dataset, new_outcome = pre_processing(dataset)
    np.save("dataset2.npy", new_dataset)
    np.save("outcome2.npy", new_outcome)
