import pandas as pd
import numpy as np

# patterning function separating different patterns from the color data


patternlist = ["Agouti", "Brindle", "Calico", "Merle", "Point", "Smoke", "Tabby", "Tick", "Tiger", "Torbie", "Tortie",
               "Tricolor"]

unique_color_set = ['White', 'Red', 'Pink', 'Silver', 'Flame', 'Blue', 'Liver', 'Ruddy', 'Lilac', 'Tan', 'Chocolate',
                    'Yellow', 'Gray', 'Seal', 'Apricot', 'Buff', 'Lynx', 'Fawn', 'Orange', 'Gold', 'Sable', 'Cream',
                    'Black', 'Brown']

unique_color_set_for_dog = ['Pink', 'Silver', 'Ruddy', 'Blue', 'Tan', 'Liver', 'White', 'Sable', 'Orange', 'Gray',
                            'Chocolate', 'Yellow', 'Fawn', 'Brown', 'Apricot', 'Cream', 'Buff', 'Gold', 'Black', 'Red']
unique_color_group_for_dog = ['light', 'light', 'l-medium', 'l-medium', 'dark', 'dark', 'light', 'dark', 'l-medium',
                              'l-medium',
                              'dark', 'light', 'light', 'dark', 'light', 'light', 'light', 'light', 'dark', 'l-medium']
unique_color_group2_for_dog = ['warm', 'medium', 'warm', 'cold', 'medium', 'medium', 'medium', 'medium', 'warm',
                               'medium', 'medium', 'warm', 'warm', 'medium', 'warm', 'medium', 'warm', 'warm', 'medium',
                               'warm']

unique_pattern_set_for_dog = ['Merle', 'Tabby', 'Tick', 'Brindle', 'Tricolor', 'Tiger', 'Smoke']


def get_colorset():  # get all the colors
    colorlist = list(set(df["Color"]))
    # print(colorlist)

    list_pattern = set()
    for item in colorlist:
        l_item = item.split('/')
        if len(item.split('/')) == 2:
            for l_i in l_item:
                if len(l_i.split(' ')) == 2:
                    list_pattern.add(l_i.split(' ')[-1])
        else:
            if len(l_item[0].split(' ')) == 2:
                list_pattern.add(l_item[0].split(' ')[-1])
    # print(list_pattern)

    newlist = []
    # split items by '/'
    for item in colorlist:
        if '/' in item:
            item1, item2 = item.split("/")
            newlist.extend((item1, item2))
        else:
            newlist.append(item)

    truelist = []
    # further split items by space
    for item in newlist:
        if " " in item:
            alist = []
            alist = item.split(" ")
            truelist.extend(alist)
        else:
            truelist.append(item)

    colorset = set(truelist) - set(patternlist)  # colorset is the color excluding the patterns
    # print(set(colorset))
    # print(set(truelist) - set(colorset))
    return colorset


def get_color(color):
    count = 0
    if 'Tricolor' in color:
        count = 3
        # print(count)
    else:
        for colortype in colorset:
            if (colortype in color) & (count == 0):
                count += 1
            # print(item, count)
            elif (colortype in color) & (count > 0):
                count += 1
    return count


def get_pattern(color):
    pattern = ''
    count = 0

    for patterntype in patternlist:
        if (patterntype in color) & (count == 0):
            pattern += patterntype
            count += 1
        elif (patterntype in color) & (count > 0):
            pattern = pattern + '/' + patterntype
            count += 1

    return pattern


def get_and_set_param_by_color_theory(color_des):
    """
    Set the color relevant parameters
    If animal has multi colors, all of them are taken into consideration
    - Light / Medium / Dark
    - Warm / Medium / Cold

    :param color_des:       color description
    :return:
    """

    color_group_1 = np.array([unique_color_set_for_dog, unique_color_group_for_dog]).T
    # dog_group_1 = np.unique(color_group_1[:, 1])

    color_group_2 = np.array([unique_color_set_for_dog, unique_color_group2_for_dog]).T
    # dog_group_2 = np.unique(color_group_2[:, 1])

    list_feature = set()
    for item in color_des.split('/'):
        for des in item.split(' '):
            if des in color_group_1[:, 0]:
                ind1 = np.where(color_group_1[:, 0] == des)[0]
                # print(des)
                # print(color_group_1[ind1, 1][0])
                ind2 = np.where(color_group_2[:, 0] == des)[0]
                # print(color_group_2[ind2, 1][0])
                # print()
                list_feature.add(color_group_1[ind1, 1][0])
                list_feature.add(color_group_2[ind2, 1][0])

            if des in unique_pattern_set_for_dog:
                list_feature.add(des)

    # print(list_feature)
    return list_feature


if __name__ == '__main__':
    df = pd.read_csv('../dataset/train.csv', sep=',')
    # df = df[df.AnimalType == 'Dog']
    df = df[df['AnimalType'] == 'Dog']
    print(df)

    colorset = list(get_colorset())
    dataset = np.load('dataset.npy')
    print(dataset.shape)
    dataset = np.hstack((dataset, np.zeros((dataset.shape[0], 6), dtype=dataset.dtype)))
    shape0 = dataset.shape[1]

    index_for_dataset = 0
    for i in df.index:
        # print(index_for_dataset)
        features = get_and_set_param_by_color_theory(df.at[i, 'Color'])
        # print(features)

        for j in range(6):
            dataset[index_for_dataset][shape0 - j - 1] = 0

        if 'light' in features:
            dataset[index_for_dataset][shape0 - 1] = 1
        if 'l-medium' in features:
            dataset[index_for_dataset][shape0 - 2] = 1
        if 'dark' in features:
            dataset[index_for_dataset][shape0 - 3] = 1
        if 'warm' in features:
            dataset[index_for_dataset][shape0 - 4] = 1
        if 'medium' in features:
            dataset[index_for_dataset][shape0 - 5] = 1
        if 'cold' in features:
            dataset[index_for_dataset][shape0 - 6] = 1

    #     # get color number
    #     colornumber = get_color(df.at[i, 'Color'])
    #     # print(colornumber)
    #
    #     if colornumber == 0:
    #         df.at[i, 'ColorCategory'] = "unknow"
    #     elif colornumber == 1:
    #         df.at[i, 'ColorCategory'] = "unicolor"
    #     elif colornumber == 2:
    #         df.at[i, 'ColorCategory'] = "two-tones"
    #     elif colornumber == 3:
    #         df.at[i, 'ColorCategory'] = "tricolor"
    #
    #     # print('Colornumber', df.at[i, 'ColorCategory'], "color", df.at[i, 'Color'])
    #
    #     # get patterns
    #     pattern = get_pattern(df.at[i, 'Color'])
    #     # print('color',df.at[i, 'Color'],'pattern',pattern)
    #     df.at[i, 'Pattern'] = pattern
    #     # if '/' in pattern:
    #     # mixedlist.append(pattern)
    #     # df['Pattern'] = "mixed"
    #     # print('new',df['Pattern'])

        index_for_dataset += 1

    np.save('dataset_color6.npy', dataset)
