import pandas as pd
import numpy as np

# patterning function separating different patterns from the color data

df = pd.read_csv('../dataset/train.csv', sep=',')
patternlist = ["Agouti", "Brindle", "Calico", "Merle", "Point", "Smoke", "Tabby", "Tick", "Tiger", "Torbie", "Tortie",
               "Tricolor"]


def get_colorset():  # get all the colors
    colorlist = list(set(df["Color"]))
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
    return (count)


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


if __name__ == '__main__':
    colorset = list(get_colorset())

    for i in df.index:
        # get color number
        colornumber = get_color(df.at[i, 'Color'])
        # print(colornumber)

        if colornumber == 0:
            df.at[i, 'ColorCategory'] = "unknow"
        elif colornumber == 1:
            df.at[i, 'ColorCategory'] = "unicolor"
        elif colornumber == 2:
            df.at[i, 'ColorCategory'] = "two-tones"
        elif colornumber == 3:
            df.at[i, 'ColorCategory'] = "tricolor"

        print('Colornumber', df.at[i, 'ColorCategory'], "color", df.at[i, 'Color'])

        # get patterns
        pattern = get_pattern(df.at[i, 'Color'])
        # print('color',df.at[i, 'Color'],'pattern',pattern)
        df.at[i, 'Pattern'] = pattern
        # if '/' in pattern:
        # mixedlist.append(pattern)
        # df['Pattern'] = "mixed"
        # print('new',df['Pattern'])
