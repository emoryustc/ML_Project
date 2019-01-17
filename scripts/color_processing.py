import pandas as pd
import numpy as np

#patterning function separating different patterns from the color data
def get_pattern(color):
    pattern = ''
    for patterntype in patternlist:
        count = 0
        if (patterntype in color) & (count == 0):
            pattern += patterntype
            count += 1
        elif (patterntype in color) & (count > 0):
            pattern = pattern + '/' + patterntype
            count += 1
            print(pattern)
    return pattern

def coloring():
    pass


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


for i in df.index:
    pattern = get_pattern(df.at[i, 'Color'])
    # print('color',df.at[i, 'Color'],'pattern',pattern)
    df.at[i, 'Pattern'] = pattern
    # if '/' in pattern:
    # mixedlist.append(pattern)
    # df['Pattern'] = "mixed"
    # print('new',df['Pattern'])


if __name__ == '__main__':
    patterning()