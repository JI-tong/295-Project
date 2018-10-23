import csv
import pandas as pd

def main():
    class_list = {
        'car' : '1',
        'bus' : '2',
        'van' : '3',
        'others' : '4',
    }
    column_name = ['class_name','id']
    classes = ['car','bus','van','others']
    class_list = []
    for (i,item) in enumerate(classes):
        value = (item, str(i + 1))
        class_list.append(value)

    data = pd.DataFrame(class_list, columns=None)
    data.to_csv(('images/class_map.csv'), index=None, header=None)
    print('Successfully generated class_map.csv')

main()