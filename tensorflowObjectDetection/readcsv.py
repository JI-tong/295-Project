import csv
# get how many categories we have
maps = []
with open('./data/nivida_labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for i, row in enumerate(csv_reader):
        if i == 0:
            continue
        veh_type = row[3]
        if veh_type not in maps:
            maps.append(veh_type)
    print('Processed {line_count} lines.')

print(maps)
