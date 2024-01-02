from collections import defaultdict
import csv

def flatten_dict(d, model_name, parent_keys=None, sep='_'):
    items = []
    parent_keys = parent_keys or []
    for k, v in d.items():
        current_keys = parent_keys + [k]
        if isinstance(v, dict):
            items.extend(flatten_dict(v, model_name, current_keys, sep=sep))
        else:
            items.append((model_name, *current_keys, v))
    return items

def write_to_csv(results, csv_file_path='metrics.csv', readable=True):
    if len(csv_file_path.split('.')) == 1:
        csv_file_path = str(csv_file_path + '.csv')

    flat_data = []
    last_row = [None]
    for main_key, nested_dict in results.items():
        flat_data.extend(flatten_dict(nested_dict, main_key))

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow(['Model', 'Space', 'Metric', 'Value'])
        for row in flat_data:
        	if readable and row[0] == last_row[0]:
        		if row[1] == last_row[1]:
        			csv_writer.writerow(['','', *row[2:]])
        		else:
        			csv_writer.writerow(['', *row[1:]])
        	else:
        		csv_writer.writerow(row)
        	last_row = row

    print(f'Data has been written to {csv_file_path}')





