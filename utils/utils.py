from collections import defaultdict
import csv
import os
import json

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

def write_to_csv(results, config, readable=True):
    file_name = config.save_file_path
    root = "/content/drive/MyDrive/DeepLearningProject" if config.colab else os.getcwd()
    csv_file_path = os.path.join(root, file_name)
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

    print(f"---wrote data to {csv_file_path}")

def load_from_csv(file_name, config, root = None):
    data_dict = defaultdict(dict)
    if not root:
        root = "/content/drive/MyDrive/DeepLearningProject" if config.colab else os.getcwd()
    csv_file_path = os.path.join(root, file_name)
    if len(csv_file_path.split('.')) == 1:
        csv_file_path = str(csv_file_path + '.csv')

    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)

        last_row = [None, None, None, None]  

        for row in csv_reader:
            model_name, metric_type, metric_name, value = row

            model_name = model_name if model_name != '' else last_row[0]
            metric_type = metric_type if metric_type != '' else last_row[1]

            current_dict = data_dict[model_name].setdefault(metric_type, defaultdict(dict))
            current_dict[metric_name] = float(value)

            last_row = [model_name, metric_type, metric_name, value]

    return data_dict

def save_results_to_json(results, filepath):
    """ Save the results dictionary to a JSON file. """
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    config = {"colab":False}
    results = load_from_csv("results_vgg11_cifar10", config, root = "/Users/benjaminjaeger/Downloads/")
    print(json.dumps(results, indent=4))

