from datasets import Dataset, DatasetDict
import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
args = parser.parse_args()

all_ds = DatasetDict()

# iterate over jsonl files in the folder
for file in os.listdir(args.folder):
    if file.endswith('.jsonl'):
        with open(os.path.join(args.folder, file), 'r') as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]
            dataset = Dataset.from_dict({'data': data}, split='train')
        all_ds[file.split('.')[0]] = dataset

all_ds.push_to_hub("allenai/preference-datasets-tulu")
