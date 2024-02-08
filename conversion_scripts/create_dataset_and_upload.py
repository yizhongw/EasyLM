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
            if 'prompt' in data[0]:
                for d in data:
                    d.pop('prompt')
                    d.pop('prompt_id')
            def genx():
                for d in data:
                    yield d
            dataset = Dataset.from_generator(genx)
        all_ds[file.split('.')[0]] = dataset
import pdb; pdb.set_trace()
all_ds.push_to_hub("allenai/preference-datasets-tulu-fixed")
