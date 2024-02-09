from datasets import Dataset, DatasetDict
import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
parser.add_argument('--output_name', type=str)
parser.add_argument('--max_samples', type=int, default=5_000_000)
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
            if len(dataset) > args.max_samples:
                dataset = dataset.shuffle(seed=42).select(range(args.max_samples))
        all_ds[file.split('.')[0]] = dataset

all_ds.push_to_hub(args.output_name)
