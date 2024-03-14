import json
from statistics import mean
from random import Random
from datasets import load_dataset
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_dataset', type=str)
parser.add_argument('--split', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--seed', type=int, default=42)
# for now, max 500 thousans samples.
parser.add_argument('--max_samples', type=int, default=500_000)
args = parser.parse_args()

if '.jsonl' in args.input_dataset:
    dataset = load_dataset('json', data_files=args.input_dataset, split=args.split)
else:
    dataset = load_dataset(args.input_dataset, split=args.split)
dataset = dataset.shuffle(args.seed).select(range(min(args.max_samples, len(dataset))))
new_data = []
random_gen = Random(args.seed)

models = []

 # parse out turns and roles from hh-style turns.
# used for Nectar and HH-RLHF
def parse_out_prompt_turns_hh_format(text):
    prompt_turns = []
    text_split = text.split('Human:')
    for entry in text_split:
        if entry.strip() != '':
            assistant_split = entry.split('Assistant:')
            human_text = assistant_split[0].strip()
            if len(assistant_split) > 1:
                assistant_text = assistant_split[1].strip()
                if human_text:
                    prompt_turns.append({"role": "user", "content": human_text})
                if assistant_text:
                    prompt_turns.append({"role": "assistant", "content": assistant_text})
            else:
                if human_text:
                    prompt_turns.append({"role": "user", "content": human_text})
    return prompt_turns

if args.input_dataset == 'nvidia/HelpSteer':
    # group by prompt
    prompts = {}
    for sample in dataset:
        prompt = sample['prompt']
        if prompt not in prompts:
            prompts[prompt] = []
        prompts[prompt].append(sample)
    # filter out prompts with less than 2 responses
    prompts = {k: v for k, v in prompts.items() if len(v) > 1}

    for prompt, samples in prompts.items():
        samples = sorted(samples, key=lambda x: mean([
            x['helpfulness'],
            x['correctness'],
            x['coherence'],
            x['complexity'],
            # x['verbosity']  - we don't really care about verbosity
        ]))
        chosen = samples[0]
        rejected = random_gen.choice(samples[1:])
        chosen =  [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': chosen['response']},
        ]
        rejected =  [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': rejected['response']},
        ]
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'helpsteer'
        })
elif args.input_dataset == 'berkeley-nest/Nectar':
    for sample in dataset:
        # we can have multiturn data
        initial_turns = parse_out_prompt_turns_hh_format(sample['prompt'])
        answers = sorted(sample['answers'], key=lambda x: x['rank'])
        chosen = answers[0]['answer']
        rejected = random_gen.choice(answers[1:])['answer']
        chosen =  initial_turns + [
            {'role': 'assistant', 'content': chosen},
        ]
        rejected =  initial_turns + [
            {'role': 'assistant', 'content': rejected},
        ]
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'nectar'
        })
elif args.input_dataset == 'argilla/ultrafeedback-binarized-preferences-cleaned':
    for sample in dataset:
        chosen = sample['chosen']
        rejected = sample['rejected']
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'argilla-ultrafeedback',
            'margin': sample['chosen-rating'] - sample['rejected-rating']
        })
elif args.input_dataset == 'argilla/distilabel-capybara-dpo-7k-binarized':
    for sample in dataset:
        chosen = sample['chosen']
        rejected = sample['rejected']
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'argilla-capybara'
        })
elif args.input_dataset == 'argilla/dpo-mix-7k':
    for sample in dataset:
        chosen = sample['chosen']
        rejected = sample['rejected']
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'argilla-capybara'
        })
elif args.input_dataset == 'HuggingFaceH4/ultrafeedback_binarized':
    argilla_dataset = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned', split='train')
    prompts_in_argilla = set([x['prompt'] for x in argilla_dataset])
    for sample in dataset:
        prompt = sample['prompt']
        if prompt in prompts_in_argilla:
            chosen = sample['chosen']
            rejected = sample['rejected']
            new_data.append({
                'chosen': chosen,
                'rejected': rejected,
                'source': 'h4-ultrafeedback'
            })
elif args.input_dataset == 'stanfordnlp/SHP' or args.input_dataset == 'stanfordnlp/SHP-2':
    for el in dataset:
        prompt = {'content': el['history'], 'role': 'user'}
        label = el['labels']
        if label == 1:
            chosen = {'content': el['human_ref_A'], 'role': 'assistant'}
            rejected = {'content': el['human_ref_B'], 'role': 'assistant'}
        else:
            chosen = {'content': el['human_ref_B'], 'role': 'assistant'}
            rejected = {'content': el['human_ref_A'], 'role': 'assistant'}
        data = {}
        data = {'chosen': [prompt, chosen], 'rejected': [prompt, rejected]}
        data['source'] = 'shp'
        new_data.append(data)
elif args.input_dataset == 'Intel/orca_dpo_pairs':
    for sample in dataset:
        prompt = {'role': 'user', 'content': sample['question']}
        chosen = [prompt, {'role': 'assistant', 'content': sample['chosen']}]
        rejected = [prompt, {'role': 'assistant', 'content': sample['rejected']}]
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'orca_dpo_pairs'
        })
elif args.input_dataset == 'argilla/distilabel-intel-orca-dpo-pairs':
    for sample in dataset:
        prompt = {'role': 'user', 'content': sample['input']}
        chosen = [prompt, {'role': 'assistant', 'content': sample['chosen']}]
        rejected = [prompt, {'role': 'assistant', 'content': sample['rejected']}]
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'orca_dpo_pairs_argilla'
        })
elif args.input_dataset == 'prm800k_train_phase1.jsonl':
    # Phase 1 of PRM: we sample reasoning paths from the set.
    # sometimes examples are malformed, so we skip them.
    for sample in dataset:
        # only collect samples that reached the solution
        if sample['label']['finish_reason'] != 'solution':
            continue
        prompt = {'role': 'user', 'content': sample['question']['problem']}
        ground_truth = sample['question']['ground_truth_answer']
        # start by gathering the full ground truth completion
        # along the way, we will gather 'wrong' reasoning paths.
        chosen_completions = []
        rejected_completions = []
        error = False
        # pick one random step to do incorrectly
        multichoice_steps = [idx for idx, step in enumerate(sample['label']['steps']) if len(step['completions']) > 1 or step['human_completion'] is not None]
        if len(multichoice_steps) == 0:
            # malformed, skip
            continue
        use_rejected = random_gen.choice(multichoice_steps)
        for idx, step in enumerate(sample['label']['steps']):
             # get chosen completion
            if step['chosen_completion'] is None:
                if step['human_completion'] is None:
                    # malformed, skip
                    error = True
                    print("Malformed sample, skipping")
                    break
                chosen_completion = step['human_completion']['text']
            else:
                chosen_completion = step['completions'][step['chosen_completion']]['text']
            chosen_completions.append(chosen_completion)
            non_chosen_completions = [x['text'] for x in step['completions'] if x['text'] != chosen_completion]
            # sometimes there is only one completion for a step. This is fine.
            if idx == use_rejected:
                if len(non_chosen_completions) == 0:
                    # malformed, skip
                    error = True
                    print("Malformed sample, skipping")
                    break
                rejected_completions.append(random_gen.choice(non_chosen_completions))
            else:
                rejected_completions.append(chosen_completion)
        if error:
            continue
        # now, we have the chosen and rejected completions
        chosen = [prompt, {'role': 'assistant', 'content': '\n'.join(chosen_completions)}]
        rejected = [prompt, {'role': 'assistant', 'content': '\n'.join(rejected_completions)}]
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'prm800k_phase1'
        })
elif args.input_dataset == 'prm800k_train_phase2.jsonl':
    # Phase 2 of PRM: multiple generations for the dataset, which are scored by humans.
    # first, we group by prompt
    prompts = {}
    for sample in dataset:
        prompt = sample['question']['problem']
        if prompt not in prompts:
            prompts[prompt] = []
        prompts[prompt].append(sample)
    # now sample our chosen and rejected completions
    # our chosen is the ground truth, and our rejected is a random sample that answers wrong
    # we ignore model-generated correct completions for now
    for prompt, samples in prompts.items():
        prompt = {'role': 'user', 'content': prompt}
        ground_truth = samples[0]['question']['ground_truth_solution']
        chosen = [prompt, {'role': 'assistant', 'content': ground_truth}]
        # gather model completions that are wrong
        wrong_completions = [x for x in samples if x['label']['finish_reason'] == 'found_error']
        if len(wrong_completions) == 0:
            continue
        sample = random_gen.choice(wrong_completions)
        model_generation = '\n'.join(sample['question']['pre_generated_steps'])
        rejected = [prompt, {'role': 'assistant', 'content': model_generation}]
        if chosen[1]['content'] is None or rejected[1]['content'] is None:
            continue
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'prm800k_phase2'
        })
elif args.input_dataset == "Anthropic/hh-rlhf":
    for sample in dataset:
        chosen_prompt_turns = parse_out_prompt_turns_hh_format(sample['chosen'])
        rejected_prompt_turns = parse_out_prompt_turns_hh_format(sample['rejected'])
        # run through the turns until they mismatch. This is our comparison point
        # sometimes the conversation keeps going on one, but just ignore that
        prompt_turns = []
        for i in range(min(len(chosen_prompt_turns), len(rejected_prompt_turns))):
            if chosen_prompt_turns[i] == rejected_prompt_turns[i]:
                prompt_turns.append(chosen_prompt_turns[i])
            else:
                break
        # malformed data
        if len(prompt_turns) >= len(rejected_prompt_turns):
            continue
        if len(prompt_turns) >= len(chosen_prompt_turns):
            continue
        final_chosen_turn = chosen_prompt_turns[len(prompt_turns)]
        final_rejected_turn = rejected_prompt_turns[len(prompt_turns)]
        new_data.append({
            'chosen': prompt_turns + [final_chosen_turn],
            'rejected': prompt_turns + [final_rejected_turn],
            'source': 'hh-rlhf'
        })
elif args.input_dataset == "lvwerra/stack-exchange-paired":
    for i, el in enumerate(dataset):
        prompt = {'content': el['question'], 'role': 'user'}
        chosen = {'content': el['response_j'], 'role': 'assistant'}
        rejected = {'content': el['response_k'], 'role': 'assistant'}
        data = {}
        data = {'chosen': [prompt, chosen], 'rejected': [prompt, rejected]}
        data['source'] = 'stack-exchange-paired'
        new_data.append(data)

# cleaning: make sure the content is always stripped
for sample in new_data:
    for msg in sample['chosen']:
        msg['content'] = msg['content'].strip()
    for msg in sample['rejected']:
        msg['content'] = msg['content'].strip()

# sanity checks over the data
# first: filter out empty content
def contains_empty(data):
    for msg in data['chosen']:
        if not msg['content']:
            return True
    for msg in data['rejected']:
        if not msg['content']:
            return True
    return False
# second: ends with assistant
def ends_with_assistant(data):
    return data['chosen'][-1]['role'] == 'assistant' and data['rejected'][-1]['role'] == 'assistant'

# apply the filters
print("Before filtering:", len(new_data))
new_data = [x for x in new_data if not contains_empty(x)]
new_data = [x for x in new_data if ends_with_assistant(x)]
print("After filtering:", len(new_data))

# save it
with open(args.output, 'w') as f:
    for sample in new_data:
        f.write(json.dumps(sample) + '\n')
