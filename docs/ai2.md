# Running EasyLM (Hamish Ver.) At AI2

Here's a crash course on running this library. I'll focus on users at AI2, who are currently the primary consumers of this fork. This is a beta of this document, as things may change over time.

## Setup

You'll need a working ai2 account, and be able to login to the [google cloud console](console.cloud.google.com). Under your team project, create a bucket (in cloud storage). Make sure it is in the us region, **MAKE SURE IT IS EAST1-D REGION** (to avoid network fees). Then give `972660293965-compute@developer.gserviceaccount.com` read and write permissions to the bucket. This is the TPU account, and will let the TPU read and write from the bucket.

Join the AI2-TPU google group and you should automatically get added to the Ai2 TPU google cloud project (also viewable through the console). This is the project we will make TPUs under, but keep the bucket under your team's project (for accounting purposes).

Download and install the [gcloud CLI](https://cloud.google.com/sdk/gcloud) on your developer device of choice (e.g. your laptop). This will be the primary way we interact with the TPUs. `gcloud auth login` should work for authenticating.

That's it! Now we can setup TPUs.

## Running your first TPU

I try to stick to the command line. Once authenticated, you can make a TPU VM with:
```bash
gcloud alpha compute tpus tpu-vm create <name> --accelerator-type=v3-8 --zone=us-central1-a --project=ai2-tpu --version=v2-alpha
```
Or, if you want a bigger TPU:
```bash
gcloud alpha compute tpus tpu-vm create <name> --accelerator-type=v3-256 --zone=us-east1-d --project=ai2-tpu --version=v2-alpha
```

Note the different `zone`s, this is because of how our quota works. I usually name my tpus something like `hamishi-v3-8` (i.e., username-tpu-size). We only have access to v3s and v2s right now, and note that the larger the TPU you want, the higher the likelihood is you can't get it immediately and might have a try a few times (ask me about this). Sometimes there are simply no TPUs available in the research cloud.

Now, the way you interact with the TPUs is going to be different depending on the size:

For a v3-8, you can just ssh directly into the TPU with `gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-central1-a --project=ai2-tpu`. Then just treat it like your development machine: clone repositories, run code, whatever! This is useful for debugging on a TPU.

For anything larger, the TPU is actually a collection of machines (1 per 8 cores). Jax can handle the coordination between the machines... so long as the environment between them is identical. So in this case, we can send single ssh commands to all machines like so: `gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-east1-d --project=ai2-tpu --worker=all --command="<command>"`
For this reason, it's useful to have scripts that do what you want to avoid having to type lots of commands.

You can quickly and easily test everything is working by running a little Jax program:
```bash
gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-east1-d --project=ai2-tpu --worker=all --command="python3 -c \"import jax; jax.devices()\""
```

Anyway, now we are set up! If you need to delete a TPU, you can use the `gcloud alpha compute tpus tpu-vm delete` command.

**NOTE: It is important to note that for all the commands below, you should be running them on all workers with `gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-east1-d --project=ai2-tpu --worker=all --command=` if you are using a TPU pod (anything over 8 cores). Yes, this means running the git clone command, setup script, and train command on multiple machines.**

### Using Someone Else's TPU

Sometimes it's useful to jump on someone else's TPU instead of yours (for example, if they are not using it). Only one person can use a TPU at a time, so please coordinate with the person as to who is using it. Sometimes the previous user may have left lockfiles that cause permission errors. You can fix these by removing the lockfiles:
```bash
sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs
```

And again, you should be running this via `gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-east1-d --project=ai2-tpu --worker=all --command=` if you are using a TPU pod.

## Running EasyLM

I'll omit the TPU ssh command bits here, but remember to use them for a TPU pod!

```bash
git clone https://github.com/hamishivi/easylm.git  # and swap to whatever branch you want
cd easylm; ./scripts/tpu_vm_setup.sh  # setup environment
wandb login <key>  # for logging into wandb
gsutil -m cp gs://hamishi-east1/easylm/data/tulu-v2-sft-mixture.jsonl .  # download whatever data you want. TODO: I need to allow data to be streamed...
```

And that's it! Then you can go nuts and train. Here's an example *finetuning* command:
```bash
cd easylm; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,8' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=50 \
    --save_model_freq=1000 \
    --save_milestone_freq=0 \
    --load_llama_config='7b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/7b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=2e-5 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='tulu_json_torch' \
    --train_dataset.json_torch_dataset.hf_name='allenai/tulu-v2-sft-mixture' \
    --train_dataset.json_torch_dataset.seq_length=8192 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.entity='rlhf-llm-dev' --logger.project='jax_sft' \
    --logger.output_dir="gs://OUTPUT_DIR" &> all.log &
```

Note a few things:
- most things load directly from the google bucket! no need to download stuff! The exception is downloading the SFT data using `gsutil cp gs://hamishi-east1/easylm/data/tulu-v2-sft-mixture.jsonl .`
- there's a bunch of scary random TPU args, these are just args I found that people recommended. I haven't properly tested them...
- the `mesh_dim` defines the parallelism strategy. Check out the EasyLM parallelism doc for more information. Generally, you want the biggest FSDP parallelism (middle number), and smallest model parallelism possible (last number). The numbers must multiply to the TPU size (e.g. 256 for v3-256).
- Currently I am lazy and just download the datafile to the TPU
- Edit the logger to point to whatever wandb project you want. Entity is the group I've been putting this stuff in.
- `output_dir` is where the model gets saved
- `&> all.log &` is there so the output gets streamed to `all.log`, and then the process runs in the background, to avoid needing constant ssh connection. There may be a less hacky way to do this. You can inspect the output with `tail -f all.log`.

Generally, I recommend waiting and checking for the model to start logging steps, and then you are probably fine.

Here's an example dpo train script, and note that it is basically the same:
```bash
cd easylm; git pull; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train_dpo \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=3 \
    --log_freq=50 \
    --save_model_freq=1000 \
    --save_milestone_freq=0 \
    --load_llama_config='7b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/tulu2_7b_fixed/263f4f758b194729b206d5adad2b50d7/streaming_params_20384' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-7 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.1 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='preference_json_torch' \
    --train_dataset.json_torch_dataset.hf_name='allenai/ultrafeedback_binarized_cleaned' \
    --train_dataset.json_torch_dataset.hf_split='train_prefs' \
    --train_dataset.json_torch_dataset.seq_length=6144 \
    --train_dataset.json_torch_dataset.batch_size=8 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='easylm_dpo' --logger.entity='rlhf-llm-dev' \
    --logger.output_dir="gs://OUTPUT_DIR" &> all.log &"
```

Again, get the dataset using `gsutil cp gs://hamishi-east1/easylm/data/ultra_feedback_tulu.jsonl .`
I have arguments for the beta, etc, but these are not used here. Other example scripts live in the `examples/` directory.

## Killing a job

I usually use `pgrep llama | xargs kill -9` to ensure everything is dead.

## Exporting to HF

You need to run the rest of the steps on a different machine. I use a cirrascale machine (without GPUs), but theoretically you could run it all locally. You can install the dependencies with `conda env create -f scripts/gpu_environment.yaml`.

Once a run is done, the model will live in your google bucket. Identify the final checkpoint (`streaming_param_<some number>`, it'll be the largest number - it saves every epoch), and then convert with:
```bash
python -m EasyLM.models.llama.convert_easylm_to_hf --load_checkpoint=params::<path> --tokenizer_path='gs://hamishi-east1/easylm/llama/tokenizer.model' --model_size=<model_size> --output_dir=<output_dir>
```
I recommend running this on a cirrascale machine and then uploading the converted model to beaker to avoid using up NFS space. After this, you can evaluate using normal pytorch code.

Once this is all done, you can swap over to the `open-instruct` repository and use the `submit_eval_jobs.py` script to evaluate the model.

## Exporting & Evaluating in (almost) one click!

I've added a little script for exporting and evaluating the model all in one. Maybe at some point I'll try to fold this into training too. To use, first make sure you have a beaker workspace, are authenticated with beaker, and have a beaker secret called `OPENAI_API_KEY` in your workspace. Then, simply install this repo (one-time thing) to a machine with enough space to download the model you want to run. Then you can use the following script:
```bash
./scripts/convert_and_submit_tuned MODEL_PATH MODEL_SIZE MODEL_NAME WORKSPACE
```

You'll need to fill in the four variables above with the requisite bits. Once done, you can go to beaker to check on the status of your evaluations.

*Note: this script assumes a llama tokenizer, which isn't true for anything that isnt llama 1/2. For those, you'll have to do the conversion + evaluation steps more manually (but trust me, it's easy work!)*.
