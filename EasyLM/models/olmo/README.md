# Instructions to download olmo model and intergrate to eazylm.

Download olmo checkpoints.
```shell
python EasyLM/scripts/download.py --repo_id allenai/OLMo-7B --from_safetensors True
```

Convert the olmo checkpoints
```shell
python  EasyLM/models/olmo/convert_hf_to_easylm.py \
    --checkpoint_dir='checkpoints/allenai/OLMo-7B' \
    --output_file='checkpoints/allenai/OLMo-7B-eazylm' 
```