"""
Usage:
python convert_hf_to_easylm.py  \
       --checkpoint_dir     /path/hf_format_dir/    \
       --output_file /path/easylm_format.stream   \
       --model_size 7b \
       --streaming
"""
import time
from pathlib import Path
import argparse

import mlxu
import torch
import flax
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from EasyLM.checkpoint import StreamingCheckpointer

LLAMA_STANDARD_CONFIGS = {
    "7b": {
        "dim": 4096,
        "intermediate_size": 11008,
        "n_layers": 2,
        "n_heads": 32,
        "norm_eps": 1e-5,
    },
}


def main(args):
    start = time.time()
    params = LLAMA_STANDARD_CONFIGS[args.model_size]

    ckpt_paths = sorted(Path(args.checkpoint_dir).glob("*.safetensors"))
    ckpt = {}
    
    for i in range(len(ckpt_paths)):
        with safe_open(ckpt_paths[i], framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.transformer."):
                    k = key[18:]
                else:
                    k = key
                ckpt[k] = f.get_tensor(key)    

    print(f"Start convert weight to easylm format...")
    
    jax_weights = {
        "transformer": {
            "wte": {"embedding": ckpt["wte.weight"].to(torch.float32).numpy()},
            "h": {
                "%d"
                % (layer): {
                    "attention": {
                        "wq": {
                            "kernel": ckpt[f"blocks.{layer}.att_proj.weight"].chunk(3)[0].to(torch.float32).numpy().transpose()
                        },
                        "wk": {
                            "kernel": ckpt[f"blocks.{layer}.att_proj.weight"].chunk(3)[1].to(torch.float32).numpy().transpose()
                            
                        },
                        "wv": {
                            "kernel": ckpt[f"blocks.{layer}.att_proj.weight"].chunk(3)[2]
                            .to(torch.float32)
                            .numpy()
                            .transpose()
                        },
                        "wo": {
                            "kernel": ckpt[f"blocks.{layer}.attn_out.weight"]
                            .to(torch.float32)
                            .numpy()
                            .transpose()
                        },
                    },
                    "feed_forward": {
                        "w1": {
                            "kernel": ckpt[f"blocks.{layer}.ff_proj.weight"].chunk(2)[0]
                            .to(torch.float32)
                            .numpy()
                            .transpose()
                        },
                        "w2": {
                            "kernel": ckpt[f"blocks.{layer}.ff_out.weight"]
                            .to(torch.float32)
                            .numpy()
                            .transpose()
                        },
                        "w3": {
                            "kernel": ckpt[f"blocks.{layer}.ff_proj.weight"].chunk(2)[1]
                            .to(torch.float32)
                            .numpy()
                            .transpose()
                        },
                    },
                }
                for layer in range(params["n_layers"])
            },
        },
        "lm_head": {"kernel": ckpt["ff_out.weight"].to(torch.float32).numpy().transpose()},
    }
    print(f"Convert weight to easylm format finished...")
    print(f"Start to save...")

    if args.streaming:
        StreamingCheckpointer.save_train_state_to_file(jax_weights, args.output_file)
    else:
        with mlxu.open_file(args.output_file, "wb") as fout:
            fout.write(flax.serialization.msgpack_serialize(jax_weights, in_place=True))

    print(
        f"Save finished!!! take time: {time.time() - start} save path: {args.output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hf to easylm format script")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Need to be converted model weight dir. it is a dir",
    )
    parser.add_argument(
        "--output_file", type=str, help="Save model weight file path, it is a file."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7b",
        choices=["7b", "13b", "30b", "65b", "70b"],
        help="model size",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="whether is model weight saved stream format",
    )

    args = parser.parse_args()

    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"output_file: {args.output_file}")
    print(f"model_size: {args.model_size}")
    print(f"streaming: {args.streaming}")

    main(args)
