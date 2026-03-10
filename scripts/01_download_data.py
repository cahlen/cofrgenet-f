"""Download and tokenize FineWeb-Edu 10BT into binary shards.

Downloads the HuggingFace dataset in streaming mode, tokenizes with GPT-2
tokenizer (tiktoken), and writes shards as np.uint16 binary files.

Output:
    data/tokenized/train_000.bin, train_001.bin, ...
    data/tokenized/val_000.bin

Usage:
    python scripts/01_download_data.py [--shard_size 100000000] [--val_fraction 0.01]
"""

import argparse
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Download and tokenize FineWeb-Edu 10BT")
    parser.add_argument("--shard_size", type=int, default=100_000_000,
                        help="Tokens per shard (default: 100M)")
    parser.add_argument("--val_fraction", type=float, default=0.01,
                        help="Fraction of tokens for validation (default: 0.01)")
    default_output = "data/tokenized"
    if os.path.isdir("/raid"):
        default_output = "/raid/cofrgenet-f/data/tokenized"
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Output directory for tokenized shards")
    parser.add_argument("--dataset_name", type=str, default="sample-10BT",
                        choices=["sample-10BT", "sample-100BT", "sample-350BT"],
                        help="FineWeb-Edu subset to download")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Stop after this many tokens (e.g., 50000000000 for 50B)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    print(f"Loading FineWeb-Edu {args.dataset_name} (streaming)...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.dataset_name,
                      split="train", streaming=True)

    # Tokenize and write shards
    shard_idx = 0
    token_buf = np.empty(args.shard_size, dtype=np.uint16)
    buf_pos = 0
    total_tokens = 0
    val_tokens = []
    val_done = False

    # We'll collect ~1% of tokens for validation from the start,
    # then switch to writing train shards.
    # Estimate: 10B tokens total, 1% = 100M tokens = 1 shard for val.
    val_target = int(args.shard_size * 1)  # 1 shard worth of val data

    print(f"Tokenizing... (shard size: {args.shard_size:,} tokens)")
    pbar = tqdm(ds, unit=" docs")

    for doc in pbar:
        if args.max_tokens and total_tokens >= args.max_tokens:
            print(f"\nReached target of {args.max_tokens:,} tokens, stopping.")
            break
        text = doc["text"]
        tokens = enc.encode_ordinary(text)
        tokens.append(eot)
        token_arr = np.array(tokens, dtype=np.uint16)

        if not val_done:
            val_tokens.append(token_arr)
            val_total = sum(len(t) for t in val_tokens)
            if val_total >= val_target:
                # Write val shard
                val_data = np.concatenate(val_tokens)[:val_target]
                val_path = os.path.join(args.output_dir, "val_000.bin")
                val_data.tofile(val_path)
                print(f"\nWrote {val_path}: {len(val_data):,} tokens")
                total_tokens += len(val_data)
                val_done = True
                val_tokens = None
                continue

        # Write to train shard buffer
        remaining = len(token_arr)
        src_pos = 0
        while remaining > 0:
            space = args.shard_size - buf_pos
            n = min(remaining, space)
            token_buf[buf_pos:buf_pos + n] = token_arr[src_pos:src_pos + n]
            buf_pos += n
            src_pos += n
            remaining -= n

            if buf_pos >= args.shard_size:
                # Flush shard
                shard_path = os.path.join(args.output_dir, f"train_{shard_idx:03d}.bin")
                token_buf.tofile(shard_path)
                total_tokens += args.shard_size
                pbar.set_postfix(shard=shard_idx, total=f"{total_tokens/1e9:.2f}B")
                shard_idx += 1
                buf_pos = 0

    # Write final partial shard if any
    if buf_pos > 0:
        shard_path = os.path.join(args.output_dir, f"train_{shard_idx:03d}.bin")
        token_buf[:buf_pos].tofile(shard_path)
        total_tokens += buf_pos
        shard_idx += 1

    print(f"\nDone! {total_tokens:,} total tokens in {shard_idx} train shards + 1 val shard")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
