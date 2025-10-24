#!/usr/bin/env python3
"""
Dataset generation: Multi-Digit Addition Task

Task: Add two numbers digit-by-digit with carry propagation.

Example:
    Input:  [3, 4, 7, +, 1, 5, 8]  (347 + 158)
    Output: [5, 0, 5]                (505)

Challenges:
- Requires carry propagation (sequential dependency)
- Must handle variable-magnitude carries
- Tests long-range dependencies (carry from rightmost to leftmost)

Fixed-length design:
- Input: [digit, digit, ..., +, digit, digit, ...]
- Output: Result padded to fixed length
- All sequences same length to avoid variable-length issues

Usage:
    python generate_data.py --seed 641
    python generate_data.py --seed 641 --output-dir ../data/problem1
"""

import os
import argparse
import json
import numpy as np


def number_to_digits(num, num_digits):
    """
    Convert number to list of digits (left-to-right, most significant first).

    Args:
        num: Integer to convert
        num_digits: Number of digits to pad to

    Returns:
        List of digits [most_significant, ..., least_significant]
    """
    digits = []
    for _ in range(num_digits):
        digits.append(num % 10)
        num //= 10
    return digits[::-1]  # Reverse to get most significant first


def generate_samples(n_samples, num_digits, vocab_size=12, seen=None):
    """
    Generate addition samples.

    Args:
        n_samples: Number of samples to generate
        num_digits: Number of digits per operand (e.g., 3 for 3-digit numbers)
        vocab_size: Size of vocabulary (0-9 digits + special tokens)
        seen: Set of already seen samples (for uniqueness)

    Returns:
        List of samples with 'input' and 'target' fields
    """
    if seen is None:
        seen = set()

    # Token mapping:
    # 0-9: digits
    # 10: '+' operator
    # 11: padding (if needed)
    PLUS_TOKEN = 10
    PAD_TOKEN = 11

    samples = []
    attempts = 0
    max_attempts = n_samples * 10

    # Maximum values for num_digits (e.g., 3 digits -> max 999)
    max_val = 10 ** num_digits - 1

    # Output can be at most num_digits + 1 (e.g., 999 + 999 = 1998, 4 digits)
    output_len = num_digits + 1

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        # Generate two random numbers with num_digits digits
        num1 = np.random.randint(0, max_val + 1)
        num2 = np.random.randint(0, max_val + 1)

        # Compute sum
        result = num1 + num2

        # Create unique key (sorted pair to avoid duplicates like 5+3 and 3+5)
        key = (min(num1, num2), max(num1, num2))

        if key not in seen:
            seen.add(key)

            # Convert to digit sequences
            num1_digits = number_to_digits(num1, num_digits)
            num2_digits = number_to_digits(num2, num_digits)
            result_digits = number_to_digits(result, output_len)

            # Create input: [num1_digits, +, num2_digits]
            input_seq = num1_digits + [PLUS_TOKEN] + num2_digits

            # Create output: [result_digits] (padded to output_len)
            output_seq = result_digits

            samples.append({
                'input': input_seq,
                'target': output_seq,
                'num1': num1,
                'num2': num2,
                'result': result
            })

    if len(samples) < n_samples:
        print(f"Warning: Could only generate {len(samples)}/{n_samples} unique samples")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-digit addition dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=641,
        help='Random seed for reproducibility (default: 641)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/problem1',
        help='Output directory (default: ../data/problem1)'
    )
    parser.add_argument(
        '--num-digits',
        type=int,
        default=3,
        help='Number of digits per operand (default: 3, i.e., 0-999)'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=10000,
        help='Number of training samples (default: 10000)'
    )
    parser.add_argument(
        '--n-val',
        type=int,
        default=2000,
        help='Number of validation samples (default: 2000)'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=2000,
        help='Number of test samples (default: 2000)'
    )

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Calculate input/output lengths
    input_len = args.num_digits * 2 + 1  # num1 + '+' + num2
    output_len = args.num_digits + 1     # result (can have one more digit for carry)
    vocab_size = 12  # 0-9 digits + '+' token + pad token

    print("=" * 60)
    print("Multi-Digit Addition Dataset Generation")
    print("=" * 60)
    print(f"Number of digits per operand: {args.num_digits}")
    print(f"  Range: 0 to {10**args.num_digits - 1}")
    print(f"  Example: 347 + 158 = 505")
    print(f"Input sequence length: {input_len}")
    print(f"  Format: [d, d, d, +, d, d, d] (for 3 digits)")
    print(f"Output sequence length: {output_len}")
    print(f"  Format: [d, d, d, d] (for 3-digit operands)")
    print(f"Vocabulary size: {vocab_size}")
    print(f"  0-9: digits, 10: '+', 11: pad")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Track seen samples for uniqueness across splits
    seen = set()

    # Generate training set
    print(f"Generating {args.n_train} training samples...")
    train_samples = generate_samples(args.n_train, args.num_digits, vocab_size, seen)

    # Generate validation set
    print(f"Generating {args.n_val} validation samples...")
    val_samples = generate_samples(args.n_val, args.num_digits, vocab_size, seen)

    # Generate test set
    print(f"Generating {args.n_test} test samples...")
    test_samples = generate_samples(args.n_test, args.num_digits, vocab_size, seen)

    # Save as JSONL
    for split_name, samples in [('train', train_samples),
                                 ('val', val_samples),
                                 ('test', test_samples)]:
        output_file = os.path.join(args.output_dir, f'{split_name}.jsonl')
        with open(output_file, 'w') as f:
            for sample in samples:
                # Save only input/target for training (keep num1/num2/result for debugging)
                save_sample = {
                    'input': sample['input'],
                    'target': sample['target']
                }
                f.write(json.dumps(save_sample) + '\n')
        print(f"  ✓ Saved {len(samples)} samples to {split_name}.jsonl")

    # Save metadata
    metadata = {
        'num_digits': args.num_digits,
        'input_len': input_len,
        'output_len': output_len,
        'vocab_size': vocab_size,
        'token_mapping': {
            '0-9': 'digits',
            '10': 'plus_operator',
            '11': 'padding'
        },
        'n_train': len(train_samples),
        'n_val': len(val_samples),
        'n_test': len(test_samples),
        'seed': args.seed
    }

    metadata_file = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to metadata.json")

    # Show some example samples
    print("\nExample samples:")
    for i in range(min(5, len(train_samples))):
        sample = train_samples[i]
        input_str = ''.join([str(d) if d < 10 else '+' for d in sample['input']])
        output_str = ''.join([str(d) for d in sample['target']])
        print(f"  {i+1}. {sample['num1']:3d} + {sample['num2']:3d} = {sample['result']:4d}")
        print(f"     Input:  {sample['input']} ({input_str})")
        print(f"     Target: {sample['target']} ({output_str})")

    print()
    print("=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"Total samples: {len(train_samples) + len(val_samples) + len(test_samples)}")
    print(f"Location: {os.path.abspath(args.output_dir)}")
    print()


if __name__ == '__main__':
    main()
