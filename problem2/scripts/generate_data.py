#!/usr/bin/env python3
"""
Dataset generation for Sorting Detection Task

Task: Given a sequence, determine if it's sorted in ascending order.

Usage:
    python generate_data.py --seed 641
    python generate_data.py --seed 641 --output-dir ../data/problem2
"""

import os
import argparse
import json
import numpy as np


def generate_sorted_sequence(length, value_range):
    """Generate a sorted sequence."""
    # If length <= range size, use unique values (replace=False)
    # Otherwise, allow duplicates (replace=True)
    range_size = value_range[1] - value_range[0]
    replace = length > range_size

    values = np.random.choice(range(*value_range), size=length, replace=replace)
    values.sort()
    return values.tolist()


def generate_unsorted_sequence(length, value_range, max_attempts=100):
    """Generate an unsorted sequence by shuffling a sorted one."""
    sorted_seq = generate_sorted_sequence(length, value_range)

    # Shuffle until actually unsorted
    for _ in range(max_attempts):
        unsorted = sorted_seq.copy()
        np.random.shuffle(unsorted)
        # Check if actually unsorted (not same as sorted)
        if unsorted != sorted_seq:
            return unsorted

    # Fallback: swap first two elements
    unsorted = sorted_seq.copy()
    if len(unsorted) >= 2:
        unsorted[0], unsorted[1] = unsorted[1], unsorted[0]
    return unsorted


def generate_split(n_samples, length_range, value_range):
    """
    Generate dataset split with variable lengths.

    Args:
        n_samples: Number of samples to generate
        length_range: Tuple (min_len, max_len) inclusive
        value_range: Tuple (min_val, max_val) exclusive max

    Returns:
        List of samples with 'sequence', 'is_sorted', 'length' fields
    """
    samples = []
    n_sorted = n_samples // 2
    n_unsorted = n_samples - n_sorted

    # Generate sorted examples
    for _ in range(n_sorted):
        length = np.random.randint(length_range[0], length_range[1] + 1)
        sequence = generate_sorted_sequence(length, value_range)
        samples.append({
            'sequence': sequence,
            'is_sorted': 1,
            'length': length
        })

    # Generate unsorted examples
    for _ in range(n_unsorted):
        length = np.random.randint(length_range[0], length_range[1] + 1)
        sequence = generate_unsorted_sequence(length, value_range)
        samples.append({
            'sequence': sequence,
            'is_sorted': 0,
            'length': length
        })

    # Shuffle samples
    np.random.shuffle(samples)
    return samples


def generate_fixed_length_split(n_samples, length, value_range):
    """
    Generate dataset with fixed length (for extrapolation testing).

    Args:
        n_samples: Number of samples to generate
        length: Fixed sequence length
        value_range: Tuple (min_val, max_val) exclusive max

    Returns:
        List of samples
    """
    samples = []
    n_sorted = n_samples // 2
    n_unsorted = n_samples - n_sorted

    # Generate sorted examples
    for _ in range(n_sorted):
        sequence = generate_sorted_sequence(length, value_range)
        samples.append({
            'sequence': sequence,
            'is_sorted': 1,
            'length': length
        })

    # Generate unsorted examples
    for _ in range(n_unsorted):
        sequence = generate_unsorted_sequence(length, value_range)
        samples.append({
            'sequence': sequence,
            'is_sorted': 0,
            'length': length
        })

    # Shuffle samples
    np.random.shuffle(samples)
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Generate sorting detection dataset'
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
        default='../data/problem2',
        help='Output directory (default: ../data/problem2)'
    )
    parser.add_argument(
        '--train-len-min',
        type=int,
        default=8,
        help='Minimum training sequence length (default: 8)'
    )
    parser.add_argument(
        '--train-len-max',
        type=int,
        default=16,
        help='Maximum training sequence length (default: 16)'
    )
    parser.add_argument(
        '--test-lengths',
        type=int,
        nargs='+',
        default=[32, 64, 128, 256],
        help='Test sequence lengths for extrapolation (default: 32 64 128 256)'
    )
    parser.add_argument(
        '--value-min',
        type=int,
        default=0,
        help='Minimum value in sequences (default: 0)'
    )
    parser.add_argument(
        '--value-max',
        type=int,
        default=100,
        help='Maximum value in sequences (exclusive, default: 100)'
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
        '--n-test-per-length',
        type=int,
        default=500,
        help='Number of test samples per length (default: 500)'
    )

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    print("=" * 60)
    print("Sorting Detection Dataset Generation")
    print("=" * 60)
    print(f"Value range: [{args.value_min}, {args.value_max})")
    print(f"Training lengths: {args.train_len_min}-{args.train_len_max}")
    print(f"Test lengths: {args.test_lengths}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    value_range = (args.value_min, args.value_max)
    length_range = (args.train_len_min, args.train_len_max)

    # Generate training set (variable lengths)
    print(f"Generating {args.n_train} training samples (lengths {args.train_len_min}-{args.train_len_max})...")
    train_samples = generate_split(args.n_train, length_range, value_range)
    output_file = os.path.join(args.output_dir, 'train.jsonl')
    with open(output_file, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"  ✓ Saved {len(train_samples)} samples to train.jsonl")

    # Generate validation set (variable lengths)
    print(f"Generating {args.n_val} validation samples (lengths {args.train_len_min}-{args.train_len_max})...")
    val_samples = generate_split(args.n_val, length_range, value_range)
    output_file = os.path.join(args.output_dir, 'val.jsonl')
    with open(output_file, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"  ✓ Saved {len(val_samples)} samples to val.jsonl")

    # Generate test sets at different lengths (for extrapolation)
    for test_len in args.test_lengths:
        print(f"Generating {args.n_test_per_length} test samples (length {test_len})...")
        test_samples = generate_fixed_length_split(args.n_test_per_length, test_len, value_range)
        output_file = os.path.join(args.output_dir, f'test_len_{test_len}.jsonl')
        with open(output_file, 'w') as f:
            for sample in test_samples:
                f.write(json.dumps(sample) + '\n')
        print(f"  ✓ Saved {len(test_samples)} samples to test_len_{test_len}.jsonl")

    # Save metadata
    metadata = {
        'task': 'sorting_detection',
        'value_range': [args.value_min, args.value_max],
        'train_len_range': [args.train_len_min, args.train_len_max],
        'test_lengths': args.test_lengths,
        'n_train': len(train_samples),
        'n_val': len(val_samples),
        'n_test_per_length': args.n_test_per_length,
        'seed': args.seed
    }

    metadata_file = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to metadata.json")

    print()
    print("=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"Training: {len(train_samples)} samples (lengths {args.train_len_min}-{args.train_len_max})")
    print(f"Validation: {len(val_samples)} samples (lengths {args.train_len_min}-{args.train_len_max})")
    print(f"Test: {args.n_test_per_length} samples each for lengths {args.test_lengths}")
    print(f"Location: {os.path.abspath(args.output_dir)}")
    print()


if __name__ == '__main__':
    main()
