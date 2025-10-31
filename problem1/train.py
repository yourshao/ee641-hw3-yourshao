"""
Training script for sequence-to-sequence addition model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import time

from model import Seq2SeqTransformer
from dataset import create_dataloaders, get_vocab_size
from attention import create_causal_mask


def compute_accuracy(outputs, targets, pad_token=0):
    """
    Compute sequence-level accuracy.

    Args:
        outputs: Model predictions [batch, seq_len, vocab_size]
        targets: Ground truth [batch, seq_len]
        pad_token: Padding token to ignore

    Returns:
        Accuracy (fraction of completely correct sequences)
    """
    # preds = outputs.argmax(dim=-1)  # [B, T]
    #
    # mask = (targets != pad_token).long()
    #
    # correct_per_pos = (preds == targets) | (mask == 0)
    # correct_seq = correct_per_pos.all(dim=1).float()  # [B]
    # return correct_seq.mean().item()

    preds = outputs.argmax(dim=-1)
    correct_seq = (preds == targets).all(dim=1).float()
    return correct_seq.mean().item()


def _make_masks(inputs, dec_inp, pad_token=0, device=None):

    device = device or inputs.device
    # src padding mask: [B, 1, 1, S]
    # src_pad = (inputs != pad_token).unsqueeze(1).unsqueeze(1).to(inputs.dtype)
    # tgt_pad = (dec_inp != pad_token).unsqueeze(1).unsqueeze(1).to(inputs.dtype)
    # causal = create_causal_mask(dec_inp.size(1), device=device)  # {0,1}

    # tgt_mask = (tgt_pad * causal).to(dec_inp.dtype)
    # src_mask = src_pad

    src_mask = None
    tgt_mask = create_causal_mask(dec_inp.size(1), device=device)

    return src_mask, tgt_mask


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        # Move to device
        inputs = batch['input'].to(device)     # [B, S]
        targets = batch['target'].to(device)   # [B, T]

        dec_inp = targets[:, :-1]              # [B, T-1]
        dec_out = targets[:, 1:]               # [B, T-1]

        # Masks
        src_mask, tgt_mask = _make_masks(inputs, dec_inp, pad_token=0, device=device)

        # Forward
        logits = model(inputs, dec_inp, src_mask=src_mask, tgt_mask=tgt_mask)  # [B, T-1, V]

        B, Tm1, V = logits.size()
        loss = criterion(logits.reshape(B * Tm1, V), dec_out.reshape(B * Tm1))

        # Backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc = compute_accuracy(logits, dec_out, pad_token=0)

        # Progress
        progress.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2%}'})

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            dec_inp = targets[:, :-1]
            dec_out = targets[:, 1:]

            src_mask, tgt_mask = _make_masks(inputs, dec_inp, pad_token=0, device=device)

            logits = model(inputs, dec_inp, src_mask=src_mask, tgt_mask=tgt_mask)

            B, Tm1, V = logits.size()
            loss = criterion(logits.reshape(B * Tm1, V), dec_out.reshape(B * Tm1))

            acc = compute_accuracy(logits, dec_out, pad_token=0)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train addition transformer')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of encoder/decoder layers')
    parser.add_argument('--d-ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size
    )

    # Model
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    ).to(args.device)

    # Optimizer / Scheduler / Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = -1.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, args.device
        )

        # Scheduler step uses validation loss
        scheduler.step(val_loss)

        # Log
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        print(f"Val  Loss: {val_loss:.4f}, Val  Acc: {val_acc:.2%}")

        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.2%}")

    # Test best
    model.load_state_dict(torch.load(output_dir / 'best_model.pth', map_location=args.device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")

    # Save history
    training_history['test_loss'] = test_loss
    training_history['test_acc'] = test_acc
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
