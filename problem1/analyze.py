"""
Analysis and visualization of attention patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from model import Seq2SeqTransformer
from dataset import create_dataloaders, get_vocab_size
from attention import create_causal_mask
from collections import defaultdict
from pathlib import Path


PAD = -1
OPERATOR_TOKEN_ID = 10


def _make_masks(inputs, dec_inp, pad_token=PAD, device=None):
    device = device or inputs.device
    src_pad = (inputs != pad_token).unsqueeze(1).unsqueeze(1).to(inputs.dtype)
    tgt_pad = (dec_inp != pad_token).unsqueeze(1).unsqueeze(1).to(inputs.dtype)
    causal = create_causal_mask(dec_inp.size(1), device=device)
    tgt_mask = (tgt_pad * causal).to(dec_inp.dtype)
    src_mask = src_pad
    return src_mask, tgt_mask


def extract_attention_weights(model, dataloader, device, num_samples=100):
    """
    Extract attention weights from model for analysis.
    """
    model.eval()

    all_encoder_attentions = []         # list of [H, Lq, Lk] per layer per sample
    all_decoder_self_attentions = []
    all_decoder_cross_attentions = []
    all_inputs = []
    all_targets = []

    samples_collected = 0

    # hook 容器（每个 batch 临时使用，然后清空）
    encoder_attn_batch = []
    decoder_self_attn_batch = []
    decoder_cross_attn_batch = []

    encoder_layer_tags, decoder_self_layer_tags, decoder_cross_layer_tags = [], [], []

    # 工具：把 [B,H,Lq,Lk] 拆成若干 [H,Lq,Lk]，只取前 N 个样本
    # def _unbatch_and_take(t, take_n):
    #     t = t[:take_n]  # [take,H,Lq,Lk]
    #     arr = [t[i].cpu() for i in range(t.shape[0])]
    #     return arr
    def _unbatch_and_take(t, take_n):
        t = t[:take_n]                   # [take_n, H, Lq, Lk]
        return [t[i].cpu() for i in range(t.shape[0])]  # List[[H,Lq,Lk]]


    def make_hook(acc_list, layer_index):
        def hook(module, inputs, outputs):
            # outputs: (attn_out, attn_weights); weights: [B,H,Lq,Lk]
            attn_weights = outputs[1].detach()
            acc_list.append((layer_index, attn_weights))
        return hook

    #  hooks
    encoder_hooks = []
    for li, layer in enumerate(model.encoder_layers):
        h = layer.self_attn.register_forward_hook(make_hook(encoder_attn_batch, li))
        encoder_hooks.append(h)

    decoder_hooks = []
    for li, layer in enumerate(model.decoder_layers):
        h1 = layer.self_attn.register_forward_hook(make_hook(decoder_self_attn_batch, li))
        h2 = layer.cross_attn.register_forward_hook(make_hook(decoder_cross_attn_batch, li))
        decoder_hooks.extend([h1, h2])

    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break

            inputs = batch['input'].to(device)   # [B,S]
            targets = batch['target'].to(device) # [B,T]
            batch_size = inputs.size(0)

            # teacher forcing 以触发所有注意力层
            # dec_inp = targets[:, :-1]
            # # dec_out = targets[:, 1:]  # 不需要监督，仅为forward走通  -1  -1  -1
            # src_mask, tgt_mask = _make_masks(inputs, dec_inp, pad_token=PAD, device=device)
            dec_inp = torch.cat([targets[:, :1], targets[:, :-1]], dim=1)  # [B, T]
            src_mask, tgt_mask = _make_masks(inputs, dec_inp, pad_token=PAD, device=device)
            # 清空本 batch 收集容器
            encoder_attn_batch.clear()
            decoder_self_attn_batch.clear()
            decoder_cross_attn_batch.clear()

            # 前向
            # _ = model(inputs, dec_inp, src_mask=src_mask, tgt_mask=tgt_mask)
            _ = model(inputs, dec_inp, src_mask=src_mask, tgt_mask=tgt_mask)
            # 取本 batch 中需要的样本数量
            take_n = min(batch_size, num_samples - samples_collected)

            # 整理 encoder attn：按层的顺序排列
            # encoder_attn_batch: list len = num_encoder_layers;
            # for layer_attn in encoder_attn_batch:
            #     all_encoder_attentions.extend(_unbatch_and_take(layer_attn, take_n))
            #
            # # decoder self
            # for layer_attn in decoder_self_attn_batch:
            #     all_decoder_self_attentions.extend(_unbatch_and_take(layer_attn, take_n))
            #
            # # decoder cross
            # for layer_attn in decoder_cross_attn_batch:
            #     all_decoder_cross_attentions.extend(_unbatch_and_take(layer_attn, take_n))

            # —— encoder
            for li, layer_attn in encoder_attn_batch:      # 现在是 (li, tensor)
                chunk = _unbatch_and_take(layer_attn, take_n)  # tensor -> List[[H,Lq,Lk]]
                all_encoder_attentions.extend(chunk)
                encoder_layer_tags.extend([li] * len(chunk))

            # —— decoder self
            for li, layer_attn in decoder_self_attn_batch:
                chunk = _unbatch_and_take(layer_attn, take_n)
                all_decoder_self_attentions.extend(chunk)
                decoder_self_layer_tags.extend([li] * len(chunk))

            # —— decoder cross
            for li, layer_attn in decoder_cross_attn_batch:
                chunk = _unbatch_and_take(layer_attn, take_n)
                all_decoder_cross_attentions.extend(chunk)
                decoder_cross_layer_tags.extend([li] * len(chunk))




            all_inputs.extend(inputs[:take_n].cpu().numpy())
            all_targets.extend(targets[:take_n].cpu().numpy())

            samples_collected += take_n


    for h in encoder_hooks + decoder_hooks:
        h.remove()

    return {
        'encoder_attention': all_encoder_attentions,               # list of [H,Lq,Lk]
        'decoder_self_attention': all_decoder_self_attentions,     # list of [H,Lq,Lk]
        'decoder_cross_attention': all_decoder_cross_attentions,   # list of [H,Lq,Lk]

        'encoder_layer_tags': encoder_layer_tags,
        'decoder_self_layer_tags': decoder_self_layer_tags,
        'decoder_cross_layer_tags': decoder_cross_layer_tags,

        'inputs': all_inputs,
        'targets': all_targets
    }


def visualize_attention_pattern(attention_weights, input_tokens, output_tokens,
                                title="Attention Pattern", save_path=None):
    """
    Visualize attention weights as heatmap.
    """
    num_heads = attention_weights.shape[0]

    # Create figure with subplots for each head
    rows = 2
    cols = (num_heads + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 8))
    axes = np.array(axes).reshape(-1)  # flatten

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        sns.heatmap(
            attention_weights[head_idx],
            ax=ax,
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            vmin=0,
            vmax=1
        )
        ax.set_title(f'Head {head_idx + 1}')
        ax.set_xlabel('Input Position')
        ax.set_ylabel('Output Position')

    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()


def _attention_entropy(weights, eps=1e-12):
    # weights: [Lq, Lk]
    p = weights.clamp(min=eps)
    ent = -(p * p.log()).sum(dim=-1).mean()  # 平均行熵
    return ent.item()


def analyze_head_specialization(attention_data, output_dir):
    """
    Analyze what each attention head specializes in.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Analyzing encoder self-attention patterns...")

    enc_list = attention_data['encoder_attention']  # list of [H,L,L]
    inputs = attention_data['inputs']               # list of [S]

    head_stats = {
        'encoder': {
            'avg_diag': [],
            'avg_offdiag_pm1': [],
            'avg_to_operator': [],
            'entropy': []
        }
    }


    if len(enc_list) == 0:
        with open(output_dir / 'head_analysis.json', 'w') as f:
            json.dump(head_stats, f, indent=2)
        return head_stats


    H = enc_list[0].shape[0]

    # 累加器
    diag_sum = np.zeros(H, dtype=np.float64)
    pm1_sum = np.zeros(H, dtype=np.float64)
    op_sum = np.zeros(H, dtype=np.float64)
    ent_sum = np.zeros(H, dtype=np.float64)
    count = 0


    for idx, attn in enumerate(enc_list):

        tokens = torch.tensor(inputs[idx % len(inputs)])
        L = attn.shape[1]

        tokens = tokens[:L]


        attn_t = attn
        H_, Lq, Lk = attn_t.shape
        assert H_ == H

        diag_mask = torch.eye(Lk, dtype=attn_t.dtype).unsqueeze(0)
        pm1_mask = torch.zeros(Lk, Lk, dtype=attn_t.dtype)
        pm1_mask += torch.diag(torch.ones(Lk - 1), diagonal=1)
        pm1_mask += torch.diag(torch.ones(Lk - 1), diagonal=-1)
        pm1_mask = pm1_mask.unsqueeze(0)

        diag_val = (attn_t * diag_mask).sum(dim=(-1, -2)) / Lk
        pm1_val = (attn_t * pm1_mask).sum(dim=(-1, -2)) / (2 * (Lk - 1) if Lk > 1 else 1)

        op_cols = (tokens == OPERATOR_TOKEN_ID).nonzero(as_tuple=False).flatten()
        if op_cols.numel() > 0:
            col_mask = torch.zeros(Lk, dtype=attn_t.dtype)
            col_mask[op_cols] = 1.0
            col_mask = col_mask.unsqueeze(0).unsqueeze(1)

            col_mask = col_mask.expand(H, Lq, Lk)
            op_val = (attn_t * col_mask).sum(dim=(-1, -2)) / op_cols.numel()
        else:
            op_val = torch.zeros(H, dtype=attn_t.dtype)


        ent = torch.tensor([_attention_entropy(attn_t[h]) for h in range(H)], dtype=attn_t.dtype)


        diag_sum += diag_val.numpy()
        pm1_sum += pm1_val.numpy()
        op_sum += op_val.numpy()
        ent_sum += ent.numpy()
        count += 1

    head_stats['encoder']['avg_diag'] = (diag_sum / max(count, 1)).tolist()
    head_stats['encoder']['avg_offdiag_pm1'] = (pm1_sum / max(count, 1)).tolist()
    head_stats['encoder']['avg_to_operator'] = (op_sum / max(count, 1)).tolist()
    head_stats['encoder']['entropy'] = (ent_sum / max(count, 1)).tolist()

    with open(output_dir / 'head_analysis.json', 'w') as f:
        json.dump(head_stats, f, indent=2)

    return head_stats


def _zero_head_slices(mha_module, head_idx):

    d_model = mha_module.d_model
    H = mha_module.num_heads
    d_k = mha_module.d_k
    s = slice(head_idx * d_k, (head_idx + 1) * d_k)

    W_q, W_k, W_v = mha_module.W_q, mha_module.W_k, mha_module.W_v

    backups = {
        'W_q': W_q.weight.data[s, :].clone(),
        'W_k': W_k.weight.data[s, :].clone(),
        'W_v': W_v.weight.data[s, :].clone(),
    }

    def apply():
        W_q.weight.data[s, :].zero_()
        W_k.weight.data[s, :].zero_()
        W_v.weight.data[s, :].zero_()

    def restore():
        W_q.weight.data[s, :].copy_(backups['W_q'])
        W_k.weight.data[s, :].copy_(backups['W_k'])
        W_v.weight.data[s, :].copy_(backups['W_v'])

    return backups, apply, restore


def evaluate_model(model, dataloader, device):
    """
    Evaluate model accuracy:
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)


            gen = model.generate(inputs, max_len=targets.size(1), start_token=targets[:, 0].mode()[0].item())

            gen = gen[:, :targets.size(1)]


            mask = (targets != PAD)
            match = ((gen == targets) | (~mask)).all(dim=1)  # 全部有效位匹配
            correct += match.sum().item()
            total += targets.size(0)

    return correct / max(total, 1)


def ablation_study(model, dataloader, device, output_dir):
    """
    Perform head ablation study.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running head ablation study...")

    # Baseline
    baseline_acc = evaluate_model(model, dataloader, device)
    print(f"Baseline accuracy: {baseline_acc:.2%}")

    ablation_results = {'baseline': baseline_acc}


    with torch.no_grad():
        # Encoder layers
        for li, layer in enumerate(model.encoder_layers):
            mha = layer.self_attn
            for h in range(mha.num_heads):
                _, apply, restore = _zero_head_slices(mha, h)
                apply()
                acc = evaluate_model(model, dataloader, device)
                restore()
                key = f'enc_l{li}_h{h}'
                ablation_results[key] = acc

        # Decoder self
        for li, layer in enumerate(model.decoder_layers):
            mha = layer.self_attn
            for h in range(mha.num_heads):
                _, apply, restore = _zero_head_slices(mha, h)
                apply()
                acc = evaluate_model(model, dataloader, device)
                restore()
                key = f'decSelf_l{li}_h{h}'
                ablation_results[key] = acc

        # Decoder cross
        for li, layer in enumerate(model.decoder_layers):
            mha = layer.cross_attn
            for h in range(mha.num_heads):
                _, apply, restore = _zero_head_slices(mha, h)
                apply()
                acc = evaluate_model(model, dataloader, device)
                restore()
                key = f'decCross_l{li}_h{h}'
                ablation_results[key] = acc

    # Save results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # Plot importance
    plot_head_importance(ablation_results, output_dir / 'head_importance.png')

    return ablation_results


def plot_head_importance(ablation_results, save_path):
    """
    Visualize head importance from ablation study.
    """
    baseline = ablation_results['baseline']
    items = [(k, v) for k, v in ablation_results.items() if k != 'baseline']
    items.sort(key=lambda x: x[0])

    labels = [k for k, _ in items]
    drops = [baseline - v for _, v in items]

    plt.figure(figsize=(max(12, 0.5 * len(labels)), 6))
    plt.bar(labels, drops)
    plt.xlabel('Head (layer & type)')
    plt.ylabel('Accuracy Drop')
    plt.title('Head Importance (Accuracy Drop When Removed)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()


def visualize_example_predictions(model, dataloader, device, output_dir, num_examples=5):
    """
    Visualize model predictions on example inputs.
    """
    output_dir = Path(output_dir)
    (output_dir / 'examples').mkdir(parents=True, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_examples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # first sample
            input_seq = inputs[0:1]
            target_seq = targets[0:1]

            # Use model.generate
            prediction = model.generate(input_seq, max_len=target_seq.size(1), start_token=target_seq[0, 0].item())

            # Convert to strings
            input_str = ' '.join(map(str, input_seq[0].cpu().numpy()))
            target_str = ''.join(map(str, target_seq[0].cpu().numpy()))
            pred_str = ''.join(map(str, prediction[0].cpu().numpy()))

            print(f"\nExample {batch_idx + 1}:")
            print(f"  Input:  {input_str}")
            print(f"  Target: {target_str}")
            print(f"  Pred:   {pred_str}")
            print(f"  Correct: {target_str == pred_str}")


            enc_attn, dec_self_attn, dec_cross_attn = [], [], []

            def make_hook(acc):
                def _h(m, i, o):
                    acc.append(o[1].detach().cpu())
                return _h

            h1 = model.decoder_layers[-1].cross_attn.register_forward_hook(make_hook(dec_cross_attn))
            dec_inp = target_seq[:, :-1]
            src_mask, tgt_mask = _make_masks(input_seq, dec_inp, pad_token=PAD, device=device)
            _ = model(input_seq, dec_inp, src_mask=src_mask, tgt_mask=tgt_mask)
            h1.remove()

            if len(dec_cross_attn) > 0:
                w = dec_cross_attn[0][0]  # [B,H,Lq,Lk] -> take B=0 => [H,Lq,Lk]
                title = f'Example {batch_idx+1} - Decoder Cross Attn (last layer)'
                save_path = output_dir / 'examples' / f'example_{batch_idx+1}_decCross.png'
                visualize_attention_pattern(
                    w.numpy(),
                    input_tokens=[str(t) for t in input_seq[0].cpu().numpy()],
                    output_tokens=[str(t) for t in dec_inp[0].cpu().numpy()],
                    title=title,
                    save_path=save_path
                )



def save_all_head_heatmaps(attention_data, output_dir):



    output_dir = Path(output_dir)
    (output_dir / 'encoder').mkdir(parents=True, exist_ok=True)
    (output_dir / 'decoder_self').mkdir(parents=True, exist_ok=True)
    (output_dir / 'decoder_cross').mkdir(parents=True, exist_ok=True)

    # 原始样本序列（numpy）
    inputs = attention_data['inputs']    # encoder 侧：list of [S]
    targets = attention_data['targets']  # decoder 侧：list of [T]

    def _strip_pad_to_str(seq_np, pad=PAD):
        seq = [int(x) for x in seq_np if int(x) != pad]
        return [str(x) for x in seq]

    encoder_inputs = [_strip_pad_to_str(seq) for seq in inputs]
    # targets_clean  = [_strip_pad_to_str(seq) for seq in targets]
    # # decoder_inputs = [BOS + target[:-1] ]
    # decoder_inputs = [tc[:-1] if len(tc) > 0 else [] for tc in targets_clean]
    targets_clean  = [_strip_pad_to_str(seq) for seq in targets]
    decoder_inputs = [ tc[:1] + tc[:-1] if len(tc) > 0 else [] for tc in targets_clean ]

    def _norm_iter(attn_list, layer_tags):

        if layer_tags is not None:
            for li, w in zip(layer_tags, attn_list):
                yield li, w
        else:
            for item in attn_list:
                if isinstance(item, tuple) and len(item) == 2:
                    li, w = item
                else:
                    li, w = -1, item
                yield li, w

    def _save(kind_name, attn_list, layer_tags):
        MAX_PER_LAYER = 6

        buckets = defaultdict(list)
        for li, weights in _norm_iter(attn_list, layer_tags):
            buckets[li].append(weights)

        for li in sorted(buckets.keys()):
            arr = buckets[li][:MAX_PER_LAYER]
            for j, weights in enumerate(arr):
                H, Lq, Lk = weights.shape

                if kind_name == 'encoder':
                    in_tokens  = encoder_inputs[j][:Lk] if j < len(encoder_inputs) else list(map(str, range(Lk)))
                    out_tokens = encoder_inputs[j][:Lq] if j < len(encoder_inputs) else list(map(str, range(Lq)))

                elif kind_name == 'decoder_self':
                    in_tokens  = decoder_inputs[j][:Lk] if j < len(decoder_inputs) else list(map(str, range(Lk)))
                    out_tokens = decoder_inputs[j][:Lq] if j < len(decoder_inputs) else list(map(str, range(Lq)))

                elif kind_name == 'decoder_cross':
                    in_tokens  = encoder_inputs[j][:Lk] if j < len(encoder_inputs) else list(map(str, range(Lk)))
                    out_tokens = decoder_inputs[j][:Lq] if j < len(decoder_inputs) else list(map(str, range(Lq)))

                else:
                    in_tokens  = list(map(str, range(Lk)))
                    out_tokens = list(map(str, range(Lq)))

                save_path = output_dir / kind_name / f'{kind_name}_l{li}_s{j}.png'
                visualize_attention_pattern(
                    weights.numpy(),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    title=f'{kind_name} - layer {li} sample {j}',
                    save_path=save_path
                )

    _save('encoder',
          attention_data['encoder_attention'],
          attention_data.get('encoder_layer_tags'))

    _save('decoder_self',
          attention_data['decoder_self_attention'],
          attention_data.get('decoder_self_layer_tags'))

    _save('decoder_cross',
          attention_data['decoder_cross_attention'],
          attention_data.get('decoder_cross_layer_tags'))


def main():
    parser = argparse.ArgumentParser(description='Analyze attention patterns')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Load model
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512
    ).to(args.device)

    state = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state)
    print(f"Loaded model from {args.model_path}")

    print(len(model.decoder_layers))
    print("===========")

    # Load data
    _, _, test_loader = create_dataloaders(args.data_dir, args.batch_size)

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'attention_patterns').mkdir(parents=True, exist_ok=True)
    (output_dir / 'head_analysis').mkdir(parents=True, exist_ok=True)

    # Extract attention weights
    print("Extracting attention weights...")
    attention_data = extract_attention_weights(
        model, test_loader, args.device, args.num_samples
    )

    save_all_head_heatmaps(attention_data, output_dir / 'attention_patterns')


    # Analyze head specialization
    head_stats = analyze_head_specialization(
        attention_data, output_dir / 'head_analysis'
    )
    print("Head stats (encoder):", json.dumps(head_stats.get('encoder', {}), indent=2))

    # Run ablation study
    ablation_results = ablation_study(
        model, test_loader, args.device, output_dir / 'head_analysis'
    )
    print("Ablation summary:", json.dumps(ablation_results, indent=2))

    # Visualize example predictions
    visualize_example_predictions(
        model, test_loader, args.device, output_dir, num_examples=5
    )

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == '__main__':

    main()
