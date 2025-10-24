"""
Verification utilities for attention implementation.

Run this to test your attention implementation before training.

Usage:
    python utils/verify_attention.py
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.attention import scaled_dot_product_attention, MultiHeadAttention, create_causal_mask


def test_attention_shapes():
    """Test that attention produces correct output shapes."""
    print("\n" + "=" * 60)
    print("TEST 1: Attention Output Shapes")
    print("=" * 60)

    batch_size, seq_len, d_k = 2, 8, 16
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    try:
        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (batch_size, seq_len, d_k), \
            f"Output shape {output.shape} != expected {(batch_size, seq_len, d_k)}"
        assert weights.shape == (batch_size, seq_len, seq_len), \
            f"Weights shape {weights.shape} != expected {(batch_size, seq_len, seq_len)}"

        print(f"✓ Attention output shape: {output.shape}")
        print(f"✓ Attention weights shape: {weights.shape}")
        return True

    except NotImplementedError:
        print("✗ scaled_dot_product_attention not implemented")
        return False
    except AssertionError as e:
        print(f"✗ Shape mismatch: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_attention_weights():
    """Test that attention weights sum to 1."""
    print("\n" + "=" * 60)
    print("TEST 2: Attention Weights Sum to 1")
    print("=" * 60)

    batch_size, seq_len, d_k = 2, 4, 8
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    try:
        output, weights = scaled_dot_product_attention(Q, K, V)

        # Check that weights sum to 1 along last dimension
        weight_sums = weights.sum(dim=-1)
        all_close_to_one = torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

        if all_close_to_one:
            print(f"✓ Attention weights sum to 1.0")
            print(f"  Example sums: {weight_sums[0].tolist()}")
            return True
        else:
            print(f"✗ Attention weights don't sum to 1.0")
            print(f"  Expected: all 1.0")
            print(f"  Got: {weight_sums[0].tolist()}")
            return False

    except NotImplementedError:
        print("✗ scaled_dot_product_attention not implemented")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_causal_masking():
    """Test that causal mask prevents attending to future positions."""
    print("\n" + "=" * 60)
    print("TEST 3: Causal Masking")
    print("=" * 60)

    seq_len, d_k = 4, 8
    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)

    try:
        # Create causal mask
        mask = create_causal_mask(seq_len)

        # Apply attention with mask
        output, weights = scaled_dot_product_attention(Q, K, V, mask)

        # Check that upper triangle is zero (no attention to future)
        upper_triangle = weights[0].triu(diagonal=1)
        is_causal = (upper_triangle.abs() < 1e-6).all().item()

        if is_causal:
            print(f"✓ Causal masking working correctly")
            print(f"  Attention pattern (should be lower triangular):")
            print(weights[0].detach().numpy().round(3))
            return True
        else:
            print(f"✗ Causal masking failed - found attention to future positions")
            print(f"  Attention pattern:")
            print(weights[0].detach().numpy().round(3))
            return False

    except NotImplementedError:
        print("✗ Not implemented")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_multihead_attention():
    """Test multi-head attention implementation."""
    print("\n" + "=" * 60)
    print("TEST 4: Multi-Head Attention")
    print("=" * 60)

    d_model, num_heads = 64, 4
    batch_size, seq_len = 2, 8

    try:
        mha = MultiHeadAttention(d_model, num_heads)

        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)

        output, weights = mha(Q, K, V)

        # Check shapes
        assert output.shape == (batch_size, seq_len, d_model), \
            f"Output shape {output.shape} != expected {(batch_size, seq_len, d_model)}"
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len), \
            f"Weights shape {weights.shape} != expected {(batch_size, num_heads, seq_len, seq_len)}"

        print(f"✓ Multi-head attention output shape: {output.shape}")
        print(f"✓ Multi-head attention weights shape: {weights.shape}")
        print(f"✓ Parameters: {sum(p.numel() for p in mha.parameters()):,}")

        # Test self-attention (Q=K=V)
        x = torch.randn(batch_size, seq_len, d_model)
        output_self, _ = mha(x, x, x)
        assert output_self.shape == x.shape, "Self-attention shape mismatch"
        print(f"✓ Self-attention works correctly")

        return True

    except NotImplementedError:
        print("✗ MultiHeadAttention not implemented")
        return False
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_head_independence():
    """Test that different heads produce different attention patterns."""
    print("\n" + "=" * 60)
    print("TEST 5: Head Independence")
    print("=" * 60)

    d_model, num_heads = 64, 4
    batch_size, seq_len = 1, 8

    try:
        mha = MultiHeadAttention(d_model, num_heads)

        x = torch.randn(batch_size, seq_len, d_model)
        output, weights = mha(x, x, x)

        # Check that different heads have different patterns
        weights_reshaped = weights.squeeze(0)  # [num_heads, seq_len, seq_len]

        differences = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                diff = (weights_reshaped[i] - weights_reshaped[j]).abs().mean().item()
                differences.append(diff)

        avg_difference = sum(differences) / len(differences)

        if avg_difference > 0.01:  # Heads should be different
            print(f"✓ Heads produce different attention patterns")
            print(f"  Average difference between heads: {avg_difference:.4f}")
            return True
        else:
            print(f"✗ All heads producing same pattern (avg diff: {avg_difference:.4f})")
            print(f"  Check that you're splitting heads correctly")
            return False

    except NotImplementedError:
        print("✗ Not implemented")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ATTENTION IMPLEMENTATION VERIFICATION")
    print("=" * 60)
    print("\nThis script tests your attention implementation.")
    print("Fix any failing tests before training your model.")

    results = []
    results.append(("Output shapes", test_attention_shapes()))
    results.append(("Weight normalization", test_attention_weights()))
    results.append(("Causal masking", test_causal_masking()))
    results.append(("Multi-head attention", test_multihead_attention()))
    results.append(("Head independence", test_head_independence()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {test_name}")

    print("\n" + "=" * 60)
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
        print("\nYour attention implementation is ready!")
        print("You can now run: python train.py")
    else:
        print(f"SOME TESTS FAILED ({passed}/{total} passed)")
        print("\nFix the failing tests before training.")
    print("=" * 60)

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
