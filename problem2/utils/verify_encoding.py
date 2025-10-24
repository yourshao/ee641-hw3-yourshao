"""
Verification utilities for positional encoding implementations.

Run this to test your positional encoding implementations before training.

Usage:
    python utils/verify_encoding.py
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    NoPositionalEncoding,
    get_positional_encoding
)


def test_sinusoidal_encoding():
    """Test sinusoidal positional encoding."""
    print("\n" + "=" * 60)
    print("TEST 1: Sinusoidal Positional Encoding")
    print("=" * 60)

    d_model = 64
    batch_size, seq_len = 2, 16

    try:
        encoding = SinusoidalPositionalEncoding(d_model, max_len=1000)
        x = torch.randn(batch_size, seq_len, d_model)
        output = encoding(x)

        # Check shape
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        print(f"✓ Output shape correct: {output.shape}")

        # Check that it's actually adding something
        difference = (output - x).abs().mean().item()
        if difference < 1e-6:
            print(f"✗ Encoding doesn't change input (difference: {difference})")
            return False
        print(f"✓ Encoding modifies input (mean change: {difference:.4f})")

        # Check buffer exists
        assert hasattr(encoding, 'pe'), "Missing 'pe' buffer"
        assert encoding.pe.shape == (1, 1000, d_model), f"Buffer shape incorrect: {encoding.pe.shape}"
        print(f"✓ PE buffer shape correct: {encoding.pe.shape}")

        # Check that different positions have different encodings
        pe_pos0 = encoding.pe[0, 0, :]
        pe_pos1 = encoding.pe[0, 1, :]
        pe_distance = torch.norm(pe_pos0 - pe_pos1).item()

        if pe_distance < 0.1:
            print(f"✗ Positions too similar (distance: {pe_distance:.4f})")
            return False
        print(f"✓ Different positions have different encodings (distance: {pe_distance:.4f})")

        return True

    except NotImplementedError:
        print("✗ SinusoidalPositionalEncoding not implemented")
        return False
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_sinusoidal_extrapolation():
    """Test that sinusoidal encoding can handle longer sequences."""
    print("\n" + "=" * 60)
    print("TEST 2: Sinusoidal Extrapolation")
    print("=" * 60)

    d_model = 64

    try:
        encoding = SinusoidalPositionalEncoding(d_model, max_len=100)

        # Test on sequence longer than what we've precomputed
        x_long = torch.randn(1, 256, d_model)
        output = encoding(x_long)

        assert output.shape == x_long.shape, "Extrapolation failed"
        print(f"✓ Can handle sequences longer than max_len")
        print(f"  Precomputed: 100, Tested: 256")
        return True

    except NotImplementedError:
        print("✗ Not implemented")
        return False
    except Exception as e:
        print(f"✗ Extrapolation failed: {e}")
        print(f"  Make sure your forward() can handle seq_len > max_len")
        return False


def test_learned_encoding():
    """Test learned positional encoding."""
    print("\n" + "=" * 60)
    print("TEST 3: Learned Positional Encoding")
    print("=" * 60)

    d_model = 64
    max_len = 100
    batch_size, seq_len = 2, 16

    try:
        encoding = LearnedPositionalEncoding(d_model, max_len=max_len)
        x = torch.randn(batch_size, seq_len, d_model)
        output = encoding(x)

        # Check shape
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        print(f"✓ Output shape correct: {output.shape}")

        # Check parameters exist
        param_count = sum(p.numel() for p in encoding.parameters())
        expected_params = max_len * d_model
        assert param_count == expected_params, \
            f"Parameter count {param_count} != expected {expected_params}"
        print(f"✓ Parameter count correct: {param_count:,}")

        # Check embedding layer exists
        assert hasattr(encoding, 'position_embeddings'), "Missing position_embeddings"
        print(f"✓ Position embeddings layer exists")

        return True

    except NotImplementedError:
        print("✗ LearnedPositionalEncoding not implemented")
        return False
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_learned_encoding_limit():
    """Test that learned encoding fails on sequences longer than max_len."""
    print("\n" + "=" * 60)
    print("TEST 4: Learned Encoding Length Limit")
    print("=" * 60)

    d_model = 64
    max_len = 50

    try:
        encoding = LearnedPositionalEncoding(d_model, max_len=max_len)

        # Should work for seq_len <= max_len
        x_ok = torch.randn(1, 50, d_model)
        output_ok = encoding(x_ok)
        print(f"✓ Works for seq_len={50} (max_len={max_len})")

        # Should fail for seq_len > max_len
        x_too_long = torch.randn(1, 100, d_model)
        try:
            output_fail = encoding(x_too_long)
            print(f"✗ Should have raised error for seq_len > max_len")
            return False
        except AssertionError:
            print(f"✓ Correctly raises error for seq_len > max_len")
            return True

    except NotImplementedError:
        print("✗ Not implemented")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_no_encoding():
    """Test that no encoding returns input unchanged."""
    print("\n" + "=" * 60)
    print("TEST 5: No Positional Encoding")
    print("=" * 60)

    d_model = 64
    batch_size, seq_len = 2, 16

    try:
        encoding = NoPositionalEncoding(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        output = encoding(x)

        # Should be exactly the same
        assert torch.allclose(output, x), "Output should equal input"
        print(f"✓ Output equals input (no position information added)")

        # Should have no parameters
        param_count = sum(p.numel() for p in encoding.parameters())
        assert param_count == 0, f"Should have 0 parameters, got {param_count}"
        print(f"✓ No learnable parameters")

        return True

    except NotImplementedError:
        print("✗ NoPositionalEncoding not implemented")
        return False
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_factory_function():
    """Test the factory function."""
    print("\n" + "=" * 60)
    print("TEST 6: Factory Function")
    print("=" * 60)

    d_model = 64
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, d_model)

    try:
        all_work = True
        for enc_type in ['sinusoidal', 'learned', 'none']:
            try:
                encoding = get_positional_encoding(enc_type, d_model, max_len=200)
                output = encoding(x)
                param_count = sum(p.numel() for p in encoding.parameters())
                print(f"✓ {enc_type:12s}: shape {output.shape}, params {param_count:6,}")
            except NotImplementedError:
                print(f"✗ {enc_type:12s}: Not implemented")
                all_work = False
            except Exception as e:
                print(f"✗ {enc_type:12s}: Error - {e}")
                all_work = False

        return all_work

    except Exception as e:
        print(f"✗ Factory function error: {e}")
        return False


def test_encoding_patterns():
    """Test that encodings have expected patterns."""
    print("\n" + "=" * 60)
    print("TEST 7: Encoding Patterns")
    print("=" * 60)

    d_model = 64

    try:
        # Sinusoidal should have periodic structure
        sin_enc = SinusoidalPositionalEncoding(d_model, max_len=1000)

        # Check that encodings at positions 0 and 100 are different
        pe_0 = sin_enc.pe[0, 0, :]
        pe_100 = sin_enc.pe[0, 100, :]
        distance = torch.norm(pe_0 - pe_100).item()

        if distance < 1.0:
            print(f"✗ Positions too similar (distance: {distance:.4f})")
            return False

        print(f"✓ Sinusoidal encoding varies with position")
        print(f"  Distance between pos 0 and 100: {distance:.4f}")

        # Check alternating sin/cos pattern
        pe_sample = sin_enc.pe[0, 10, :10]
        print(f"✓ Position 10 encoding (first 10 dims): {pe_sample.tolist()}")

        return True

    except NotImplementedError:
        print("✗ Not implemented")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("POSITIONAL ENCODING VERIFICATION")
    print("=" * 60)
    print("\nThis script tests your positional encoding implementations.")
    print("Fix any failing tests before training your models.")

    results = []
    results.append(("Sinusoidal encoding", test_sinusoidal_encoding()))
    results.append(("Sinusoidal extrapolation", test_sinusoidal_extrapolation()))
    results.append(("Learned encoding", test_learned_encoding()))
    results.append(("Learned encoding limit", test_learned_encoding_limit()))
    results.append(("No encoding", test_no_encoding()))
    results.append(("Factory function", test_factory_function()))
    results.append(("Encoding patterns", test_encoding_patterns()))

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
        print("\nYour positional encoding implementations are ready!")
        print("You can now run: python train.py")
    else:
        print(f"SOME TESTS FAILED ({passed}/{total} passed)")
        print("\nFix the failing tests before training.")
    print("=" * 60)

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
