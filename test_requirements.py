"""
Unit tests for Qwen3-235B-A22B performance estimation functions.
"""

import pytest
import math
from requirements import (
    estimate_reqs_embedding,
    estimate_reqs_attention,
    estimate_reqs_moe_mlp,
    estimate_reqs_transformer_layer,
    estimate_reqs_qwen3,
    VOCAB_SIZE, NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM,
    NUM_Q_HEADS, NUM_KV_HEADS, TOTAL_EXPERTS, ACTIVATED_EXPERTS_PER_NODE,
    NUM_NODES, PARAM_BYTES, ACTIVATION_BYTES
)


class TestEmbedding:
    """Test embedding layer estimation."""

    def test_embedding_basic(self):
        """Test basic embedding layer computation."""
        batch_size = 1
        context_length = 1024

        memory, interconnect, flops, kv_cache = estimate_reqs_embedding(batch_size, context_length)

        # Should read one embedding vector per token
        expected_memory = HIDDEN_DIM * PARAM_BYTES
        assert memory == expected_memory

        # No interconnect for embedding lookup
        assert interconnect == 0.0

        # No FLOPs for embedding lookup
        assert flops == 0.0

        # No KV cache for embedding
        assert kv_cache == 0.0

    def test_embedding_memory_per_token_is_constant(self):
        """Test that embedding memory per token is constant."""
        mem_per_token = []
        for batch_size in [1, 4, 16]:
            for context_length in [1024, 4096, 128*1024]:
                # Test in prefill mode where tokens_processed = batch_size * context_length
                memory, _, _, _ = estimate_reqs_embedding(batch_size, context_length, is_prefill=True)
                tokens_processed = batch_size * context_length
                mem_per_token.append(memory / tokens_processed)

        # All results should be identical
        first_result = mem_per_token[0]
        for result in mem_per_token[1:]:
            assert abs(result - first_result) < 1e-9


class TestAttention:
    """Test attention block estimation."""

    def test_attention_basic(self):
        """Test basic attention computation."""
        batch_size = 1
        context_length = 1024

        memory, interconnect, flops, kv_cache = estimate_reqs_attention(batch_size, context_length)

        # Should have positive values for all metrics
        assert memory > 0
        assert interconnect > 0
        assert flops > 0
        assert kv_cache > 0

    def test_attention_flops_scale_quadratically_with_context(self):
        """Test that attention FLOPs scale quadratically with context length during prefill."""
        batch_size = 1

        # Prefill mode, where the quadratic term dominates
        _, _, flops_1k, _ = estimate_reqs_attention(batch_size, 1024, is_prefill=True)
        _, _, flops_2k, _ = estimate_reqs_attention(batch_size, 2048, is_prefill=True)
        _, _, flops_4k, _ = estimate_reqs_attention(batch_size, 4096, is_prefill=True)

        # FLOPs should scale quadratically with context length (N^2)
        # So, doubling context should quadruple FLOPs.
        ratio_2k_1k = flops_2k / flops_1k
        ratio_4k_2k = flops_4k / flops_2k

        # Allow a wide tolerance because of the significant linear-term FLOPs from projections.
        # The key is that the ratio should be super-linear (well above 2.0).
        assert 2.2 < ratio_2k_1k < 3.0
        assert 2.2 < ratio_4k_2k < 3.0

    def test_attention_scales_with_batch(self):
        """Test that attention requirements scale with batch size."""
        context_length = 1024

        memory_1, interconnect_1, flops_1, kv_cache_1 = estimate_reqs_attention(1, context_length)
        memory_4, interconnect_4, flops_4, kv_cache_4 = estimate_reqs_attention(4, context_length)

        # Most metrics should scale with batch size
        assert flops_4 / flops_1 == 4.0
        assert kv_cache_4 / kv_cache_1 == 4.0
        assert interconnect_4 / interconnect_1 == 4.0

    def test_kv_cache_size(self):
        """Test KV cache size calculation for prefill and generation."""
        batch_size = 2
        context_length = 1024
        kv_heads_per_node = NUM_KV_HEADS // NUM_NODES
        head_dim = HIDDEN_DIM // NUM_Q_HEADS

        # Test Prefill
        _, _, _, kv_cache_prefill = estimate_reqs_attention(batch_size, context_length, is_prefill=True)
        expected_kv_cache_prefill = 2 * batch_size * context_length * kv_heads_per_node * head_dim * ACTIVATION_BYTES
        assert abs(kv_cache_prefill - expected_kv_cache_prefill) < 1e-6

        # Test Generation
        _, _, _, kv_cache_gen = estimate_reqs_attention(batch_size, context_length, is_prefill=False)
        effective_seq_len_gen = context_length + 1
        expected_kv_cache_gen = 2 * batch_size * effective_seq_len_gen * kv_heads_per_node * head_dim * ACTIVATION_BYTES
        assert abs(kv_cache_gen - expected_kv_cache_gen) < 1e-6


class TestMoEMLP:
    """Test MoE MLP block estimation."""

    def test_moe_basic(self):
        """Test basic MoE computation."""
        batch_size = 1
        context_length = 1024

        memory, interconnect, flops, kv_cache = estimate_reqs_moe_mlp(batch_size, context_length)

        # Should have positive values for first three metrics
        assert memory > 0
        assert interconnect > 0
        assert flops > 0

        # No KV cache for MLP
        assert kv_cache == 0.0

    def test_moe_expert_scaling(self):
        """Test that MoE FLOPs are proportional to activated experts."""
        batch_size = 1
        context_length = 1024

        _, _, flops, _ = estimate_reqs_moe_mlp(batch_size, context_length)

        # FLOPs should be roughly proportional to activated experts per node
        # Each expert does: 2 * hidden_dim * intermediate_dim + intermediate_dim FLOPs
        expected_flops_per_expert = batch_size * (
            HIDDEN_DIM * INTERMEDIATE_DIM +  # Gate
            HIDDEN_DIM * INTERMEDIATE_DIM +  # Up
            INTERMEDIATE_DIM * HIDDEN_DIM +  # Down
            INTERMEDIATE_DIM  # SwiGLU
        )

        # Should be roughly ACTIVATED_EXPERTS_PER_NODE * expected + gating overhead
        expected_min = ACTIVATED_EXPERTS_PER_NODE * expected_flops_per_expert
        assert flops >= expected_min

    def test_moe_scales_with_batch(self):
        """Test MoE scaling with batch size."""
        context_length = 1024

        _, _, flops_1, _ = estimate_reqs_moe_mlp(1, context_length)
        _, _, flops_4, _ = estimate_reqs_moe_mlp(4, context_length)

        # FLOPs should scale linearly with batch size
        ratio = flops_4 / flops_1
        assert 3.8 < ratio < 4.2  # Allow some tolerance


class TestTransformerLayer:
    """Test complete transformer layer estimation."""

    def test_layer_combines_components(self):
        """Test that layer estimates combine attention and MLP correctly."""
        batch_size = 2
        context_length = 2048

        attn_memory, attn_interconnect, attn_flops, attn_kv = estimate_reqs_attention(
            batch_size, context_length)
        mlp_memory, mlp_interconnect, mlp_flops, mlp_kv = estimate_reqs_moe_mlp(
            batch_size, context_length)

        layer_memory, layer_interconnect, layer_flops, layer_kv = estimate_reqs_transformer_layer(
            batch_size, context_length)

        # Layer should be sum of components plus layer norm overhead
        assert layer_memory >= attn_memory + mlp_memory
        assert layer_interconnect == attn_interconnect + mlp_interconnect
        assert layer_flops >= attn_flops + mlp_flops
        assert layer_kv == attn_kv + mlp_kv

    def test_layer_scaling(self):
        """Test transformer layer scaling properties."""
        # Test with different batch sizes
        _, _, flops_1, _ = estimate_reqs_transformer_layer(1, 1024)
        _, _, flops_2, _ = estimate_reqs_transformer_layer(2, 1024)

        # Should scale roughly linearly with batch size
        ratio = flops_2 / flops_1
        assert 1.8 < ratio < 2.2


class TestFullModel:
    """Test complete Qwen3 model estimation."""

    def test_full_model_basic(self):
        """Test basic full model computation."""
        batch_size = 1
        context_length = 4096

        memory, interconnect, flops, kv_cache = estimate_reqs_qwen3(batch_size, context_length)

        # All metrics should be positive
        assert memory > 0
        assert interconnect > 0
        assert flops > 0
        assert kv_cache > 0

    def test_full_model_has_all_components(self):
        """Test that full model includes all major components."""
        batch_size = 1
        context_length = 1024

        # Get individual component estimates
        emb_memory, _, emb_flops, _ = estimate_reqs_embedding(batch_size, context_length)
        layer_memory, _, layer_flops, _ = estimate_reqs_transformer_layer(batch_size, context_length)

        # Get full model estimate
        full_memory, _, full_flops, _ = estimate_reqs_qwen3(batch_size, context_length)

        # Full model should include embedding + all layers + output
        expected_min_memory = emb_memory + NUM_LAYERS * layer_memory
        expected_min_flops = emb_flops + NUM_LAYERS * layer_flops

        assert full_memory >= expected_min_memory
        assert full_flops >= expected_min_flops

    def test_reasonable_parameter_count(self):
        """Test that implied parameter count is reasonable."""
        batch_size = 1
        context_length = 1024

        memory_per_token, _, _, _ = estimate_reqs_qwen3(batch_size, context_length)

        # Memory per token gives us rough sense of parameter access
        # Should be in reasonable range (not orders of magnitude off)
        # Activated params ~22B, so roughly 22GB of parameter access per token
        expected_order_of_magnitude = 20e9  # ~20GB

        assert 5e9 < memory_per_token < 100e9  # 5GB to 100GB range

    def test_kv_cache_grows_with_context(self):
        """Test that KV cache grows linearly with context length."""
        batch_size = 1

        _, _, _, kv_1k = estimate_reqs_qwen3(batch_size, 1024)
        _, _, _, kv_2k = estimate_reqs_qwen3(batch_size, 2048)

        # KV cache should double when context length doubles
        ratio = kv_2k / kv_1k
        assert 1.9 < ratio < 2.1

    def test_context_length_parameter(self):
        """Test different context lengths."""
        batch_size = 1

        # Test with various context lengths
        for context_length in [1024, 4096, 32768, 128*1024]:
            memory, interconnect, flops, kv_cache = estimate_reqs_qwen3(batch_size, context_length)

            # All should be positive
            assert memory > 0
            assert interconnect > 0
            assert flops > 0
            assert kv_cache > 0

            # KV cache should grow with context length
            assert kv_cache > 0


class TestReasonableValues:
    """Test that estimated values are in reasonable ranges."""

    def test_memory_bandwidth_feasible(self):
        """Test that memory bandwidth requirements are feasible."""
        batch_size = 1
        context_length = 4096

        memory_per_token, _, _, _ = estimate_reqs_qwen3(batch_size, context_length)

        # Memory per token should be feasible given 256 GB/s bandwidth
        # At realistic token rates (e.g., 10 tokens/sec for large model inference)
        tokens_per_second = 10  # More realistic for 235B model
        bandwidth_required = memory_per_token * tokens_per_second
        available_bandwidth = 256e9  # 256 GB/s

        # Should use reasonable fraction of available bandwidth
        utilization = bandwidth_required / available_bandwidth
        assert utilization < 1.0  # Should not exceed 100% utilization

    def test_interconnect_bandwidth_feasible(self):
        """Test that interconnect bandwidth is feasible."""
        batch_size = 4  # Higher batch size for more communication
        context_length = 4096

        _, interconnect_per_token, _, _ = estimate_reqs_qwen3(batch_size, context_length)

        # Interconnect per token should be feasible given 8 GB/s per pair
        tokens_per_second = 50
        interconnect_required = interconnect_per_token * tokens_per_second
        available_interconnect = 8e9  # 8 GB/s per pair

        utilization = interconnect_required / available_interconnect
        assert utilization < 1.0  # Should not exceed bandwidth

    def test_flops_reasonable(self):
        """Test that FLOP counts are reasonable."""
        batch_size = 1
        context_length = 1024

        _, _, flops_per_token, _ = estimate_reqs_qwen3(batch_size, context_length)

        # FLOPs per token should be in reasonable range
        # Rough estimate: ~22B activated params * 2 FLOPs/param = ~44B FLOPs
        expected_order = 50e9  # ~50 billion FLOPs

        # Should be within order of magnitude
        assert 10e9 < flops_per_token < 500e9


if __name__ == "__main__":
    # Run basic smoke test
    print("Running basic smoke test...")

    batch_size = 1
    context_length = 4096

    print("\n=== Component Tests ===")

    # Test each component
    emb_results = estimate_reqs_embedding(batch_size, context_length)
    print(f"Embedding: Memory={emb_results[0]/1e6:.1f}MB, Flops={emb_results[2]/1e9:.1f}G")

    attn_results = estimate_reqs_attention(batch_size, context_length)
    print(f"Attention: Memory={attn_results[0]/1e6:.1f}MB, Flops={attn_results[2]/1e9:.1f}G, KV={attn_results[3]/1e6:.1f}MB")

    mlp_results = estimate_reqs_moe_mlp(batch_size, context_length)
    print(f"MoE MLP: Memory={mlp_results[0]/1e6:.1f}MB, Flops={mlp_results[2]/1e9:.1f}G")

    layer_results = estimate_reqs_transformer_layer(batch_size, context_length)
    print(f"Layer: Memory={layer_results[0]/1e6:.1f}MB, Flops={layer_results[2]/1e9:.1f}G, KV={layer_results[3]/1e6:.1f}MB")

    full_results = estimate_reqs_qwen3(batch_size, context_length)
    print(f"Full Model: Memory={full_results[0]/1e9:.1f}GB, Interconnect={full_results[1]/1e6:.1f}MB, Flops={full_results[2]/1e12:.1f}T, KV={full_results[3]/1e9:.1f}GB")

    print("\n=== Scaling Tests ===")

    # Test scaling with batch size
    for bs in [1, 2, 4, 8]:
        results = estimate_reqs_qwen3(bs, 4096)
        print(f"Batch {bs}: Flops={results[2]/1e12:.2f}T, KV={results[3]/1e9:.2f}GB")

    # Test scaling with context length
    for ctx in [1024, 4096, 16384, 65536]:
        results = estimate_reqs_qwen3(1, ctx)
        print(f"Context {ctx}: Flops={results[2]/1e12:.2f}T, KV={results[3]/1e9:.2f}GB")

    print("\nSmoke test completed successfully!")
