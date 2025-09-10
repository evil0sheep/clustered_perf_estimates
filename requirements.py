"""
Qwen3-235B-A22B cluster performance estimation functions.

This module estimates memory bandwidth, interconnect bandwidth, FLOPs, and KV cache
requirements per forward pass for running Qwen3-235B-A22B on a 4-node cluster with
distributed MoE experts and GQA attention.
"""

import math
from typing import Tuple, Dict, Any


# Model architecture constants for Qwen3-235B-A22B
VOCAB_SIZE = 151669
NUM_LAYERS = 94
HIDDEN_DIM = 4096   # Base model dimension
INTERMEDIATE_DIM = 2048  # Sized to match ~22B active params constraint (~5.5B per node)
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = HIDDEN_DIM // NUM_Q_HEADS  # 64
TOTAL_EXPERTS = 128
EXPERTS_PER_NODE = 32  # 128 total experts / 4 nodes
ACTIVATED_EXPERTS_PER_TOKEN = 8  # Global
ACTIVATED_EXPERTS_PER_NODE = 2  # Average per node

# Hardware constants
NUM_NODES = 4
MEMORY_BANDWIDTH_PER_NODE = 256e9  # 256 GB/s
INTERCONNECT_BANDWIDTH_PER_PAIR = 8e9  # 8 GB/s between each pair

# Data type sizes
PARAM_BYTES = 1  # 8-bit quantized parameters
ACTIVATION_BYTES = 2  # fp16 activations


def estimate_reqs_embedding(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for embedding layer per forward pass.

    Args:
        batch_size: Number of sequences in batch
        context_length: Sequence length in tokens
        is_prefill: If True, assume empty KV cache (prefill). If False, assume populated cache (generation)

    Returns:
        (memory_bus_bytes_per_forward_pass, interconnect_bytes_per_forward_pass,
         floating_point_ops_per_forward_pass, max_kv_cache_bytes)
    """
    # Determine tokens processed in this forward pass
    tokens_processed = (context_length * batch_size) if is_prefill else batch_size

    # Embedding lookup: just indexing, not matrix multiplication
    # Memory bandwidth: read embedding parameters for each token
    memory_per_token = HIDDEN_DIM * PARAM_BYTES
    memory_per_forward_pass = memory_per_token * tokens_processed

    # No interconnect communication for embedding lookup
    interconnect_per_forward_pass = 0.0

    # No FLOPs for embedding lookup (just memory access)
    flops_per_forward_pass = 0.0

    # No KV cache for embedding layer
    kv_cache_bytes = 0.0

    return memory_per_forward_pass, interconnect_per_forward_pass, flops_per_forward_pass, kv_cache_bytes


def estimate_reqs_attention(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for a single GQA attention block per forward pass.

    With GQA, we have 64 Q heads and 4 KV heads distributed across 4 nodes.
    Each node handles 16 Q heads and 1 KV head group.

    Args:
        batch_size: Number of sequences in batch
        context_length: Sequence length in tokens
        is_prefill: If True, assume empty KV cache (prefill). If False, assume populated cache (generation)

    Returns:
        (memory_bus_bytes_per_forward_pass, interconnect_bytes_per_forward_pass,
         floating_point_ops_per_forward_pass, max_kv_cache_bytes)
    """
    # Determine effective sequence length for attention computation
    if is_prefill:
        # During prefill, we process the entire context at once
        effective_seq_len = context_length
        tokens_processed = context_length * batch_size
    else:
        # During generation, we only process 1 new token but attend to full context
        effective_seq_len = context_length + 1  # Attending to existing + new token
        tokens_processed = batch_size  # Only processing 1 new token per sequence

    # Parameter sizes (per node) - reused across batch, so amortize over batch_size
    q_params = (HIDDEN_DIM * HIDDEN_DIM // NUM_NODES) * PARAM_BYTES  # 16 of 64 heads
    kv_params = (HIDDEN_DIM * (HIDDEN_DIM // NUM_Q_HEADS * NUM_KV_HEADS) // NUM_NODES) * PARAM_BYTES  # 1 of 4 KV heads
    o_params = (HIDDEN_DIM * HIDDEN_DIM // NUM_NODES) * PARAM_BYTES  # Output projection split

    memory_per_forward_pass = q_params + kv_params + o_params

    # Flash Attention: only need to read/write Q, K, V, O - no large attention matrices materialized
    # Activation memory reads/writes for all tokens processed
    q_activations = tokens_processed * (HIDDEN_DIM // NUM_NODES) * ACTIVATION_BYTES
    kv_activations = tokens_processed * (HIDDEN_DIM // NUM_Q_HEADS * NUM_KV_HEADS // NUM_NODES) * ACTIVATION_BYTES
    attention_output = tokens_processed * (HIDDEN_DIM // NUM_NODES) * ACTIVATION_BYTES

    memory_per_forward_pass += q_activations + kv_activations + attention_output

    # KV cache reads during attention computation
    if not is_prefill and effective_seq_len > 1:
        # During generation, we read existing KV cache for all sequences
        kv_heads_per_node = NUM_KV_HEADS // NUM_NODES
        kv_cache_read = batch_size * (effective_seq_len - 1) * kv_heads_per_node * HEAD_DIM * ACTIVATION_BYTES * 2  # K + V
        memory_per_forward_pass += kv_cache_read

    # FLOPs for all tokens processed
    heads_per_node = NUM_Q_HEADS // NUM_NODES

    # Q @ K^T: batch_size * heads_per_node * effective_seq_len * head_dim per token processed
    qk_flops_per_token = batch_size * heads_per_node * effective_seq_len * HEAD_DIM

    # Attention weights @ V: batch_size * heads_per_node * effective_seq_len * head_dim per token processed
    av_flops_per_token = batch_size * heads_per_node * effective_seq_len * HEAD_DIM

    # Output projection: batch_size * hidden_dim * (hidden_dim // num_nodes) per token processed
    o_flops_per_token = batch_size * HIDDEN_DIM * (HIDDEN_DIM // NUM_NODES)

    # Total FLOPs for forward pass
    if is_prefill:
        # For prefill, FLOPs scale with context_length tokens processed
        flops_per_forward_pass = (qk_flops_per_token + av_flops_per_token) * context_length + o_flops_per_token * context_length
    else:
        # For generation, just process 1 new token per sequence
        flops_per_forward_pass = qk_flops_per_token + av_flops_per_token + o_flops_per_token

    # Interconnect: gather attention outputs from all nodes for all tokens processed
    interconnect_per_forward_pass = (NUM_NODES - 1) * tokens_processed * (HIDDEN_DIM // NUM_NODES) * ACTIVATION_BYTES

    # KV cache per node: stores keys and values for 1 KV head group
    kv_heads_per_node = NUM_KV_HEADS // NUM_NODES
    kv_cache_per_head = batch_size * effective_seq_len * HEAD_DIM * ACTIVATION_BYTES
    kv_cache_bytes = 2 * kv_heads_per_node * kv_cache_per_head  # K + V

    return memory_per_forward_pass, interconnect_per_forward_pass, flops_per_forward_pass, kv_cache_bytes


def estimate_reqs_moe_mlp(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for MoE MLP block per forward pass.

    Each node has 32 experts, and on average 2 experts per node are activated per token.
    Expert routing is replicated across nodes.

    Args:
        batch_size: Number of sequences in batch
        context_length: Sequence length in tokens
        is_prefill: If True, assume empty KV cache (prefill). If False, assume populated cache (generation)

    Returns:
        (memory_bus_bytes_per_forward_pass, interconnect_bytes_per_forward_pass,
         floating_point_ops_per_forward_pass, max_kv_cache_bytes)
    """
    tokens_processed = (context_length * batch_size) if is_prefill else batch_size

    # Parameters per expert (SwiGLU: gate, up, down projections)
    gate_params = HIDDEN_DIM * INTERMEDIATE_DIM * PARAM_BYTES
    up_params = HIDDEN_DIM * INTERMEDIATE_DIM * PARAM_BYTES
    down_params = INTERMEDIATE_DIM * HIDDEN_DIM * PARAM_BYTES
    params_per_expert = gate_params + up_params + down_params

    # Memory bandwidth: read parameters for activated experts only (reused across batch)
    expert_params_total = ACTIVATED_EXPERTS_PER_NODE * params_per_expert

    # Gating network parameters (reused across batch)
    gate_network_params = HIDDEN_DIM * TOTAL_EXPERTS * PARAM_BYTES // NUM_NODES

    memory_per_forward_pass = expert_params_total + gate_network_params

    # Activations for all tokens processed: input, intermediate outputs
    expert_activations = tokens_processed * (
        HIDDEN_DIM +  # Input
        INTERMEDIATE_DIM +  # Gate output
        INTERMEDIATE_DIM +  # Up output
        HIDDEN_DIM  # Down output
    ) * ACTIVATION_BYTES * ACTIVATED_EXPERTS_PER_NODE

    memory_per_forward_pass += expert_activations

    # FLOPs per activated expert for all tokens processed
    flops_per_expert_per_token = batch_size * (
        HIDDEN_DIM * INTERMEDIATE_DIM +  # Gate
        HIDDEN_DIM * INTERMEDIATE_DIM +  # Up
        INTERMEDIATE_DIM * HIDDEN_DIM +  # Down
        INTERMEDIATE_DIM  # SwiGLU activation
    )

    # Total FLOPs for forward pass
    if is_prefill:
        flops_per_forward_pass = ACTIVATED_EXPERTS_PER_NODE * flops_per_expert_per_token * context_length
    else:
        flops_per_forward_pass = ACTIVATED_EXPERTS_PER_NODE * flops_per_expert_per_token

    # Gating network FLOPs for all tokens processed
    gate_flops_per_token = batch_size * HIDDEN_DIM * (TOTAL_EXPERTS // NUM_NODES)
    if is_prefill:
        gate_flops = gate_flops_per_token * context_length
    else:
        gate_flops = gate_flops_per_token

    flops_per_forward_pass += gate_flops

    # Interconnect: Expert routing decisions and output aggregation for all tokens processed
    routing_bytes_per_token = TOTAL_EXPERTS // 8  # Routing decisions (bits to bytes)
    output_bytes_per_token = batch_size * HIDDEN_DIM * ACTIVATION_BYTES

    interconnect_per_forward_pass = (NUM_NODES - 1) * tokens_processed * (routing_bytes_per_token + output_bytes_per_token)

    # No KV cache for MLP
    kv_cache_bytes = 0.0

    return memory_per_forward_pass, interconnect_per_forward_pass, flops_per_forward_pass, kv_cache_bytes


def estimate_reqs_transformer_layer(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for a complete transformer layer (attention + MoE MLP) per forward pass.

    Args:
        batch_size: Number of sequences in batch
        context_length: Sequence length in tokens
        is_prefill: If True, assume empty KV cache (prefill). If False, assume populated cache (generation)

    Returns:
        (memory_bus_bytes_per_forward_pass, interconnect_bytes_per_forward_pass,
         floating_point_ops_per_forward_pass, max_kv_cache_bytes)
    """
    # Get attention requirements
    attn_memory, attn_interconnect, attn_flops, attn_kv_cache = estimate_reqs_attention(
        batch_size, context_length, is_prefill)

    # Get MoE MLP requirements
    mlp_memory, mlp_interconnect, mlp_flops, mlp_kv_cache = estimate_reqs_moe_mlp(
        batch_size, context_length, is_prefill)

    # Layer norms: 2 per layer, minimal cost
    tokens_processed = (context_length * batch_size) if is_prefill else batch_size
    layernorm_params = 2 * HIDDEN_DIM * PARAM_BYTES
    layernorm_flops = 2 * tokens_processed * HIDDEN_DIM

    # Combine requirements
    total_memory = attn_memory + mlp_memory + layernorm_params
    total_interconnect = attn_interconnect + mlp_interconnect
    total_flops = attn_flops + mlp_flops + layernorm_flops
    total_kv_cache = attn_kv_cache + mlp_kv_cache

    return total_memory, total_interconnect, total_flops, total_kv_cache


def estimate_reqs_qwen3(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for complete Qwen3-235B-A22B model per forward pass.

    Args:
        batch_size: Number of sequences in batch
        context_length: Sequence length in tokens
        is_prefill: If True, assume empty KV cache (prefill). If False, assume populated cache (generation)

    Returns:
        (memory_bus_bytes_per_forward_pass, interconnect_bytes_per_forward_pass,
         floating_point_ops_per_forward_pass, max_kv_cache_bytes)
    """
    tokens_processed = (context_length * batch_size) if is_prefill else batch_size

    # Embedding layer
    emb_memory, emb_interconnect, emb_flops, emb_kv_cache = estimate_reqs_embedding(
        batch_size, context_length, is_prefill)

    # All transformer layers
    layer_memory, layer_interconnect, layer_flops, layer_kv_cache = estimate_reqs_transformer_layer(
        batch_size, context_length, is_prefill)

    # Output projection (language modeling head) - parameters reused across batch
    output_memory = HIDDEN_DIM * VOCAB_SIZE * PARAM_BYTES // NUM_NODES
    output_flops = tokens_processed * HIDDEN_DIM * (VOCAB_SIZE // NUM_NODES)
    output_interconnect = (NUM_NODES - 1) * tokens_processed * (VOCAB_SIZE // NUM_NODES) * ACTIVATION_BYTES

    # Combine all components
    total_memory = emb_memory + NUM_LAYERS * layer_memory + output_memory
    total_interconnect = emb_interconnect + NUM_LAYERS * layer_interconnect + output_interconnect
    total_flops = emb_flops + NUM_LAYERS * layer_flops + output_flops
    total_kv_cache = emb_kv_cache + NUM_LAYERS * layer_kv_cache

    return total_memory, total_interconnect, total_flops, total_kv_cache


def breakdown_memory_bandwidth(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Dict[str, Any]:
    """
    Provide detailed breakdown of where memory bandwidth is being used per forward pass.

    Args:
        batch_size: Number of sequences in batch
        context_length: Sequence length in tokens
        is_prefill: If True, assume empty KV cache (prefill). If False, assume populated cache (generation)

    Returns:
        Dict with detailed breakdown of memory usage by component per forward pass
    """
    tokens_processed = (context_length * batch_size) if is_prefill else batch_size

    breakdown = {
        'mode': 'prefill' if is_prefill else 'generation',
        'batch_size': batch_size,
        'context_length': context_length,
        'tokens_processed': tokens_processed,
        'components': {}
    }

    # Embedding
    emb_memory, _, _, _ = estimate_reqs_embedding(batch_size, context_length, is_prefill)
    breakdown['components']['embedding'] = {
        'total_bytes_per_forward_pass': emb_memory,
        'details': {
            'embedding_lookup': HIDDEN_DIM * PARAM_BYTES * tokens_processed
        }
    }

    # Single attention layer breakdown
    attn_memory, _, _, _ = estimate_reqs_attention(batch_size, context_length, is_prefill)

    q_params = (HIDDEN_DIM * HIDDEN_DIM // NUM_NODES) * PARAM_BYTES
    kv_params = (HIDDEN_DIM * (HIDDEN_DIM // NUM_Q_HEADS * NUM_KV_HEADS) // NUM_NODES) * PARAM_BYTES
    o_params = (HIDDEN_DIM * HIDDEN_DIM // NUM_NODES) * PARAM_BYTES

    q_activations = tokens_processed * (HIDDEN_DIM // NUM_NODES) * ACTIVATION_BYTES
    kv_activations = tokens_processed * (HIDDEN_DIM // NUM_Q_HEADS * NUM_KV_HEADS // NUM_NODES) * ACTIVATION_BYTES
    attention_output = tokens_processed * (HIDDEN_DIM // NUM_NODES) * ACTIVATION_BYTES

    # KV cache reads
    kv_cache_reads = 0.0
    if not is_prefill and context_length > 0:
        kv_heads_per_node = NUM_KV_HEADS // NUM_NODES
        kv_cache_reads = batch_size * (context_length + 1 - 1) * kv_heads_per_node * HEAD_DIM * ACTIVATION_BYTES * 2

    breakdown['components']['attention_per_layer'] = {
        'total_bytes_per_forward_pass': attn_memory,
        'details': {
            'q_params': q_params,
            'kv_params': kv_params,
            'output_params': o_params,
            'q_activations': q_activations,
            'kv_activations': kv_activations,
            'attention_output': attention_output,
            'kv_cache_reads': kv_cache_reads,
            'layernorm_params': HIDDEN_DIM * PARAM_BYTES
        }
    }

    # Single MoE MLP layer breakdown
    mlp_memory, _, _, _ = estimate_reqs_moe_mlp(batch_size, context_length, is_prefill)

    params_per_expert = HIDDEN_DIM * INTERMEDIATE_DIM * 3 * PARAM_BYTES  # gate + up + down
    expert_params_total = ACTIVATED_EXPERTS_PER_NODE * params_per_expert
    gate_network_params = HIDDEN_DIM * TOTAL_EXPERTS * PARAM_BYTES // NUM_NODES

    expert_activations = tokens_processed * (HIDDEN_DIM + 2*INTERMEDIATE_DIM + HIDDEN_DIM) * ACTIVATION_BYTES * ACTIVATED_EXPERTS_PER_NODE

    breakdown['components']['moe_mlp_per_layer'] = {
        'total_bytes_per_forward_pass': mlp_memory,
        'details': {
            'expert_params': expert_params_total,
            'gate_network_params': gate_network_params,
            'expert_activations': expert_activations,
            'layernorm_params': HIDDEN_DIM * PARAM_BYTES
        }
    }

    # Output layer
    output_memory = HIDDEN_DIM * VOCAB_SIZE * PARAM_BYTES // NUM_NODES
    breakdown['components']['output_projection'] = {
        'total_bytes_per_forward_pass': output_memory,
        'details': {
            'lm_head_params': output_memory
        }
    }

    # Total per layer
    layer_memory, _, _, _ = estimate_reqs_transformer_layer(batch_size, context_length, is_prefill)
    breakdown['per_layer_total'] = layer_memory

    # Full model
    full_memory, _, _, _ = estimate_reqs_qwen3(batch_size, context_length, is_prefill)
    breakdown['full_model_total'] = full_memory
    breakdown['total_layers'] = NUM_LAYERS

    # Summary
    breakdown['summary'] = {
        'embedding_fraction': emb_memory / full_memory,
        'attention_fraction': (attn_memory * NUM_LAYERS) / full_memory,
        'mlp_fraction': (mlp_memory * NUM_LAYERS) / full_memory,
        'output_fraction': output_memory / full_memory,
        'parameters_vs_activations': {
            'param_bytes_per_forward_pass': (emb_memory +
                                           (q_params + kv_params + o_params + gate_network_params + expert_params_total + 2*HIDDEN_DIM*PARAM_BYTES) * NUM_LAYERS +
                                           output_memory),
            'activation_bytes_per_forward_pass': ((q_activations + kv_activations + attention_output + expert_activations) * NUM_LAYERS),
            'kv_cache_reads_per_forward_pass': kv_cache_reads * NUM_LAYERS
        }
    }

    return breakdown


if __name__ == "__main__":
    # Example usage with breakdown
    batch_size = 4
    context_length = 4096

    print("=== PREFILL MODE ===")
    prefill_breakdown = breakdown_memory_bandwidth(batch_size, context_length, is_prefill=True)
    memory_bw, interconnect_bw, flops, kv_cache = estimate_reqs_qwen3(batch_size, context_length, is_prefill=True)

    print(f"Batch size: {batch_size}, Context length: {context_length}")
    print(f"Tokens processed: {prefill_breakdown['tokens_processed']}")
    print(f"Memory bandwidth per forward pass: {memory_bw / 1e9:.2f} GB")
    print(f"Interconnect bandwidth per forward pass: {interconnect_bw / 1e9:.2f} GB")
    print(f"FLOPs per forward pass: {flops / 1e12:.2f} TFLOPs")
    print(f"KV cache per node: {kv_cache / 1e9:.2f} GB")

    print(f"\nBreakdown:")
    for component, data in prefill_breakdown['components'].items():
        print(f"  {component}: {data['total_bytes_per_forward_pass'] / 1e9:.2f} GB")

    print("\n" + "="*50)
    print("=== GENERATION MODE ===")
    gen_breakdown = breakdown_memory_bandwidth(batch_size, context_length, is_prefill=False)
    memory_bw, interconnect_bw, flops, kv_cache = estimate_reqs_qwen3(batch_size, context_length, is_prefill=False)

    print(f"Batch size: {batch_size}, Context length: {context_length}")
    print(f"Tokens processed: {gen_breakdown['tokens_processed']}")
    print(f"Memory bandwidth per forward pass: {memory_bw / 1e9:.2f} GB")
    print(f"Interconnect bandwidth per forward pass: {interconnect_bw / 1e9:.2f} GB")
    print(f"FLOPs per forward pass: {flops / 1e12:.2f} TFLOPs")
    print(f"KV cache per node: {kv_cache / 1e9:.2f} GB")

    print(f"\nBreakdown:")
    for component, data in gen_breakdown['components'].items():
        print(f"  {component}: {data['total_bytes_per_forward_pass'] / 1e9:.2f} GB")

    print("\n" + "="*50)
    print("=== SCALING COMPARISON ===")
    for bs in [1, 2, 4, 8]:
        prefill_mem, _, prefill_flops, _ = estimate_reqs_qwen3(bs, 4096, is_prefill=True)
        gen_mem, _, gen_flops, _ = estimate_reqs_qwen3(bs, 4096, is_prefill=False)

        print(f"Batch {bs}:")
        print(f"  Prefill:    {prefill_mem/1e9:.1f} GB memory, {prefill_flops/1e12:.1f} TFLOPs")
        print(f"  Generation: {gen_mem/1e9:.1f} GB memory, {gen_flops/1e12:.1f} TFLOPs")
