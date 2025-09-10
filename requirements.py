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


def _estimate_reqs_attention_pass(batch_size: int, num_new_tokens: int, kv_cache_size: int) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for a single GQA attention block for a single forward pass.

    Args:
        batch_size: Number of sequences in batch
        num_new_tokens: Number of new tokens being processed in this pass
        kv_cache_size: Number of tokens already in the KV cache

    Returns:
        (memory_bus_bytes, interconnect_bytes, floating_point_ops, max_kv_cache_bytes)
    """
    tokens_processed = batch_size * num_new_tokens
    effective_seq_len = kv_cache_size + num_new_tokens

    # Parameter sizes (per node)
    q_params = (HIDDEN_DIM * HIDDEN_DIM // NUM_NODES) * PARAM_BYTES
    kv_params = (HIDDEN_DIM * (HIDDEN_DIM // NUM_Q_HEADS * NUM_KV_HEADS) // NUM_NODES) * PARAM_BYTES
    o_params = (HIDDEN_DIM * HIDDEN_DIM // NUM_NODES) * PARAM_BYTES
    memory_per_forward_pass = q_params + kv_params + o_params

    # Activation memory reads/writes
    q_activations = tokens_processed * (HIDDEN_DIM // NUM_NODES) * ACTIVATION_BYTES
    kv_activations = tokens_processed * (HIDDEN_DIM // NUM_Q_HEADS * NUM_KV_HEADS // NUM_NODES) * ACTIVATION_BYTES
    attention_output = tokens_processed * (HIDDEN_DIM // NUM_NODES) * ACTIVATION_BYTES
    memory_per_forward_pass += q_activations + kv_activations + attention_output

    # KV cache reads
    if kv_cache_size > 0:
        kv_heads_per_node = NUM_KV_HEADS // NUM_NODES
        kv_cache_read = batch_size * kv_cache_size * kv_heads_per_node * HEAD_DIM * ACTIVATION_BYTES * 2  # K + V
        memory_per_forward_pass += kv_cache_read

    # FLOPs (per node)
    heads_per_node = NUM_Q_HEADS // NUM_NODES
    kv_heads_per_node = NUM_KV_HEADS // NUM_NODES
    q_proj_flops = tokens_processed * 2 * HIDDEN_DIM * (heads_per_node * HEAD_DIM)
    k_proj_flops = tokens_processed * 2 * HIDDEN_DIM * (kv_heads_per_node * HEAD_DIM)
    v_proj_flops = tokens_processed * 2 * HIDDEN_DIM * (kv_heads_per_node * HEAD_DIM)
    qk_flops = tokens_processed * 2 * heads_per_node * effective_seq_len * HEAD_DIM
    av_flops = tokens_processed * 2 * heads_per_node * effective_seq_len * HEAD_DIM
    o_proj_flops = tokens_processed * 2 * HIDDEN_DIM * (HIDDEN_DIM // NUM_NODES)
    flops_per_forward_pass = q_proj_flops + k_proj_flops + v_proj_flops + qk_flops + av_flops + o_proj_flops

    # Interconnect
    interconnect_per_forward_pass = (NUM_NODES - 1) * tokens_processed * (HIDDEN_DIM // NUM_NODES) * ACTIVATION_BYTES

    # KV cache size
    kv_heads_per_node = NUM_KV_HEADS // NUM_NODES
    kv_cache_per_head = batch_size * effective_seq_len * HEAD_DIM * ACTIVATION_BYTES
    kv_cache_bytes = 2 * kv_heads_per_node * kv_cache_per_head

    return memory_per_forward_pass, interconnect_per_forward_pass, flops_per_forward_pass, kv_cache_bytes

def _estimate_reqs_moe_mlp_pass(batch_size: int, num_new_tokens: int) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for MoE MLP block for a single forward pass.
    """
    tokens_processed = batch_size * num_new_tokens

    # Parameter memory
    params_per_expert = (HIDDEN_DIM * INTERMEDIATE_DIM * 3) * PARAM_BYTES
    expert_params_total = ACTIVATED_EXPERTS_PER_NODE * params_per_expert
    gate_network_params = (HIDDEN_DIM * TOTAL_EXPERTS // NUM_NODES) * PARAM_BYTES
    memory_per_forward_pass = expert_params_total + gate_network_params

    # Activation memory
    input_activation_bytes = tokens_processed * HIDDEN_DIM * ACTIVATION_BYTES
    intermediate_activation_bytes = tokens_processed * ACTIVATED_EXPERTS_PER_NODE * (INTERMEDIATE_DIM * 3) * ACTIVATION_BYTES
    output_activation_bytes = tokens_processed * HIDDEN_DIM * ACTIVATION_BYTES
    expert_activations = input_activation_bytes + intermediate_activation_bytes + output_activation_bytes
    memory_per_forward_pass += expert_activations

    # FLOPs
    gate_flops = tokens_processed * 2 * HIDDEN_DIM * (TOTAL_EXPERTS // NUM_NODES)
    flops_per_expert = tokens_processed * (
        2 * HIDDEN_DIM * INTERMEDIATE_DIM +
        2 * HIDDEN_DIM * INTERMEDIATE_DIM +
        INTERMEDIATE_DIM +
        2 * INTERMEDIATE_DIM * HIDDEN_DIM
    )
    expert_flops = ACTIVATED_EXPERTS_PER_NODE * flops_per_expert
    flops_per_forward_pass = gate_flops + expert_flops

    # Interconnect
    bytes_per_token = HIDDEN_DIM * ACTIVATION_BYTES
    interconnect_per_forward_pass = 2 * tokens_processed * bytes_per_token

    return memory_per_forward_pass, interconnect_per_forward_pass, flops_per_forward_pass, 0.0

def _estimate_reqs_transformer_layer_pass(batch_size: int, num_new_tokens: int, kv_cache_size: int) -> Tuple[float, float, float, float]:
    """Estimate requirements for a complete transformer layer for a single forward pass."""
    attn_mem, attn_inter, attn_flops, attn_kv = _estimate_reqs_attention_pass(batch_size, num_new_tokens, kv_cache_size)
    mlp_mem, mlp_inter, mlp_flops, _ = _estimate_reqs_moe_mlp_pass(batch_size, num_new_tokens)

    tokens_processed = batch_size * num_new_tokens
    layernorm_params = 2 * HIDDEN_DIM * PARAM_BYTES
    layernorm_flops = 2 * tokens_processed * HIDDEN_DIM

    total_memory = attn_mem + mlp_mem + layernorm_params
    total_interconnect = attn_inter + mlp_inter
    total_flops = attn_flops + mlp_flops + layernorm_flops

    return total_memory, total_interconnect, total_flops, attn_kv

def _estimate_reqs_embedding_pass(batch_size: int, num_new_tokens: int) -> Tuple[float, float, float, float]:
    """Estimate requirements for the embedding layer for a single forward pass."""
    tokens_processed = batch_size * num_new_tokens
    memory = tokens_processed * HIDDEN_DIM * PARAM_BYTES
    return memory, 0.0, 0.0, 0.0

def estimate_reqs_embedding(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for embedding layer per forward pass.
    (Wrapper for backward compatibility)
    """
    num_new_tokens = context_length if is_prefill else 1
    return _estimate_reqs_embedding_pass(batch_size, num_new_tokens)

def estimate_reqs_attention(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for a single GQA attention block per forward pass.
    (Wrapper for backward compatibility)
    """
    if is_prefill:
        return _estimate_reqs_attention_pass(batch_size, num_new_tokens=context_length, kv_cache_size=0)
    else:
        return _estimate_reqs_attention_pass(batch_size, num_new_tokens=1, kv_cache_size=context_length)

def estimate_reqs_moe_mlp(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for MoE MLP block per forward pass.
    (Wrapper for backward compatibility)
    """
    num_new_tokens = context_length if is_prefill else 1
    return _estimate_reqs_moe_mlp_pass(batch_size, num_new_tokens)

def estimate_reqs_transformer_layer(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for a complete transformer layer (attention + MoE MLP) per forward pass.
    (Wrapper for backward compatibility)
    """
    if is_prefill:
        return _estimate_reqs_transformer_layer_pass(batch_size, num_new_tokens=context_length, kv_cache_size=0)
    else:
        return _estimate_reqs_transformer_layer_pass(batch_size, num_new_tokens=1, kv_cache_size=context_length)

def estimate_reqs_qwen3(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Tuple[float, float, float, float]:
    """
    Estimate requirements for complete Qwen3-235B-A22B model per forward pass.
    (Wrapper for backward compatibility)
    """
    if is_prefill:
        num_new_tokens = context_length
        kv_cache_size = 0
    else:
        num_new_tokens = 1
        kv_cache_size = context_length

    tokens_processed = batch_size * num_new_tokens

    # Embedding layer
    emb_mem, emb_inter, emb_flops, _ = _estimate_reqs_embedding_pass(batch_size, num_new_tokens)

    # All transformer layers
    layer_mem, layer_inter, layer_flops, layer_kv = _estimate_reqs_transformer_layer_pass(batch_size, num_new_tokens, kv_cache_size)

    # Output projection (language modeling head)
    output_mem = (HIDDEN_DIM * VOCAB_SIZE // NUM_NODES) * PARAM_BYTES
    output_flops = tokens_processed * 2 * HIDDEN_DIM * (VOCAB_SIZE // NUM_NODES)
    output_inter = (NUM_NODES - 1) * tokens_processed * (VOCAB_SIZE // NUM_NODES) * ACTIVATION_BYTES

    # Combine all components
    total_memory = emb_mem + NUM_LAYERS * layer_mem + output_mem
    total_interconnect = emb_inter + NUM_LAYERS * layer_inter + output_inter
    total_flops = emb_flops + NUM_LAYERS * layer_flops + output_flops
    total_kv_cache = NUM_LAYERS * layer_kv

    return total_memory, total_interconnect, total_flops, total_kv_cache

def estimate_reqs_qwen3_chunked_prefill(batch_size: int, context_length: int, chunk_size: int) -> Tuple[float, float, float, float]:
    """
    Estimate total requirements for a complete Qwen3 model using chunked prefill.
    """
    total_mem, total_inter, total_flops = 0.0, 0.0, 0.0
    final_kv_cache = 0.0

    for i in range(0, context_length, chunk_size):
        current_chunk_size = min(chunk_size, context_length - i)
        kv_cache_size = i

        # We only need to calculate the cost of the final layer's KV cache size
        if i + current_chunk_size >= context_length:
             _, _, _, final_kv_cache = _estimate_reqs_transformer_layer_pass(batch_size, current_chunk_size, kv_cache_size)
             final_kv_cache *= NUM_LAYERS

        # Embedding for the current chunk
        emb_mem, emb_inter, emb_flops, _ = _estimate_reqs_embedding_pass(batch_size, current_chunk_size)

        # Transformer layers for the current chunk
        layer_mem, layer_inter, layer_flops, _ = _estimate_reqs_transformer_layer_pass(batch_size, current_chunk_size, kv_cache_size)

        # Output projection for the current chunk
        tokens_processed = batch_size * current_chunk_size
        output_mem = (HIDDEN_DIM * VOCAB_SIZE // NUM_NODES) * PARAM_BYTES
        output_flops = tokens_processed * 2 * HIDDEN_DIM * (VOCAB_SIZE // NUM_NODES)
        output_inter = (NUM_NODES - 1) * tokens_processed * (VOCAB_SIZE // NUM_NODES) * ACTIVATION_BYTES

        # Accumulate totals
        total_mem += emb_mem + NUM_LAYERS * layer_mem + output_mem
        total_inter += emb_inter + NUM_LAYERS * layer_inter + output_inter
        total_flops += emb_flops + NUM_LAYERS * layer_flops + output_flops

    return total_mem, total_inter, total_flops, final_kv_cache

def breakdown_memory_bandwidth(batch_size: int, context_length: int = 128 * 1024, is_prefill: bool = False) -> Dict[str, Any]:
    """
    Provide detailed breakdown of where memory bandwidth is being used per forward pass.
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

    print("\n" + "="*50 + "\n")
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

    print("\n" + "="*50 + "\n")
    print("=== SCALING COMPARISON ===")
    for bs in [1, 2, 4, 8]:
        prefill_mem, _, prefill_flops, _ = estimate_reqs_qwen3(bs, 4096, is_prefill=True)
        gen_mem, _, gen_flops, _ = estimate_reqs_qwen3(bs, 4096, is_prefill=False)

        print(f"Batch {bs}:")
        print(f"  Prefill:    {prefill_mem/1e9:.1f} GB memory, {prefill_flops/1e12:.1f} TFLOPs")
        print(f"  Generation: {gen_mem/1e9:.1f} GB memory, {gen_flops/1e12:.1f} TFLOPs")
