"""
Analyzes the memory bandwidth contributions for different scenarios.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from requirements import breakdown_memory_bandwidth, NUM_LAYERS

def analyze_memory_breakdown():
    """
    Calculates and prints a breakdown of memory usage for four key scenarios.
    """
    scenarios = {
        "Small Batch, Small Context": (1, 1024),
        "Small Batch, Large Context": (1, 128 * 1024),
        "Large Batch, Small Context": (1024, 1024),
        "Large Batch, Large Context": (1024, 128 * 1024),
    }

    print("="*80)
    print("Memory Bandwidth Breakdown Analysis")
    print("="*80)

    for name, (bs, ctx) in scenarios.items():
        print(f"\n--- Scenario: {name} (Batch: {bs}, Context: {ctx}) ---")

        # --- Generation Mode ---
        print("\n  ** Generation Mode **")
        try:
            gen_breakdown = breakdown_memory_bandwidth(bs, ctx, is_prefill=False)
            summary = gen_breakdown['summary']['parameters_vs_activations']
            total_mem = gen_breakdown['full_model_total']

            param_bytes = summary['param_bytes_per_forward_pass']
            act_bytes = summary['activation_bytes_per_forward_pass']
            kv_read_bytes = summary['kv_cache_reads_per_forward_pass']

            if total_mem > 0:
                print(f"    Total Memory per Pass: {total_mem / 1e9:.2f} GB")
                print(f"      - Parameter Reads:     {param_bytes / 1e9:.2f} GB ({param_bytes/total_mem:.1%})")
                print(f"      - Activation R/W:      {act_bytes / 1e9:.2f} GB ({act_bytes/total_mem:.1%})")
                print(f"      - KV Cache Reads:      {kv_read_bytes / 1e9:.2f} GB ({kv_read_bytes/total_mem:.1%})")
            else:
                print("    No memory usage.")

        except Exception as e:
            print(f"    Error calculating generation breakdown: {e}")


        # --- Prefill Mode ---
        print("\n  ** Prefill Mode **")
        try:
            prefill_breakdown = breakdown_memory_bandwidth(bs, ctx, is_prefill=True)
            summary = prefill_breakdown['summary']['parameters_vs_activations']
            total_mem = prefill_breakdown['full_model_total']

            param_bytes = summary['param_bytes_per_forward_pass']
            act_bytes = summary['activation_bytes_per_forward_pass']
            kv_read_bytes = summary['kv_cache_reads_per_forward_pass'] # Should be 0

            if total_mem > 0:
                print(f"    Total Memory per Pass: {total_mem / 1e9:.2f} GB")
                print(f"      - Parameter Reads:     {param_bytes / 1e9:.2f} GB ({param_bytes/total_mem:.1%})")
                print(f"      - Activation R/W:      {act_bytes / 1e9:.2f} GB ({act_bytes/total_mem:.1%})")
                print(f"      - KV Cache Reads:      {kv_read_bytes / 1e9:.2f} GB ({kv_read_bytes/total_mem:.1%})")
            else:
                print("    No memory usage.")

        except Exception as e:
            print(f"    Error calculating prefill breakdown: {e}")

        print("-" * 80)

def plot_generation_memory_bandwidth_vs_context():
    """
    Generates and saves a stacked area chart of memory bandwidth vs. context length for four batch sizes.
    """
    context_lengths = [2**i for i in range(0, 18)]  # 1 to 128*1024
    batch_sizes = [1, 8, 64, 256]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True)
    fig.suptitle('Generation Phase Memory Bandwidth vs Context Length', fontsize=16)
    axes = axes.flatten()

    for i, bs in enumerate(batch_sizes):
        data = []
        for ctx in context_lengths:
            breakdown = breakdown_memory_bandwidth(batch_size=bs, context_length=ctx, is_prefill=False)
            summary = breakdown['summary']['parameters_vs_activations']

            param_bytes = summary['param_bytes_per_forward_pass']
            act_bytes = summary['activation_bytes_per_forward_pass']
            kv_read_bytes = summary['kv_cache_reads_per_forward_pass']

            data.append({
                'context_length': ctx,
                'Parameter Reads': param_bytes,
                'Activation R/W': act_bytes,
                'KV Cache Reads': kv_read_bytes
            })

        df = pd.DataFrame(data)
        df.set_index('context_length', inplace=True)
        df_gb = df / 1e9

        ax = axes[i]
        ax.stackplot(df_gb.index, df_gb['Parameter Reads'], df_gb['Activation R/W'], df_gb['KV Cache Reads'],
                     labels=['Parameter Reads', 'Activation R/W', 'KV Cache Reads'],
                     alpha=0.8)

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Context Length (Tokens)')
        ax.set_ylabel('Memory Bandwidth per Batch (GB)')
        ax.set_title(f'Batch Size = {bs}')
        ax.grid(True, which="both", ls="--")
        ax.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('generation_memory_bandwidth_vs_context.png')
    print("\nChart saved to 'generation_memory_bandwidth_vs_context.png'")



if __name__ == "__main__":
    analyze_memory_breakdown()
    plot_generation_memory_bandwidth_vs_context()
