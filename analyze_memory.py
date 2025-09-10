"""
Analyzes the memory bandwidth contributions for different scenarios.
"""

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

if __name__ == "__main__":
    analyze_memory_breakdown()
