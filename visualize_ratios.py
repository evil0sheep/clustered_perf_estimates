"""
Generates plots for the ratio of memory to interconnect bandwidth requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
from requirements import estimate_reqs_qwen3

def generate_ratio_plots():
    """
    Generates and displays plots of the memory-to-interconnect bandwidth ratio.
    """
    batch_sizes = [2**i for i in range(11)]  # 1, 2, 4, ..., 1024
    context_lengths = [1024 * (4**i) for i in range(5)]  # 1k, 4k, 16k, 64k, 256k -> up to 128k as per model
    context_lengths = [ctx for ctx in context_lengths if ctx <= 128 * 1024] # ensure we don't exceed model max

    # Data storage
    prefill_ratios_by_ctx = {}
    gen_ratios_by_ctx = {}

    for ctx in context_lengths:
        prefill_ratios = []
        gen_ratios = []
        for bs in batch_sizes:
            # --- Prefill calculations ---
            mem_pf, interconnect_pf, _, _ = estimate_reqs_qwen3(
                bs, ctx, is_prefill=True
            )
            # Avoid division by zero if interconnect is ever zero
            prefill_ratios.append(mem_pf / interconnect_pf if interconnect_pf else float('inf'))

            # --- Generation calculations ---
            mem_gen, interconnect_gen, _, _ = estimate_reqs_qwen3(
                bs, ctx, is_prefill=False
            )
            gen_ratios.append(mem_gen / interconnect_gen if interconnect_gen else float('inf'))

        prefill_ratios_by_ctx[ctx] = prefill_ratios
        gen_ratios_by_ctx[ctx] = gen_ratios


    # --- Plotting ---
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Ratio of Memory Bandwidth to Interconnect Bandwidth Requirements', fontsize=16)

    # Prefill Plot
    for ctx, ratios in prefill_ratios_by_ctx.items():
        axs[0].plot(batch_sizes, ratios, 'o-', label=f'Ctx: {ctx // 1024}k')
    axs[0].set_title('Prefill')
    axs[0].set_xlabel('Batch Size')
    axs[0].set_ylabel('Memory BW / Interconnect BW Ratio')
    axs[0].set_xscale('log', base=2)
    axs[0].grid(True, which="both", ls="--")
    axs[0].legend()

    # Generation Plot
    for ctx, ratios in gen_ratios_by_ctx.items():
        axs[1].plot(batch_sizes, ratios, 's-', label=f'Ctx: {ctx // 1024}k')
    axs[1].set_title('Generation')
    axs[1].set_xlabel('Batch Size')
    axs[1].set_xscale('log', base=2)
    axs[1].grid(True, which="both", ls="--")
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    generate_ratio_plots()
