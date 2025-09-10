"""
Generates plots for Qwen3 performance metrics vs. batch size.
"""

import numpy as np
import matplotlib.pyplot as plt
from requirements import estimate_reqs_qwen3

def generate_plots():
    """
    Generates and saves plots for memory, interconnect, and FLOPs vs. batch size.
    """
    batch_sizes = [2**i for i in range(11)]  # 1, 2, 4, ..., 1024
    context_length = 128 * 1024

    # Data storage
    prefill_mem = []
    prefill_interconnect = []
    prefill_flops = []
    gen_mem = []
    gen_interconnect = []
    gen_flops = []

    for bs in batch_sizes:
        # --- Prefill calculations ---
        mem_pf, interconnect_pf, flops_pf, _ = estimate_reqs_qwen3(
            bs, context_length, is_prefill=True
        )
        prefill_mem.append(mem_pf)
        prefill_interconnect.append(interconnect_pf)
        prefill_flops.append(flops_pf)

        # --- Generation calculations ---
        mem_gen, interconnect_gen, flops_gen, _ = estimate_reqs_qwen3(
            bs, context_length, is_prefill=False
        )
        gen_mem.append(mem_gen)
        gen_interconnect.append(interconnect_gen)
        gen_flops.append(flops_gen)

    # --- Plotting ---
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Qwen3-235B-A22B Performance Estimates per Forward Pass (Context Length: {context_length})', fontsize=16)

    # --- Prefill Row ---
    # Memory
    axs[0, 0].plot(batch_sizes, prefill_mem, 'o-')
    axs[0, 0].set_title('Prefill - Memory Bus')
    axs[0, 0].set_ylabel('Bytes / Forward Pass')
    axs[0, 0].set_xscale('log', base=2)
    axs[0, 0].grid(True, which="both", ls="--")

    # Interconnect
    axs[0, 1].plot(batch_sizes, prefill_interconnect, 'o-')
    axs[0, 1].set_title('Prefill - Interconnect')
    axs[0, 1].set_xscale('log', base=2)
    axs[0, 1].grid(True, which="both", ls="--")

    # FLOPs
    axs[0, 2].plot(batch_sizes, prefill_flops, 'o-')
    axs[0, 2].set_title('Prefill - FLOPs')
    axs[0, 2].set_xscale('log', base=2)
    axs[0, 2].grid(True, which="both", ls="--")

    # --- Generation Row ---
    # Memory
    axs[1, 0].plot(batch_sizes, gen_mem, 's-')
    axs[1, 0].set_title('Generation - Memory Bus')
    axs[1, 0].set_xlabel('Batch Size')
    axs[1, 0].set_ylabel('Bytes / Forward Pass')
    axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].grid(True, which="both", ls="--")

    # Interconnect
    axs[1, 1].plot(batch_sizes, gen_interconnect, 's-')
    axs[1, 1].set_title('Generation - Interconnect')
    axs[1, 1].set_xlabel('Batch Size')
    axs[1, 1].set_xscale('log', base=2)
    axs[1, 1].grid(True, which="both", ls="--")

    # FLOPs
    axs[1, 2].plot(batch_sizes, gen_flops, 's-')
    axs[1, 2].set_title('Generation - FLOPs')
    axs[1, 2].set_xlabel('Batch Size')
    axs[1, 2].set_xscale('log', base=2)
    axs[1, 2].grid(True, which="both", ls="--")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    generate_plots()
