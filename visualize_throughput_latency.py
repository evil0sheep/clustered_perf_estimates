"""
Generates plots for cluster throughput and time-to-first-token latency.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from requirements import estimate_reqs_qwen3

# Hardware constants from README
MEMORY_BANDWIDTH_PER_NODE = 256e9  # 256 GB/s
INTERCONNECT_BANDWIDTH_PER_NODE = 189e9 / 8  # 189 gbps -> 23.625 GB/s
FLOPS_PER_NODE = 126e12  # 126 TOPS

def generate_performance_plots():
    """
    Generates and displays plots for cluster throughput, latency, and utilization.
    """
    # --- Scenarios ---
    gen_batch_sizes = [2**i for i in range(11)]  # 1, 2, 4, ..., 1024
    gen_context_lengths = [2**10, 2**12, 2**14, 2**16] # 1k, 4k, 16k, 64k

    prefill_prompt_lengths = [2**i for i in range(18)] # 1, 2, 4, ..., 128k
    prefill_batch_sizes = [1, 8, 64, 256]

    # --- Data Calculation ---
    gen_perf_data = {}
    gen_util_data = {}
    for ctx in gen_context_lengths:
        throughputs = []
        utils_mem, utils_inter, utils_flops = [], [], []
        for bs in gen_batch_sizes:
            mem, inter, flops, _ = estimate_reqs_qwen3(bs, ctx, is_prefill=False)
            t_mem = mem / MEMORY_BANDWIDTH_PER_NODE
            t_inter = inter / INTERCONNECT_BANDWIDTH_PER_NODE
            t_flops = flops / FLOPS_PER_NODE
            t_pass = max(t_mem, t_inter, t_flops)

            throughputs.append(1.0 / t_pass if t_pass > 0 else 0)
            if t_pass > 0:
                utils_mem.append(t_mem / t_pass)
                utils_inter.append(t_inter / t_pass)
                utils_flops.append(t_flops / t_pass)
            else:
                utils_mem.append(0); utils_inter.append(0); utils_flops.append(0)
        gen_perf_data[ctx] = throughputs
        gen_util_data[ctx] = {'mem': utils_mem, 'inter': utils_inter, 'flops': utils_flops}

    prefill_perf_data = {}
    prefill_util_data = {}
    for bs in prefill_batch_sizes:
        latencies = []
        utils_mem, utils_inter, utils_flops = [], [], []
        for p_len in prefill_prompt_lengths:
            mem, inter, flops, _ = estimate_reqs_qwen3(bs, p_len, is_prefill=True)
            t_mem = mem / MEMORY_BANDWIDTH_PER_NODE
            t_inter = inter / INTERCONNECT_BANDWIDTH_PER_NODE
            t_flops = flops / FLOPS_PER_NODE
            t_pass = max(t_mem, t_inter, t_flops)

            latencies.append(t_pass * 1000) # ms
            if t_pass > 0:
                utils_mem.append(t_mem / t_pass)
                utils_inter.append(t_inter / t_pass)
                utils_flops.append(t_flops / t_pass)
            else:
                utils_mem.append(0); utils_inter.append(0); utils_flops.append(0)
        prefill_perf_data[bs] = latencies
        prefill_util_data[bs] = {'mem': utils_mem, 'inter': utils_inter, 'flops': utils_flops}

    # --- Plotting ---
    fig, axs = plt.subplots(3, 2, figsize=(18, 16), sharex='col')

    # Main Title and Subtitle
    fig.suptitle('Theoretical Cluster Performance and Bottleneck Analysis', fontsize=20)
    subtitle = (
        "Model: Qwen3-235B-A22B (235B Total, 22B Active, 8-bit) | "
        "Cluster: 4 Nodes, Fully Connected Mesh\n"
        "Per Node: 256 GB/s Memory BW, 189 gbps Interconnect BW, 126 TOPS AI"
    )
    fig.text(0.5, 0.92, subtitle, ha='center', fontsize=12)


    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = {'mem': '-', 'inter': '--', 'flops': ':'}
    label_fontsize = 12
    tick_fontsize = 10

    # == Column 1: Generation Throughput ==
    axs[0, 0].set_title('Generation Throughput', fontsize=14)
    axs[0, 0].set_ylabel('Batches per Second', fontsize=label_fontsize)
    axs[0, 0].set_yscale('log')
    for i, (ctx, throughputs) in enumerate(gen_perf_data.items()):
        axs[0, 0].plot(gen_batch_sizes, throughputs, color=colors[i], label=f'Ctx: {ctx//1024}k')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="--")
    axs[0, 0].tick_params(axis='y', labelsize=tick_fontsize)


    # Utilization for smallest context
    ctx_small = gen_context_lengths[0]
    utils_small = gen_util_data[ctx_small]
    axs[1, 0].set_title(f'Resource Utilization (Ctx: {ctx_small//1024}k)', fontsize=14)
    axs[1, 0].set_ylabel('Utilization (%)', fontsize=label_fontsize)
    axs[1, 0].plot(gen_batch_sizes, utils_small['mem'], color=colors[0], linestyle=linestyles['mem'], label='Memory')
    axs[1, 0].plot(gen_batch_sizes, utils_small['inter'], color=colors[0], linestyle=linestyles['inter'], label='Interconnect')
    axs[1, 0].plot(gen_batch_sizes, utils_small['flops'], color=colors[0], linestyle=linestyles['flops'], label='FLOPs')
    axs[1, 0].set_ylim(0, 1.1)
    axs[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axs[1, 0].grid(True, which="both", ls="--")
    axs[1, 0].legend()
    axs[1, 0].tick_params(axis='y', labelsize=tick_fontsize)

    # Utilization for largest context
    ctx_large = gen_context_lengths[-1]
    utils_large = gen_util_data[ctx_large]
    color_large_ctx = colors[len(gen_context_lengths) - 1]
    axs[2, 0].set_title(f'Resource Utilization (Ctx: {ctx_large//1024}k)', fontsize=14)
    axs[2, 0].set_ylabel('Utilization (%)', fontsize=label_fontsize)
    axs[2, 0].plot(gen_batch_sizes, utils_large['mem'], color=color_large_ctx, linestyle=linestyles['mem'], label='Memory')
    axs[2, 0].plot(gen_batch_sizes, utils_large['inter'], color=color_large_ctx, linestyle=linestyles['inter'], label='Interconnect')
    axs[2, 0].plot(gen_batch_sizes, utils_large['flops'], color=color_large_ctx, linestyle=linestyles['flops'], label='FLOPs')
    axs[2, 0].set_ylim(0, 1.1)
    axs[2, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axs[2, 0].grid(True, which="both", ls="--")
    axs[2, 0].legend()
    axs[2, 0].set_xlabel('Batch Size', fontsize=label_fontsize)
    axs[2, 0].set_xscale('log', base=2)
    axs[2, 0].tick_params(axis='both', labelsize=tick_fontsize)


    # == Column 2: Prefill Latency ==
    axs[0, 1].set_title('Prefill Latency (Time to First Token)', fontsize=14)
    axs[0, 1].set_ylabel('Latency (ms)', fontsize=label_fontsize)
    axs[0, 1].set_yscale('log')
    for i, (bs, latencies) in enumerate(prefill_perf_data.items()):
        axs[0, 1].plot(prefill_prompt_lengths, latencies, color=colors[i], label=f'Batch: {bs}')
    axs[0, 1].legend()
    axs[0, 1].grid(True, which="both", ls="--")
    axs[0, 1].tick_params(axis='y', labelsize=tick_fontsize)

    # Utilization for smallest batch size
    bs_small = prefill_batch_sizes[0]
    utils_small = prefill_util_data[bs_small]
    axs[1, 1].set_title(f'Resource Utilization (Batch: {bs_small})', fontsize=14)
    axs[1, 1].set_ylabel('Utilization (%)', fontsize=label_fontsize)
    axs[1, 1].plot(prefill_prompt_lengths, utils_small['mem'], color=colors[0], linestyle=linestyles['mem'], label='Memory')
    axs[1, 1].plot(prefill_prompt_lengths, utils_small['inter'], color=colors[0], linestyle=linestyles['inter'], label='Interconnect')
    axs[1, 1].plot(prefill_prompt_lengths, utils_small['flops'], color=colors[0], linestyle=linestyles['flops'], label='FLOPs')
    axs[1, 1].set_ylim(0, 1.1)
    axs[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axs[1, 1].grid(True, which="both", ls="--")
    axs[1, 1].legend()
    axs[1, 1].tick_params(axis='y', labelsize=tick_fontsize)

    # Utilization for largest batch size
    bs_large = prefill_batch_sizes[-1]
    utils_large = prefill_util_data[bs_large]
    color_large_bs = colors[len(prefill_batch_sizes) - 1]
    axs[2, 1].set_title(f'Resource Utilization (Batch: {bs_large})', fontsize=14)
    axs[2, 1].set_ylabel('Utilization (%)', fontsize=label_fontsize)
    axs[2, 1].plot(prefill_prompt_lengths, utils_large['mem'], color=color_large_bs, linestyle=linestyles['mem'], label='Memory')
    axs[2, 1].plot(prefill_prompt_lengths, utils_large['inter'], color=color_large_bs, linestyle=linestyles['inter'], label='Interconnect')
    axs[2, 1].plot(prefill_prompt_lengths, utils_large['flops'], color=color_large_bs, linestyle=linestyles['flops'], label='FLOPs')
    axs[2, 1].set_ylim(0, 1.1)
    axs[2, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axs[2, 1].grid(True, which="both", ls="--")
    axs[2, 1].legend()
    axs[2, 1].set_xlabel('Prompt Length (Tokens)', fontsize=label_fontsize)
    axs[2, 1].set_xscale('log', base=2)
    axs[2, 1].tick_params(axis='both', labelsize=tick_fontsize)


    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig('cluster_performance.png')
    plt.show()

if __name__ == "__main__":
    generate_performance_plots()
