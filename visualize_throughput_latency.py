"""
Generates plots for cluster throughput and time-to-first-token latency.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, LogLocator
from matplotlib.colors import LogNorm
from requirements import estimate_reqs_qwen3, estimate_reqs_qwen3_chunked_prefill

# Hardware constants from README
MEMORY_BANDWIDTH_PER_NODE = 256e9  # 256 GB/s
INTERCONNECT_BANDWIDTH_PER_NODE = 189e9 / 8  # 189 gbps -> 23.625 GB/s
FLOPS_PER_NODE = 126e12  # 126 TOPS

def _calculate_performance_data(estimation_func, batch_sizes, context_lengths, is_prefill, chunk_size=None):
    """
    Helper function to calculate performance and utilization data for different scenarios.
    """
    perf_data = {}
    util_data = {}

    outer_loop_vars = batch_sizes if is_prefill else context_lengths
    inner_loop_vars = context_lengths if is_prefill else batch_sizes

    for outer_var in outer_loop_vars:
        metrics = []
        utils_mem, utils_inter, utils_flops = [], [], []
        for inner_var in inner_loop_vars:
            if is_prefill:
                bs, p_len = outer_var, inner_var
            else:
                bs, p_len = inner_var, outer_var

            if estimation_func == estimate_reqs_qwen3_chunked_prefill:
                first_chunk_size = min(p_len, chunk_size)
                mem, inter, flops, _ = estimation_func(bs, first_chunk_size, chunk_size)
            else:
                mem, inter, flops, _ = estimation_func(bs, p_len, is_prefill=is_prefill)

            t_mem = mem / MEMORY_BANDWIDTH_PER_NODE
            t_inter = inter / INTERCONNECT_BANDWIDTH_PER_NODE
            t_flops = flops / FLOPS_PER_NODE
            t_pass = max(t_mem, t_inter, t_flops)

            if is_prefill:
                metrics.append(t_pass * 1000)  # ms for latency
            else:
                metrics.append(1.0 / t_pass if t_pass > 0 else 0) # batches/sec for throughput

            if t_pass > 0:
                utils_mem.append(t_mem / t_pass)
                utils_inter.append(t_inter / t_pass)
                utils_flops.append(t_flops / t_pass)
            else:
                utils_mem.append(0); utils_inter.append(0); utils_flops.append(0)

        perf_data[outer_var] = metrics
        util_data[outer_var] = {'mem': utils_mem, 'inter': utils_inter, 'flops': utils_flops}

    return perf_data, util_data


def generate_performance_plots():
    """
    Generates and displays plots for cluster throughput, latency, and utilization.
    """
    # --- Scenarios ---
    gen_batch_sizes = [2**i for i in range(11)]  # 1, 2, 4, ..., 1024
    gen_context_lengths = [2**4, 2**8, 2**12, 2**16] # 1k, 4k, 16k, 64k

    prefill_prompt_lengths = [2**i for i in range(18)] # 1, 2, 4, ..., 128k
    prefill_batch_sizes = [1, 8, 64, 256]
    chunk_size = 256

    # --- Data Calculation ---
    gen_perf_data, gen_util_data = _calculate_performance_data(
        estimate_reqs_qwen3, gen_batch_sizes, gen_context_lengths, is_prefill=False
    )
    prefill_perf_data, prefill_util_data = _calculate_performance_data(
        estimate_reqs_qwen3, prefill_batch_sizes, prefill_prompt_lengths, is_prefill=True
    )
    chunked_prefill_perf_data, chunked_prefill_util_data = _calculate_performance_data(
        estimate_reqs_qwen3_chunked_prefill, prefill_batch_sizes, prefill_prompt_lengths, is_prefill=True, chunk_size=chunk_size
    )

    # --- Plotting ---
    fig, axs = plt.subplots(3, 3, figsize=(16,12), sharex='col')

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
        axs[0, 0].plot(gen_batch_sizes, throughputs, color=colors[i], label=f'Context: {ctx} tokens')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="--")
    axs[0, 0].tick_params(axis='y', labelsize=tick_fontsize)


    # Utilization for smallest context
    ctx_small = gen_context_lengths[0]
    utils_small = gen_util_data[ctx_small]
    axs[1, 0].set_title(f'Resource Utilization (Context: {ctx_small} Tokens)', fontsize=14)
    axs[1, 0].set_ylabel('Utilization (%)', fontsize=label_fontsize)
    axs[1, 0].plot(gen_batch_sizes, utils_small['mem'], color=colors[0], linestyle=linestyles['mem'], label='Memory BW')
    axs[1, 0].plot(gen_batch_sizes, utils_small['inter'], color=colors[0], linestyle=linestyles['inter'], label='Interconnect BW')
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
    axs[2, 0].set_title(f'Resource Utilization (Context: {ctx_large} Tokens)', fontsize=14)
    axs[2, 0].set_ylabel('Utilization (%)', fontsize=label_fontsize)
    axs[2, 0].plot(gen_batch_sizes, utils_large['mem'], color=color_large_ctx, linestyle=linestyles['mem'], label='Memory BW')
    axs[2, 0].plot(gen_batch_sizes, utils_large['inter'], color=color_large_ctx, linestyle=linestyles['inter'], label='Interconnect BW')
    axs[2, 0].plot(gen_batch_sizes, utils_large['flops'], color=color_large_ctx, linestyle=linestyles['flops'], label='FLOPs')
    axs[2, 0].set_ylim(0, 1.1)
    axs[2, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axs[2, 0].grid(True, which="both", ls="--")
    axs[2, 0].legend()
    axs[2, 0].set_xlabel('Batch Size', fontsize=label_fontsize)
    axs[2, 0].set_xscale('log', base=2)
    axs[2, 0].tick_params(axis='both', labelsize=tick_fontsize)


    # == Column 2: Prefill Latency (No Chunking) ==
    axs[0, 1].set_title('Time To First Token (No Chunking)', fontsize=14)
    axs[0, 1].set_ylabel('Time To First Token (ms)', fontsize=label_fontsize)
    axs[0, 1].set_yscale('log')
    for i, (bs, latencies) in enumerate(prefill_perf_data.items()):
        axs[0, 1].plot(prefill_prompt_lengths, latencies, color=colors[i], label=f'Batch Size: {bs}')
    axs[0, 1].legend()
    axs[0, 1].grid(True, which="both", ls="--")
    axs[0, 1].tick_params(axis='y', labelsize=tick_fontsize)

    # Utilization for smallest batch size (No Chunking)
    bs_small = prefill_batch_sizes[0]
    utils_small = prefill_util_data[bs_small]
    axs[1, 1].set_title(f'Resource Utilization (Batch Size: {bs_small})', fontsize=14)
    axs[1, 1].set_ylabel('Utilization (%)', fontsize=label_fontsize)
    axs[1, 1].plot(prefill_prompt_lengths, utils_small['mem'], color=colors[0], linestyle=linestyles['mem'], label='Memory BW')
    axs[1, 1].plot(prefill_prompt_lengths, utils_small['inter'], color=colors[0], linestyle=linestyles['inter'], label='Interconnect BW')
    axs[1, 1].plot(prefill_prompt_lengths, utils_small['flops'], color=colors[0], linestyle=linestyles['flops'], label='FLOPs')
    axs[1, 1].set_ylim(0, 1.1)
    axs[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axs[1, 1].grid(True, which="both", ls="--")
    axs[1, 1].legend()
    axs[1, 1].tick_params(axis='y', labelsize=tick_fontsize)

    # Utilization for largest batch size (No Chunking)
    bs_large = prefill_batch_sizes[-1]
    utils_large = prefill_util_data[bs_large]
    color_large_bs = colors[len(prefill_batch_sizes) - 1]
    axs[2, 1].set_title(f'Resource Utilization (Batch Size: {bs_large})', fontsize=14)
    axs[2, 1].set_ylabel('Utilization (%)', fontsize=label_fontsize)
    axs[2, 1].plot(prefill_prompt_lengths, utils_large['mem'], color=color_large_bs, linestyle=linestyles['mem'], label='Memory BW')
    axs[2, 1].plot(prefill_prompt_lengths, utils_large['inter'], color=color_large_bs, linestyle=linestyles['inter'], label='Interconnect BW')
    axs[2, 1].plot(prefill_prompt_lengths, utils_large['flops'], color=color_large_bs, linestyle=linestyles['flops'], label='FLOPs')
    axs[2, 1].set_ylim(0, 1.1)
    axs[2, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axs[2, 1].grid(True, which="both", ls="--")
    axs[2, 1].legend()
    axs[2, 1].set_xlabel('Prompt Length (Tokens)', fontsize=label_fontsize)
    axs[2, 1].set_xscale('log', base=2)
    axs[2, 1].tick_params(axis='both', labelsize=tick_fontsize)

    # == Column 3: Prefill Latency (Chunked) ==
    axs[0, 2].set_title(f'Time To First Token (Chunked Prefill, {chunk_size} Tokens/Chunk)', fontsize=14)
    axs[0, 2].set_ylabel('Time To First Token (ms)', fontsize=label_fontsize)
    axs[0, 2].set_yscale('log')
    for i, (bs, latencies) in enumerate(chunked_prefill_perf_data.items()):
        axs[0, 2].plot(prefill_prompt_lengths, latencies, color=colors[i], label=f'Batch Size: {bs}')
    axs[0, 2].legend()
    axs[0, 2].grid(True, which="both", ls="--")
    axs[0, 2].tick_params(axis='y', labelsize=tick_fontsize)

    # Utilization for smallest batch size (Chunked)
    bs_small = prefill_batch_sizes[0]
    utils_small = chunked_prefill_util_data[bs_small]
    axs[1, 2].set_title(f'Resource Utilization (Batch Size: {bs_small})', fontsize=14)
    axs[1, 2].set_ylabel('Utilization (%)', fontsize=label_fontsize)
    axs[1, 2].plot(prefill_prompt_lengths, utils_small['mem'], color=colors[0], linestyle=linestyles['mem'], label='Memory BW')
    axs[1, 2].plot(prefill_prompt_lengths, utils_small['inter'], color=colors[0], linestyle=linestyles['inter'], label='Interconnect BW')
    axs[1, 2].plot(prefill_prompt_lengths, utils_small['flops'], color=colors[0], linestyle=linestyles['flops'], label='FLOPs')
    axs[1, 2].set_ylim(0, 1.1)
    axs[1, 2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axs[1, 2].grid(True, which="both", ls="--")
    axs[1, 2].legend()
    axs[1, 2].tick_params(axis='y', labelsize=tick_fontsize)

    # Utilization for largest batch size (Chunked)
    bs_large = prefill_batch_sizes[-1]
    utils_large = chunked_prefill_util_data[bs_large]
    color_large_bs = colors[len(prefill_batch_sizes) - 1]
    axs[2, 2].set_title(f'Resource Utilization (Batch Size: {bs_large})', fontsize=14)
    axs[2, 2].set_ylabel('Utilization (%)', fontsize=label_fontsize)
    axs[2, 2].plot(prefill_prompt_lengths, utils_large['mem'], color=color_large_bs, linestyle=linestyles['mem'], label='Memory BW')
    axs[2, 2].plot(prefill_prompt_lengths, utils_large['inter'], color=color_large_bs, linestyle=linestyles['inter'], label='Interconnect BW')
    axs[2, 2].plot(prefill_prompt_lengths, utils_large['flops'], color=color_large_bs, linestyle=linestyles['flops'], label='FLOPs')
    axs[2, 2].set_ylim(0, 1.1)
    axs[2, 2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axs[2, 2].grid(True, which="both", ls="--")
    axs[2, 2].legend()
    axs[2, 2].set_xlabel('Prompt Length (Tokens)', fontsize=label_fontsize)
    axs[2, 2].set_xscale('log', base=2)
    axs[2, 2].tick_params(axis='both', labelsize=tick_fontsize)


    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig('cluster_performance.png')
    plt.show()

def generate_2d_performance_plots():
    """
    Generates 2D colormap plots for throughput and latency vs. batch size and context length.
    """
    batch_sizes = [2**i for i in range(11)]  # 1, 2, 4, ..., 1024
    context_lengths = [2**i for i in range(18)] # 1, 2, 4, ..., 128k
    chunk_size = 256

    batches_per_sec = np.zeros((len(context_lengths), len(batch_sizes)))
    tok_per_sec = np.zeros((len(context_lengths), len(batch_sizes)))
    ttft = np.zeros((len(context_lengths), len(batch_sizes)))
    ttft_chunked = np.zeros((len(context_lengths), len(batch_sizes)))

    for i, ctx in enumerate(context_lengths):
        for j, bs in enumerate(batch_sizes):
            # Generation throughput
            mem_gen, inter_gen, flops_gen, _ = estimate_reqs_qwen3(bs, ctx, is_prefill=False)
            t_pass_gen = max(mem_gen / MEMORY_BANDWIDTH_PER_NODE,
                             inter_gen / INTERCONNECT_BANDWIDTH_PER_NODE,
                             flops_gen / FLOPS_PER_NODE)
            batches_per_sec[i, j] = 1.0 / t_pass_gen if t_pass_gen > 0 else 0
            tok_per_sec[i, j] = bs / t_pass_gen if t_pass_gen > 0 else 0

            # Prefill TTFT (ms) - No Chunking
            mem_prefill, inter_prefill, flops_prefill, _ = estimate_reqs_qwen3(bs, ctx, is_prefill=True)
            t_pass_prefill = max(mem_prefill / MEMORY_BANDWIDTH_PER_NODE,
                                 inter_prefill / INTERCONNECT_BANDWIDTH_PER_NODE,
                                 flops_prefill / FLOPS_PER_NODE)
            ttft[i, j] = t_pass_prefill * 1000

            # Prefill TTFT (ms) - Chunked
            first_chunk_context = min(ctx, chunk_size)
            mem_chunked, inter_chunked, flops_chunked, _ = estimate_reqs_qwen3(bs, first_chunk_context, is_prefill=True)
            t_pass_chunked = max(mem_chunked / MEMORY_BANDWIDTH_PER_NODE,
                                 inter_chunked / INTERCONNECT_BANDWIDTH_PER_NODE,
                                 flops_chunked / FLOPS_PER_NODE)
            ttft_chunked[i, j] = t_pass_chunked * 1000


    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Performance vs. Batch Size and Context Length', fontsize=16)

    vmin = min(ttft[ttft > 0].min(), ttft_chunked[ttft_chunked > 0].min())
    vmax = max(ttft.max(), ttft_chunked.max())

    # Batches per Second plot
    im1 = axs[0, 0].pcolormesh(batch_sizes, context_lengths, batches_per_sec, shading='auto', cmap='viridis')
    axs[0, 0].set_title('Generation Throughput (Batches/Second)')
    axs[0, 0].set_xscale('log', base=2)
    axs[0, 0].set_yscale('log', base=2)
    axs[0, 0].set_xlabel('Batch Size')
    axs[0, 0].set_ylabel('Context Length')
    fig.colorbar(im1, ax=axs[0, 0], label='Batches / Second')

    # TTFT plot (No Chunking)
    im2 = axs[0, 1].pcolormesh(batch_sizes, context_lengths, ttft, shading='auto', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
    axs[0, 1].set_title('Time To First Token (ms) - No Chunking')
    axs[0, 1].set_xscale('log', base=2)
    axs[0, 1].set_yscale('log', base=2)
    axs[0, 1].set_xlabel('Batch Size')
    axs[0, 1].set_ylabel('Context Length')
    fig.colorbar(im2, ax=axs[0, 1], label='Milliseconds')

    # Tokens per Second plot
    im3 = axs[1, 0].pcolormesh(batch_sizes, context_lengths, tok_per_sec, shading='auto', cmap='viridis')
    axs[1, 0].set_title('Generation Throughput (Total Tokens/Second)')
    axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].set_yscale('log', base=2)
    axs[1, 0].set_xlabel('Batch Size')
    axs[1, 0].set_ylabel('Context Length')
    fig.colorbar(im3, ax=axs[1, 0], label='Tokens / Second')

    # TTFT plot (Chunked)
    im4 = axs[1, 1].pcolormesh(batch_sizes, context_lengths, ttft_chunked, shading='auto', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
    axs[1, 1].set_title(f'Time To First Token (ms) - Chunked ({chunk_size} Tokens)')
    axs[1, 1].set_xscale('log', base=2)
    axs[1, 1].set_yscale('log', base=2)
    axs[1, 1].set_xlabel('Batch Size')
    axs[1, 1].set_ylabel('Context Length')
    fig.colorbar(im4, ax=axs[1, 1], label='Milliseconds')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('performance_vs_batch_size.png')
    plt.show()


if __name__ == "__main__":
    generate_performance_plots()
    generate_2d_performance_plots()
