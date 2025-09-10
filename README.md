## Qwen3 cluster performance estimation

The goal of this repository is to estimate as closely as possible the requirements per token for memory bandwidth, interconnect bandwidth, and FPU throughput for qwen3 235b at 8bit quantization running on a cluster of 4x Ryzen 395 max nodes with 128GB of unified memory and one GPU per node. The qwen3 technical report is in the directory as `qwen3.pdf. Please assume all parameters are 1 byte each, intermediates like embedding vectors are 2 bytes per element (bf16), and that we put one of each of the 4 GQA attention groups on each node for each attention blocks, and that we put 1/4 of the experts on each node. We will be running with the full 128k context (but please keep this parameter free and take it as a command line argument so we can explore RoPE based context extenstion) with full KV caching.

The ultimate goal is to produce a single function `estimate_reqs_qwen3(batch_size, context_length=128*1024)` which returns `(memory_bus_bytes_per_token, interconnect_bytes_per_token, floating_point_ops_per_token, max_kv_cache_bytes)`. We will implement that function with a series of more limited functions with similar signatures (e.g. `estimate_reqs_per_attn_block(...)` etc). We are going to start from the bottom up and build functions and unit tests, and run the unit tests to verify that function before moving onto the next function.

Once we have the final function we will make plots of requirements as functions of context length and batch size. this step will be interactive so stop when you get here and we will discuss next steps

work in python, use pytest for tests and pandas and matplotlib for visualization. Put code in requirements.py and test in test_requirements.py
