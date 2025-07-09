#!/bin/bash

# This script runs the pytest benchmark for the ZDT1 problem, comparing
# the 'swarm' library against 'pymoo'.
#
# It uses specific pytest-benchmark options to control the output columns,
# maximum run time, and minimum number of rounds for statistical significance.

# The name of the Python file containing the benchmarks.
BENCHMARK_FILE="zdt1_bench.py"
# BENCHMARK_FILE="blackbox_par_bench.py"
OUTPUT_JSON="benchmark_data.json"

echo "--- Starting Benchmark: Swarm vs. Pymoo ---"

pytest ${BENCHMARK_FILE} \
	--benchmark-columns='mean,median,stddev,iqr,outliers,rounds,iterations' \
	--benchmark-max-time=10 \
	--benchmark-min-rounds=10 \
	--benchmark-json=${OUTPUT_JSON}

echo "--- Benchmark Complete ---"
echo "Results saved to ${OUTPUT_JSON}"
