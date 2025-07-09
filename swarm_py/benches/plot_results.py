import json
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_results(json_file="benchmark_data.json"):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The benchmark data file '{json_file}' was not found.")
        print("Please run the 'run_benchmark.sh' script first to generate it.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file}'. The file might be empty or corrupt.")
        return


    labels = []
    means = []
    
    name_map = {
        'test_swarm_zdt1': 'Swarm Py',
        'test_pymoo_zdt1': 'Pymoo Py'
    }

    benchmarks_data = data.get('benchmarks', [])
    if not benchmarks_data:
        print("No benchmark data found in the JSON file.")
        return

    for benchmark in benchmarks_data:
        test_name = benchmark.get('name')
        if test_name in name_map:
            labels.append(name_map[test_name])
            means.append(benchmark.get('stats', {}).get('mean', 0))

    if not labels:
        print("No benchmark data matching the expected names was found.")
        print(f"Script is looking for one of: {list(name_map.keys())}")
        return

    faster_index = np.argmin(means)
    colours = ['#d62728' if i != faster_index else '#2ca02c' for i in range(len(labels))] 

    _, ax = plt.subplots(figsize=(10, 5))
    
    bars = ax.barh(labels, means, color=colours, edgecolor='black')
    
    ax.set_xlabel('Mean Execution Time (seconds) - Lower is Better')
    ax.set_title('Performance Comparison: Swarm vs. Pymoo')
    ax.invert_yaxis()  # Display the first item on top
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                f'{width:.4f} s',
                va='center',
                fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_benchmark_results()