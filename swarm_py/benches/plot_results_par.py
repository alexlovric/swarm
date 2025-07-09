import json
import matplotlib.pyplot as plt

def plot_alternative_benchmark_results(json_file="benchmark_data.json"):
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

    name_map = {
        'test_swarm_serial': 'Swarm Py (Serial)',
        'test_swarm_parallel': 'Swarm Py (Parallel)',
    }

    results = {}
    benchmarks_data = data.get('benchmarks', [])
    if not benchmarks_data:
        print("No benchmark data found in the JSON file.")
        return

    for benchmark in benchmarks_data:
        test_name = benchmark.get('name')
        if test_name in name_map:
            label = name_map[test_name]
            stats = benchmark.get('stats', {})
            results[label] = {
                'mean': stats.get('mean', 0),
                'stddev': stats.get('stddev', 0)
            }

    if not results:
        print("No benchmark data matching the expected names was found.")
        print(f"Script is looking for names like: {list(name_map.keys())}")
        return

    sorted_labels = sorted(results.keys(), key=lambda label: results[label]['mean'])
    means = [results[label]['mean'] for label in sorted_labels]
    std_devs = [results[label]['stddev'] for label in sorted_labels]

    def get_colour(label):
        if 'Parallel' in label:
            return '#2ca02c'
        else:
            return '#1f77b4'

    colours = [get_colour(label) for label in sorted_labels]

    _, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(sorted_labels, means, xerr=std_devs, color=colours,
                   capsize=5, edgecolor='black', alpha=0.9)

    ax.set_xlabel('Mean Execution Time (seconds) - Lower is Better')
    ax.set_title('Swarm Performance: Serial vs. Parallel')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.invert_yaxis()

    ax.bar_label(bars, fmt='%.4f s', padding=5, fontweight='bold')

    ax.set_xlim(0, max(means) * 1.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_alternative_benchmark_results()