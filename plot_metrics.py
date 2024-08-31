import matplotlib.pyplot as plt
import argparse
import json
import os

def plot_metric(episodes, metric, metric_name, output_dir):
    '''Plot the metric per episode and save it to a file'''
    plt.plot(episodes, metric)
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} per Episode")
    plt.savefig(os.path.join(output_dir, f"{metric_name.replace(' ', '_').lower()}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting the metrics of the training process.')
    parser.argparse.ArgumentParser('-f', '--file', required=True, help='The json file that contains the metric data.')
    parser.argparse.ArgumentParser('-o', '--output', default=".", help='The output directory to save the plots.')
    args = parser.parse_args()
    file_path = args.file
    with open(file_path, 'r') as file:
        data = json.load(file)
    os.makedirs(args.output, exist_ok=True)
    plot_metric(data["episode"], data["total_rewards"], "Total Rewards", args.output)
    plot_metric(data["episode"], data["frames"], "Frames", args.output)
    plot_metric(data["episode"], data["epsilon"], "Epsilon", args.output)