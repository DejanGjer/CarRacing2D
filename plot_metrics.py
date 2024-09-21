import matplotlib.pyplot as plt
import argparse
import json
import os

def plot_metric(episodes, metric, metric_name, output_dir):
    '''Plot the metric per episode and save it to a file'''
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, metric)
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} per Episode")
    plt.savefig(os.path.join(output_dir, f"{metric_name.replace(' ', '_').lower()}.png"))

def average_reward_per_episode(rewards, window=10):
    '''Calculate the average reward per episode'''
    avg_rewards = []
    for i in range(len(rewards)):
        start = max(0, i - window)
        avg_rewards.append(sum(rewards[start:i+1]) / (i - start + 1))
    return avg_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the file with the metrics")
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    args = parser.parse_args()
    file_path = args.file
    with open(file_path, 'r') as file:
        data = json.load(file)
    os.makedirs(args.output, exist_ok=True)
    plot_metric(data["episode"], data["total_rewards"], "Total Rewards", args.output)
    plot_metric(data["episode"], data["frames"], "Frames", args.output)
    plot_metric(data["episode"], data["epsilon"], "Epsilon", args.output)
    average_rewards = average_reward_per_episode(data["total_rewards"])
    plot_metric(data["episode"], average_rewards, "Average Rewards", args.output)