import argparse
import gym
import pygame
import time
from collections import deque
import importlib.util
import sys
import torch
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    return config

def play(train_model, play_episodes, render_mode, config):
    env = gym.make('CarRacing-v2', render_mode=render_mode)
    agent = CarRacingDQNAgent(
        action_space=config.actions_space,
        frame_stack_num=config.state_frame_stack,
        memory_size=config.memory_buffer_size,
        gamma=config.gamma,
        epsilon=0,  # Set epsilon to 0 to ensure all actions are instructed by the agent
        epsilon_min=0,
        epsilon_decay=0,
        learning_rate=config.learning_rate, 
        device = torch.device("cpu")
    )
    agent.load_inference(train_model)

    actions = []

    for e in range(play_episodes):
        init_state = env.reset()[0]
        init_state = process_state_image(init_state)

        total_reward = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        actions.append([])
        frames = 0
        
        while True:
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            actions[-1].append(action)

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            frames += 1

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.5}'.format(e+1, play_episodes, frames, float(total_reward)))
                break
    env.close()
    return actions

def render(actions):
    env = gym.make('CarRacing-v2', render_mode="human")
    for e in range(len(actions)):
        init_state = env.reset()[0]
        init_state = process_state_image(init_state)
        done = False
        frame = 0
        
        while not done:
            env.render()
            action = actions[e][frame]
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame += 1
            time.sleep(0.02)

    env.close()
    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True, type=str, help='The `.pt` file of the trained model.')
    parser.add_argument('-c', '--config', required=True, type=str, help='The configuration file from the training process.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes should the model plays.')
    parser.add_argument('-p', '--precompute', action='store_true', help='Precompute the actions for each time frame.')
    args = parser.parse_args()
    train_model = args.model
    play_episodes = args.episodes
    config = load_config(args.config)

    if args.precompute:
        print("Precomputing the actions for each time frame...")
        actions = play(train_model, play_episodes, "rgb_array", config)
        print("Rendering the actions...")
        render(actions)
    else:
        print("Playing the CarRacing...")
        play(train_model, play_episodes, "human")