import gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
from tqdm import tqdm
import time
import os
import json
import shutil

import config
from rewards import OutOfTrackReward, GrassPenatly, PreventDriftingPenalty, GasReward

if __name__ == '__main__':
    print(config.name)
    output_path = os.path.join(os.getcwd(), "output", time.strftime('%Y%m%d-%H%M%S'))
    print(f"Output path: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    # copy current configuration file to the output path
    shutil.copy("config.py", os.path.join(output_path, "config.py"))

    results = {
        "episode": [],
        "frames": [],
        "total_rewards": [],
        "epsilon": []
    }

    print("Loading environment ...")
    env = gym.make('CarRacing-v2')
    print("Loading agent ...")
    agent = CarRacingDQNAgent(
        action_space=config.actions_space,
        image_size=config.image_size,
        frame_stack_num=config.state_frame_stack,
        memory_size=config.memory_buffer_size,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        learning_rate=config.learning_rate
    )
    if config.pretrained_model_path:
        agent.load(config.pretrained_model_path)

    print("Running training loop")
    for e in tqdm(range(config.num_episodes)):
        init_state = env.reset()[0]
        init_state = process_state_image(init_state, config.image_size)

        total_reward = 0
        negative_reward_counter = 0
        reward_history = []
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        if "out_of_track" in config.rewards:
            out_of_track_reward = OutOfTrackReward(**config.out_of_track_reward_args)
        if "grass" in config.rewards:
            grass_penalty = GrassPenatly(**config.grass_penalty_args)
        if "prevent_drifting" in config.rewards:
            prevent_drifting_penalty = PreventDriftingPenalty(**config.prevent_drifting_penalty_args)
        if "gas" in config.rewards:
            gas_reward = GasReward(**config.gas_reward_args)
        
        while True:
            if config.render:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            reward = 0
            for _ in range(config.skip_frames+1):
                next_state, r, terminated, truncated, info = env.step(action)
                reward += r
                done = terminated or truncated
                if done:
                    break

            reward_history.append(reward)
            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Extra bonus for the model if it uses full gas
            if "gas" in config.rewards and action[1] == 1 and action[2] == 0:
                reward += gas_reward.is_full_gas(action)

            # Extra penalty for the model if it goes out of the track
            if "out_of_track" in config.rewards:
                reward += out_of_track_reward.get_reward(reward_history)

            # Extra penalty for the model if it goes on the grass
            if "grass" in config.rewards:
                reward += grass_penalty.is_grass(next_state, time_frame_counter)

            # Extra penalty for the model if car drifts
            if "prevent_drifting" in config.rewards:
                reward += prevent_drifting_penalty.prevent_drifting(action)

            total_reward += reward

            next_state = process_state_image(next_state, config.image_size)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= config.max_consecutive_negative_steps or total_reward < 0:
                results["episode"].append(e+1)
                results["frames"].append(time_frame_counter)
                results["total_rewards"].append(total_reward)
                results["epsilon"].append(agent.epsilon)
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.5}, Epsilon: {:.5}'.format(e+1, config.num_episodes, time_frame_counter, float(total_reward), float(agent.epsilon)))
                break
            if len(agent.memory) > config.batch_size:
                agent.replay(config.batch_size)
            time_frame_counter += 1

        if "out_of_track" in config.rewards:
            out_of_track_reward.decay_steps()

        if (e+1) % config.update_target_model_frequency == 0:
            agent.update_target_model()

        if (e+1) % config.checkpoint_frequency == 0:
            os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)
            agent.save(os.path.join(output_path, "checkpoints" , f"model_{e+1}.pt"))
            
    env.close()
    print("Training completed")

    # save results to a json file
    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results, f)
    print("Metrics saved")

