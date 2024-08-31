import argparse
import gym
from collections import deque
from CarRacingDQNAgentTorch import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
from tqdm import tqdm
import time
import os
import json


RENDER                        = False
STARTING_EPISODE              = 1
ENDING_EPISODE                = 10
SKIP_FRAMES                   = 2
TRAINING_BATCH_SIZE           = 64
SAVE_TRAINING_FREQUENCY       = 5
UPDATE_TARGET_MODEL_FREQUENCY = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    parser.add_argument('-w', '--windows_size', type=int, default=3, help='The number of frames to stack, default to 3.')
    args = parser.parse_args()

    output_path = os.path.join(os.getcwd(), "output", time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(output_path, exist_ok=True)
    # save all arguments to config file
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(vars(args), f)

    results = {
        "episode": [],
        "frames": [],
        "total_rewards": [],
        "epsilon": []
    }

    print("Loading environment ...")
    env = gym.make('CarRacing-v2')
    print("Loading agent ...")
    agent = CarRacingDQNAgent(epsilon=args.epsilon)
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end

    print("Running training loop")
    for e in tqdm(range(STARTING_EPISODE, ENDING_EPISODE+1)):
        init_state = env.reset()[0]
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        
        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            reward = 0
            for _ in range(SKIP_FRAMES+1):
                next_state, r, terminated, truncated, info = env.step(action)
                reward += r
                done = terminated or truncated
                if done:
                    break

            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Extra bonus for the model if it uses full gas
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                results["episode"].append(e)
                results["frames"].append(time_frame_counter)
                results["total_rewards"].append(total_reward)
                results["epsilon"].append(agent.epsilon)
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.5}, Epsilon: {:.5}'.format(e, ENDING_EPISODE, time_frame_counter, float(total_reward), float(agent.epsilon)))
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)
            agent.save(os.path.join(output_path, "checkpoints" , f"model_{e}.pt"))
            
    env.close()
    print("Training completed")

    # save results to a json file
    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results, f)
    print("Metrics saved")

