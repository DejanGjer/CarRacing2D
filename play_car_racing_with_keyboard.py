import gym
import numpy as np
import pygame
import time

pygame.init()

# Initialize the CarRacing environment
env = gym.make("CarRacing-v2", render_mode="human")
env.reset()

# Setup the Pygame display
pygame.display.set_mode((1000, 800))

# Define a function to capture keyboard inputs
def get_action():
    # Create an action array for [steering, acceleration, brake]
    action = np.array([0.0, 0.0, 0.0])

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:  # Left
        action[0] = -0.5
    if keys[pygame.K_RIGHT]:  # Right
        action[0] = 0.5
    if keys[pygame.K_UP]:  # Accelerate
        action[1] = 1.0
    if keys[pygame.K_DOWN]:  # Brake
        action[2] = 0.5

    return action

# Run the environment loop
done = False
reward = 0
frames = 0
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    action = get_action()
    _, r, terminated, truncated, _ = env.step(action)
    reward += r
    frames += 1
    print(f"Frames: {frames}, Reward: {r}, Total Reward: {reward}")
    done = terminated or truncated
    if done:
        env.reset()
    time.sleep(0.02)

env.close()
pygame.quit()

print(f"Reward: {reward}")
