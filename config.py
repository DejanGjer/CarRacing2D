# Training configuration
name = "Trying more frequent target model updates"
pretrained_model_path = None
num_episodes = 1000
render = False
skip_frames = 2
batch_size = 64
learning_rate = 0.001
checkpoint_frequency = 100
update_target_model_frequency = 1
max_consecutive_negative_steps = 25

# Agent configuration
image_size = (96, 96)
state_frame_stack = 3
memory_buffer_size = 5000
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.9999
actions_space = [
    (-1, 1,   0), (0, 1,   0), (1, 1,   0),
    (-1, 0,   0), (0, 0,   0), (1, 0,   0),   #           Action Space Structure
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),   #        (Steering Wheel, Gas, Break)
    (-1, 0, 0.5), (0, 0, 0.5), (1, 0, 0.5),   # Range        -1~1       0~1   0~1
]
rewards = ["gas"]
# out_of_track_reward_args = {
#     "value": -10,
#     "min_negative_steps": 8,
#     "max_negative_steps": 36,
#     "decay_episodes_number": 400
# }