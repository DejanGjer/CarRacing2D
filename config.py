# Training configuration
name = "Grass penalty"
pretrained_model_path = ""
num_episodes = 1000
render = False
skip_frames = 2
batch_size = 64
learning_rate = 0.001
checkpoint_frequency = 50
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
    (-0.5, 1,   0), (0, 1,   0), (0.5, 1,   0),
    (-0.5, 0,   0), (0, 0,   0), (0.5, 0,   0),   #           Action Space Structure
    (-0.5, 0, 0.2), (0, 0, 0.2), (0.5, 0, 0.2),   #        (Steering Wheel, Gas, Break)
    (-0.5, 0, 0.5), (0, 0, 0.5), (0.5, 0, 0.5),   # Range        -1~1       0~1   0~1
]
rewards = ["gas", "grass"]
# out_of_track_reward_args = {
#     "value": -1,
#     "min_negative_steps": 8,
#     "max_negative_steps": 16,
#     "decay_episodes_number": 400
# }
grass_penalty_args = {
    "value": -5,
    "start_from_step": 25
}
prevent_drifting_penalty_args = {
    "value": -0.2
}
gas_reward_args = {
    "value": 0.5
}