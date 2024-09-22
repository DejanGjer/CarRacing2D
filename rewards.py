import cv2
import numpy as np

class OutOfTrackReward:
    def __init__(self, value, min_negative_steps=8, max_negative_steps=16, decay_episodes_number=400):
        self.value = value
        self.min_negative_steps = min_negative_steps
        self.max_negative_steps = max_negative_steps
        self.decay_episodes_number = decay_episodes_number
        self.decay_value = (max_negative_steps - min_negative_steps) / decay_episodes_number
        self.negative_steps = max_negative_steps
    
    def decay_steps(self):
        self.negative_steps = max(self.min_negative_steps, self.negative_steps - self.decay_value)
        
    def outside_of_the_track(self, reward_history):
        if len(reward_history) < int(self.negative_steps):
            return False
        outside = True
        for r in reward_history[-int(self.negative_steps):]:
            if r > 0:
                outside = False
                
        return outside
    
    def get_reward(self, reward_history):
        if self.outside_of_the_track(reward_history):
            return self.value
        return 0
    
class GrassPenatly:
    def __init__(self, value, start_from_step=25):
        self.value = value
        self.start_from_step = start_from_step
    
    def is_green(self, pixel):
        # Check if the pixel is green
        if pixel[1] < pixel[0] or pixel[1] - pixel[0] <= 50:
            return False
        if pixel[1] < pixel[2] or pixel[1] - pixel[2] <= 50:
            return False
        return True

    def is_grass(self, frame, step):
        # Check if the car is on the grass
        if step < self.start_from_step:
            return 0
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.is_green(rgb_frame[65, 47]) and self.is_green(rgb_frame[72, 45]) and self.is_green(rgb_frame[72, 50]):
            return self.value
        return 0
    
class PreventDriftingPenalty:
    def __init__(self, value):
        self.value = value
    
    def prevent_drifting(self, action):
        if action[1] == 1 and action[0] != 0:
            return self.value
        return 0
    
class GasReward:
    def __init__(self, value):
        self.value = value
    
    def is_full_gas(self, action):
        if action[1] == 1 and action[2] == 0 and action[0] == 0:
            return self.value
        return 0
        
class TimeReward:
    def __init__(self, sucessfull_reward_threshold, good_avg_speed, failure_value, sucessfull_value):
        self.sucessfull_reward_threshold = sucessfull_reward_threshold
        self.good_avg_speed = good_avg_speed
        self.failure_value = failure_value
        self.sucessfull_value = sucessfull_value
        
    def get_reward(self, tile_reward, time_frame_counter):
        if tile_reward < self.sucessfull_reward_threshold:
            return self.failure_value
        avg_speed = tile_reward / time_frame_counter
        if avg_speed < self.good_avg_speed:
            return 0
        return ((avg_speed / self.good_avg_speed) ** 2) * self.sucessfull_value


        
