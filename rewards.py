
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
        
