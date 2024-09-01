import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, kernel_size=7, stride=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(432, 216)
        self.fc2 = nn.Linear(216, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)   
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CarRacingDQNAgent:
    def __init__(
        self,
        action_space,
        frame_stack_num = 3,
        memory_size     = 5000,
        gamma           = 0.95,  # discount rate
        epsilon         = 1.0,   # exploration rate
        epsilon_min     = 0.1,
        epsilon_decay   = 0.9999,
        learning_rate   = 0.001,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.action_space    = action_space
        self.frame_stack_num = frame_stack_num
        self.memory          = deque(maxlen=memory_size)
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        self.device          = device
        print(f"Using device: {self.device}")

        self.model           = DQN((frame_stack_num, 96, 96), len(action_space)).to(self.device)
        self.target_model    = DQN((frame_stack_num, 96, 96), len(action_space)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.update_target_model()


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                act_values = self.model(state)
            action_index = torch.argmax(act_values[0]).item()
        else:
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            target = self.model(state).detach().cpu().numpy()[0]
            if done:
                target[action_index] = reward
            else:
                t = self.target_model(next_state).detach().cpu().numpy()[0]
                target[action_index] = reward + self.gamma * np.amax(t)
            train_state.append(state.squeeze(0).cpu().numpy())
            train_target.append(target)
        
        train_state = torch.tensor(train_state, dtype=torch.float32).to(self.device)
        train_target = torch.tensor(train_target, dtype=torch.float32).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(train_state)
        loss = self.criterion(outputs, train_target)
        loss.backward()
        self.optimizer.step()
    
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.update_target_model()

    def load_inference(self, name):
        self.model.load_state_dict(torch.load(name, weights_only=True, map_location=torch.device("cpu")))
        self.model.eval()

    def save(self, name):
        torch.save(self.model.state_dict(), name)
