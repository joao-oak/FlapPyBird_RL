import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

# placeholder hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.1
EPS_END = 0.0001
EPS_DECAY = 3000
TAU = 0.005
LR = 5e-5

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) # custom data structure for organization purposes

# This is to save the agent's experiences and sample from them later when training the NN 
class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Network architecture
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, tau=0.005, epsilon_start=0.5, epsilon_min=0.001, epsilon_decay=1000, batch_size=32, memory_size=10000):
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.update_target_frequency = 20

        self.env = env
        state = env.reset()
        self.state_size = len(state)
        self.action_size = env.action_space.n

        self.policy_model = DQNetwork(self.state_size, self.action_size)
        self.target_model = DQNetwork(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.policy_model.state_dict()) # copying initial weights to the target network

        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=learning_rate, amsgrad=True)

        self.memory = ReplayMemory(memory_size)
        self.steps_done = 0
        self.episode_rewards = [] # for plotting purposes later on

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, deterministic=False):

        eps_threshold = 0 if deterministic else self.epsilon_min + (self.epsilon_start - self.epsilon_min) * math.exp(-1. * self.steps_done / self.epsilon_decay)

        self.steps_done += 1

        if random.random() < eps_threshold:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long) # random action
        
        with torch.no_grad():
            return self.policy_model(state).max(1).indices.view(1, 1)
        

    # def plot_rewards(self, episode):

    #     rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
    #     plt.title('DQN Result')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Total Rewards')
    #     plt.plot(rewards_t.numpy(), color="tab:blue")
    #     # Take 100 episode averages and plot them too
    #     if len(rewards_t) >= 100:
    #         means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
    #         means = torch.cat((torch.zeros(99), means))
    #         plt.plot(means.numpy(), color="tab:orange")
    #     plt.savefig(f'chart_{episode}.png', dpi=300) 

    def plot_rewards(self, episode):
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        plt.title('DQN Result')
        plt.xlabel('Episode')
        plt.ylabel('Total Rewards')

        plt.scatter(range(len(rewards_t)), rewards_t.numpy(), color="tab:blue", label='Rewards', s=2)

        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), color="tab:orange", label='100-episode average')
        plt.legend()
        plt.savefig(f'chart_{episode}.png', dpi=300)
        plt.close()

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        minibatch = Transition(*zip(*transitions)) # better than the for loop from before

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, minibatch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in minibatch.next_state if s is not None])
        state_minibatch = torch.cat(minibatch.state)
        action_minibatch = torch.cat(minibatch.action)
        reward_minibatch = torch.cat(minibatch.reward)

        # Q(s_t, a)
        state_action_values = self.policy_model(state_minibatch).gather(1, action_minibatch)

        # V(s_t+1)
        next_state_values = torch.zeros(self.batch_size)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1).values # .values instead of [0].detach()

        # Q(s_t, a) = γ * V(s_t+1) + r
        expected_state_action_values = (self.gamma * next_state_values) + reward_minibatch

        # Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()
    
        # # Decay epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def training(self, episodes, seed=21):

        # Deep Q-Network (DQN)

        # PERCEBER SE O MODEL.TRAIN() É PRECISO AQUI OU NÃO

        for episode in range(episodes):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            total_reward = 0
            self.steps_done = 0
            done = False
            
            while not done:
                # Choose action
                action = self.choose_action(state, deterministic=True)
                
                # act in the environment
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward])
                
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)

                # save the experience in memory
                self.memory.push(state, action, next_state, reward)
                
                # Update state
                state = next_state
                total_reward += reward

                # Train the policy model
                self.replay()

                # soft update of the target model
                # # if self.steps_done % self.update_target_frequency == 0: # to update the target model every update_target_frequency steps
                # target_model_state_dict = self.target_model.state_dict() # dictionary that maps each layer to its parameter tensor
                # policy_model_state_dict = self.policy_model.state_dict()
                # for key in policy_model_state_dict:
                #     target_model_state_dict[key] = policy_model_state_dict[key]*self.tau + target_model_state_dict[key]*(1-self.tau)
                # self.target_model.load_state_dict(target_model_state_dict)

                if self.steps_done % self.update_target_frequency == 0:
                    for target_param, policy_param in zip(self.target_model.parameters(), self.policy_model.parameters()):
                        target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)


                if done:
                    self.episode_rewards.append(total_reward)
                    # self.plot_rewards()

            if episode > 1 and (episode+1) % 500 == 0:
                torch.save(self.target_model.state_dict(), f'DQNweights_{episode}.pt')
                print(f'Saved weigths after {episode} episodes')
                self.plot_rewards(episode)

            print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")

            # if total_reward > 70000:
            #     torch.save(self.target_model.state_dict(), f'DQNweights_{episode}.pt')
            #     print(f'Saved weigths after {episode} episodes')
        
        print('Finished training')
    
    def ai_play(self, filename):

        loaded_state_dict = torch.load(filename)
        self.policy_model.load_state_dict(loaded_state_dict)
        self.target_model.load_state_dict(loaded_state_dict)

        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        while True:
            
            action = self.choose_action(state, deterministic=True)

            next_state, reward, done, _ = self.env.step(action.item())
            total_reward += reward
            if done:
                print(f"Total Reward: {total_reward}")
                total_reward = 0
                state = self.env.reset()

            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)