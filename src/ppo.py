import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import os
import numpy as np
from datetime import datetime
import pandas as pd

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
     

        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, env, lr_actor, lr_critic, update_timestep, gamma, K_epochs, eps_clip, action_std_init=0.6):


        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.update_timestep = update_timestep
        
        self.buffer = RolloutBuffer()

        self.env = env
        self.state_dim = self.env.state_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.policy = ActorCritic(self.state_dim, self.action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(self.state_dim, self.action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
    def train(self, random_seed, episodes):

        ################### checkpointing ###################
        run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + 'FlappyBird' + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        #variables needed to compare models
        reward_goal = 100
        max_reward = 0

        ################# training procedure ################


        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        update_timestep = self.update_timestep
        checkpoint_path = directory + "PPO_{}_actor{}_critic{}_tmsp{}.pth".format('FlappyBird', self.lr_actor, self.lr_critic, update_timestep)
        print("save checkpoint path : " + checkpoint_path)
        #####################################################

        print("============================================================================================")
        time_step = 0

        # training loop
        csv_episode = []
        csv_reward = []
        
        for episode in range(episodes):

            state = self.env.reset()
            current_ep_reward = 0
            episode_step = 0
            done = False

            while not done:

                # select action with policy
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)

                # saving reward and is_terminals
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                time_step +=1
                episode_step += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % update_timestep == 0:
                    self.update()

                # save model weights
                if current_ep_reward > reward_goal and current_ep_reward > max_reward:
                    self.save(checkpoint_path)
                    max_reward=current_ep_reward
            
            csv_episode.append(episode+1)
            csv_reward.append(current_ep_reward)
            print(f"Episode: {episode+1}/{episodes}, Total Reward: {current_ep_reward}")

        #Save file
        file = pd.DataFrame({'Episode': csv_episode,
                    'Reward': csv_reward})
        
        file.to_csv(f'PPO_tests/PPO_actor{str(self.lr_actor)}_critic{str(self.lr_critic)}_timestep{str(update_timestep)}.csv', index=False)
        
        self.env.close()

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")

    def test(self):
        print("============================================================================================")

        # preTrained weights directory

        random_seed = 1             #### set this to load a particular checkpoint trained on random seed
        run_num_pretrained = 0      #### set this to load a particular checkpoint num

        directory = "PPO_preTrained" + '/' + 'FlappyBird' + '/'
        checkpoint_path = directory + "PPO_{}_{}_{}.pth".format('FlappyBird', random_seed, run_num_pretrained)
        print("loading network from : " + checkpoint_path)

        self.load(checkpoint_path)

        print("--------------------------------------------------------------------------------------------")

        ep_reward = 0
        state = self.env.reset()

        while True:
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            ep_reward += reward

            if done:
                print(f"Total Reward: {ep_reward}")
                ep_reward = 0
                state = self.env.reset()

            # clear buffer
            self.buffer.clear()

        # self.env.close()


#CHANGES:
#reward goal to 100
#change save model name
# add csv
# change place for checkpoint save
# add for cicle for timesteps
