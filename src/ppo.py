import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import os
import numpy as np
from datetime import datetime

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
    def __init__(self, env, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):


        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
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
    
    def train(self, max_training_timesteps, random_seed, max_ep_len):

        ###################### logging ######################

        #### log files for multiple runs are NOT overwritten
        log_dir = "PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + 'FlappyBird' + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #### get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        log_f_name = log_dir + '/PPO_' + 'FlappyBird' + "_log_" + str(run_num) + ".csv"

        print("current logging run number for " + 'FlappyBird' + " : ", run_num)
        print("logging at : " + log_f_name)
        #####################################################

        ################### checkpointing ###################
        run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + 'FlappyBird' + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        #variables needed to compare models
        rew = 0
        print_avg_reward = 0

        checkpoint_path = directory + "PPO_{}_{}_{}.pth".format('FlappyBird', random_seed, run_num_pretrained)
        print("save checkpoint path : " + checkpoint_path)
        #####################################################

        print("============================================================================================")

        ################# training procedure ################


        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # logging file
        log_f = open(log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        update_timestep = max_ep_len * 4
        log_freq = max_ep_len * 2
        print_freq = max_ep_len * 10
        save_model_freq = int(1e5) 

        # training loop
        while time_step <= max_training_timesteps:

            state = self.env.reset()
            current_ep_reward = 0

            for t in range(1, max_ep_len+1):

                # select action with policy
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)

                # saving reward and is_terminals
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % update_timestep == 0:
                    self.update()

                # log in logging file
                if time_step % log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                #if time_step % save_model_freq == 0: ## Previous approach to save
                if print_avg_reward > rew:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    self.save(checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print(f"previous max: {rew}, new max: {print_avg_reward}")
                    rew=print_avg_reward
                    print("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        log_f.close()
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

        random_seed = 0             #### set this to load a particular checkpoint trained on random seed
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