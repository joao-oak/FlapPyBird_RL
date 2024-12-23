import asyncio

from src.flappybird_env import FlappyBirdEnv
from src import deep_q_n
from src.flappy import Flappy
from src import ppo
# from src import train

if __name__ == "__main__":

    # # To run the game
    # asyncio.run(Flappy().start())

    ###############################################################################################

    # # To train a model
    # ## DQN
    # env = FlappyBirdEnv()
    # model = deep_q_n.DQNAgent(env)
    # model.training(5000)

    ###############################################################################################

    # ##PPO
    # env = FlappyBirdEnv()

    # ####### initialize environment hyperparameters ######
    # max_ep_len = 1000                   # max timesteps in one episode
    # max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    # print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    # log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    # save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    # action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    # action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    # min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    # action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    # #####################################################


    # ################ PPO hyperparameters ################
    # update_timestep = max_ep_len * 4      # update policy every n timesteps
    # K_epochs = 80               # update policy for K epochs in one PPO update

    # eps_clip = 0.2          # clip parameter for PPO
    # gamma = 0.99            # discount factor

    # lr_actor = 0.0003       # learning rate for actor network
    # lr_critic = 0.001       # learning rate for critic network

    # random_seed = 0         # set random seed if required (0 = no random seed)
    # #####################################################

    # # state space dimension
    # state_dim = env.state_space.shape[0]

    # # action space dimension
    # action_dim = env.action_space.n

    # # initialize a PPO agent
    # ppo_agent = ppo.PPO(env, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # ppo_agent.train(max_training_timesteps, random_seed, max_ep_len, update_timestep, log_freq, print_freq, save_model_freq)

    ###############################################################################################

    # # To play the game using a trained model
    # ##DQN
    # env = FlappyBirdEnv()
    # model = deep_q_n.DQNAgent(env)
    # model.ai_play('models/TopModel/50.pt')

    # ###############################################################################################

    ##PPO
    env = FlappyBirdEnv()

    ################## hyperparameters ##################
    max_ep_len = 3000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 50    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = FlappyBirdEnv()

    # state space dimension
    state_dim = env.state_space.shape[0]

    # action space dimension
    action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = ppo.PPO(env, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    ppo_agent.test(total_test_episodes, max_ep_len)