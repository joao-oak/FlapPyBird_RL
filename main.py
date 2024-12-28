import asyncio
from src.flappybird_env import FlappyBirdEnv
from src import deep_q_n
from src.flappy import Flappy
from src import ppo

if __name__ == "__main__":

    fps = int(input("How many fps do you want? \n"))

    action = input("Choose what you want to do: \n[1] Play \n[2] Train AI \n[3] Play with AI \n")

    if action == "1":
        # To run the game
        asyncio.run(Flappy(fps=fps).start())

    elif action == "2":
        algo = input("Choose the algorithm you want to use: \n[1] DQN \n[2] PPO \n")

        # To train a model
        if algo == "1":
            # DQN
            env = FlappyBirdEnv(fps=fps)
            model = deep_q_n.DQNAgent(env)
            model.training(5000)

        elif algo == "2":
            #PPO
            for lr_actor in [0.0001,0.0003,0.0005,0.001]:
                for lr_critic in [0.001, 0.005, 0.009]:
                    for update_timestep in [1000, 4000, 9000]:
                        print(f"actor:{lr_actor}, critic:{lr_critic}, update_timestep:{update_timestep}")
                        env = FlappyBirdEnv(fps=fps)
                        ppo_agent = ppo.PPO(env, lr_actor=lr_actor, lr_critic=lr_critic, update_timestep=update_timestep, gamma=0.99, K_epochs=80, eps_clip=0.2, action_std_init=0.6)
                        ppo_agent.train(random_seed=5, episodes=3000)

    elif action == "3":
        algo = input("Choose the algorithm you want to use: \n[1] DQN \n[2] PPO \n")

        # To play the game using a trained model
        if algo == "1":
            # DQN
            env = FlappyBirdEnv(fps=fps)
            model = deep_q_n.DQNAgent(env)
            model.ai_play('models/TopModel/50.pt')

        elif algo == "2":
            # PPO
            env = FlappyBirdEnv(fps=fps)
            ppo_agent = ppo.PPO(env, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2, action_std_init=0.1)
            ppo_agent.test()