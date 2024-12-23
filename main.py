import asyncio

from src.flappybird_env import FlappyBirdEnv
from src import deep_q_n
from src.flappy import Flappy
# from src import train

if __name__ == "__main__":

    # # To run the game
    # asyncio.run(Flappy().start())

    # # To train a model
    # ## DQN
    # env = FlappyBirdEnv()
    # model = deep_q_n.DQNAgent(env)
    # model.training(5000)

    # ##PPO
    # train()

    # To play the game using a trained model
    env = FlappyBirdEnv()
    model = deep_q_n.DQNAgent(env)
    model.ai_play('models/TopModel/50.pt')