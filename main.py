import asyncio

from src.flappybird_env import FlappyBirdEnv
from src import deep_q_n
from src.flappy import Flappy

if __name__ == "__main__":

    # # To run the game
    # asyncio.run(Flappy().start())

    # # To train a model
    # env = FlappyBirdEnv()
    # model = deep_q_n.DQNAgent(env)
    # model.training(10000)

    # To play the game using a trained model
    env = FlappyBirdEnv()
    model = deep_q_n.DQNAgent(env)
    model.ai_play('100.pt')