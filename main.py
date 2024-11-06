import asyncio

from src.flappybird_env import FlappyBirdEnv
from src import deep_q_n

if __name__ == "__main__":
    # asyncio.run(Flappy().start())

    env = FlappyBirdEnv()
    model = deep_q_n.DQNAgent(env)
    model.train(3000)
