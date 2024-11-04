import gym
from gym import spaces
import numpy as np
import random
import time
import asyncio
import pygame
from .flappy import Flappy

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        
        # original game instance
        self.game = Flappy()

        # To flap or not to flap: 0 for no flap, 1 for flap
        self.action_space = spaces.Discrete(2)

        # continuous observation space
        # the limits should be adjusted later based on the actual variables somehow
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -512, -512]),
            high=np.array([512, 400, 512, 512]),
            shape=(4,),
            dtype=np.float32
            )

    def reset(self):
        # reset to initial state
        asyncio.run(self.game.reset())
        
        # initial state observations
        return self._get_state()
   
    def step(self, action):

        # self.frames_since_last_action += 1

        terminated = self.game.tick(action == 1) # True if action -> flap
        state = self._get_state()
        
        reward = 1 # 1 for not dying
        if self.game.player.collided_pipe(self.game.pipes):
            reward -= 10 
        if self.game.player.collided_floor(self.game.floor):
            reward -= 10
        # if self.game.player.cy <= 0:
        #      reward -= 5
        for i, pipe in enumerate(self.game.pipes.upper):
            if self.game.player.crossed(pipe):
                reward += 50  # Reward for passing a pipe
                self.pipe_count += 1
                print(f"Passed pipe: {self.pipe_count}")

        # self.game._draw_observation_points(obs)
        return state, reward, terminated, False

    # não sei se não é melhor implementar a função aqui logo
    def _get_state(self):
        pos, next_h, next_v_l, next_v_u = self.game.game_state()
        return np.array([pos, next_h, next_v_l, next_v_u])