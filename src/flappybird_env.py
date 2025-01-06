import gym
from gym import spaces
import numpy as np
import random
import time
import asyncio
import pygame
from .flappy import Flappy

class FlappyBirdEnv(gym.Env):
    def __init__(self, fps=30):
        
        # original game instance
        self.game = Flappy(fps=fps)

        # To flap or not to flap: 0 for no flap, 1 for flap
        self.action_space = spaces.Discrete(2)

        # continuous observation space
        self.state_space = spaces.Box(
            low=np.array([0, -10, 0, 0, 0]),
            high=np.array([512, 10, 300, 512, 512]),
            shape=(5,),
            dtype=np.float32
            )

    def reset(self):
        # reset to initial state
        self.game.reset()
        
        # initial state observations
        return self._get_state()
   
    def step(self, action):

        # self.frames_since_last_action += 1

        terminated = self.game.tick(action == 1) # True if action -> flap
        state = self._get_state()
        
        ######## REWARDS ########
        reward = 1 # 1 for not dying

        if self.game.player.collided(self.game.pipes, self.game.floor):
            reward -= 100

        for i, pipe in enumerate(self.game.pipes.upper):
            if self.game.player.crossed(pipe):
                reward += 10  # Reward for passing a pipe
        
        if state[0] >= state[4] and state[0] <= state[3]:
            reward += 5 # Reward for staying within the pipe gap

        if self.game.score.score % 5 == 0 and self.game.score.score != 0:
            reward += 100
        
        return state, reward, terminated, False

    def _get_state(self):
        pos, vel, next_h, next_l_y, next_u_y = self.game.game_state()
        return np.array([pos, vel, next_h, next_l_y, next_u_y], dtype=np.float32)