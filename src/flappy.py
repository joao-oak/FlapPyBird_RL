import asyncio
import sys

import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .entities import Background, Floor, GameOver, Pipes, Player, PlayerMode, Score, WelcomeMessage
from .utils import GameConfig, Images, Sounds, Window

import random

class Flappy:
    def __init__(self, fps=30):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        self.screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=self.screen,
            clock=pygame.time.Clock(),
            fps=fps,
            window=window,
            images=images,
            sounds=Sounds(),
        )

    async def start(self):
        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)
            await self.splash()
            await self.play()
            await self.game_over()

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
            event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        ##
        # last_flap_time = pygame.time.get_ticks()
        ##
        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()        

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            player_height, player_velocity, next_pipe_distance_h, next_pipe_l_y, next_pipe_u_y  = self.game_state()

            next_pipe_x = next_pipe_distance_h + self.player.x - 26 # the pipe sprite is 52 pixels wide

            # # Desenhinhos reward function
            # pygame.draw.line(self.screen, (255,0,0), (0, next_pipe_l_y), (288, next_pipe_l_y), 2)
            # pygame.draw.line(self.screen, (255,0,0), (0, next_pipe_u_y), (288, next_pipe_u_y), 2)
            # pygame.draw.circle(self.screen, (255,255,255), (next_pipe_x, (next_pipe_l_y+next_pipe_u_y)/2), 3)

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)

    def game_state(self):

        # getting the next pipe and player's horizontal distance to the next pipe
        next_pipe_distance_h = self.pipes.lower[0].x - self.player.x + 52 # passing the distance to the end of the pipe, not the beginning

        if next_pipe_distance_h > 0:
            next_pipe = self.pipes.lower[0]
        else:
            next_pipe = self.pipes.lower[1]
            next_pipe_distance_h = next_pipe.x - self.player.x + 52

        # player's vertical distance to the ceiling
        player_height = self.player.y

        # player's vertical velocity
        player_velocity = self.player.vel_y

        # next lower pipe's height
        next_pipe_l_y = next_pipe.y

        # next upper pipe's height
        next_pipe_u_y = next_pipe.y - self.pipes.pipe_gap

        # mid-point 
        next_pipe_mid = (next_pipe_l_y + next_pipe_u_y) /2 # not including for now

        return player_height, player_velocity, next_pipe_distance_h, next_pipe_l_y, next_pipe_u_y


    def reset(self):
        # from start()
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.welcome_message = WelcomeMessage(self.config)
        self.game_over_message = GameOver(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)

        # from play()
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

    def tick(self, flap_this_frame):
        # player input
        for event in pygame.event.get():
            if self.is_tap_event(event):
                flap_this_frame = True

            self.check_quit_event(event)

        if self.player.collided(self.pipes, self.floor):
            return True

        for i, pipe in enumerate(self.pipes.upper):
            if self.player.crossed(pipe):
                self.score.add()
                if self.score.score == 1000:
                    return True

        if flap_this_frame:
            self.player.flap()

        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        self.score.tick()
        self.player.tick()

        # # Desenhinhos
        # ht, vel, d_h, d_v_l, d_v_u = self.game_state()
        # next_pipe_y_l = self.pipes.lower[0].y
        # next_pipe_y_u = self.pipes.lower[0].y - self.pipes.pipe_gap
        # pygame.draw.line(self.screen, (255,0,0), (0, next_pipe_y_l), (288, next_pipe_y_l), 2)
        # pygame.draw.line(self.screen, (255,0,0), (0, next_pipe_y_u), (288, next_pipe_y_u), 2)

        pygame.display.update()
        # await asyncio.sleep(0)
        self.config.tick()
        return False