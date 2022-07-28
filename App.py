import pygame
import time

from constants import *
from grid import Grid
from snake import Snake


class App:
    def __init__(self):
        pygame.init()
        self.surface = pygame.display.set_mode(WINDOW_RES)
        self.clock = pygame.time.Clock()

        self.time_prev = time.time()
        self.speed = 5
        self.snakeDirection = -1

        self.grid = Grid(self.surface)
        self.snake = Snake(self.surface)


    def run(self):
        self.surface.fill(pygame.Color(CL_BLACK))

        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.snakeDirection = 0
                    elif event.key == pygame.K_RIGHT:
                        self.snakeDirection = 1
                    elif event.key == pygame.K_DOWN:
                        self.snakeDirection = 2
                    elif event.key == pygame.K_LEFT:
                        self.snakeDirection = 3


            if (time.time() - self.time_prev) > 1 / self.speed:
                self.snake.setDirection(self.snakeDirection)
                self.snake.moveForward()
                self.time_prev = time.time()
            self.grid.draw()
            self.snake.draw()
            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    app = App()
    app.run()
