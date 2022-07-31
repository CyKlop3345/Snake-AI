import pygame
import time

from constants import *
from grid import Grid
from snake import Snake
from apple import Apple
from AI import AI


class App:
    def __init__(self):
        pygame.init()
        self.surface = pygame.display.set_mode(WINDOW_RES)
        self.clock = pygame.time.Clock()

        self.time_prev = time.time() # For creating a timer
        self.speed = 4  # speed of the "game"
        self.snakeDirection = -1    # "stop" in default

        self.grid = Grid(self.surface)
        self.snake = Snake(self.surface)
        self.apple = Apple(self.surface)
        self.AI = AI()
        self.AI.setSnakeControl(self.snake)
        self.AI.addApple(self.apple)


    def reset(self):
        # Reset the Snake
        self.snake.__init__(self.surface)
        self.apple.__init__(self.surface)

        self.time_prev = time.time()
        self.speed = 4
        self.snakeDirection = -1
        # exit()


    def run(self):
        # Filling in black
        self.surface.fill(pygame.Color(CL_BLACK))
        # Main loop
        while True:

            for event in pygame.event.get():
                # Exit
                if event.type == pygame.QUIT:
                    exit() # Exit button
                # Snake controll by arrow buttons
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.snakeDirection = 0
                    elif event.key == pygame.K_RIGHT:
                        self.snakeDirection = 1
                    elif event.key == pygame.K_DOWN:
                        self.snakeDirection = 2
                    elif event.key == pygame.K_LEFT:
                        self.snakeDirection = 3

            # Moving the snake with timeout (timeout = 1 / "gameSpeed")
            if (time.time() - self.time_prev) > 1 / self.speed:
                self.snake.setDirection(self.snakeDirection)
                self.snake.moveForward()
                # self.AI.run()
                self.time_prev = time.time()
                # Check for the death
                if self.snake.getStatus() == 1:
                    self.reset()
                # Check for the eating
                if self.snake.getPos() == self.apple.getPos():
                    self.snake.eatingApple()
                    self.apple.randPos()
                    self.speed += 0.25
            # Drawing
            self.grid.draw()
            self.apple.draw()
            self.snake.draw()
            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    # Start there!
    app = App()
    app.run()
