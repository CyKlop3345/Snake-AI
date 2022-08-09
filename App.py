import pygame
import time
import os

from constants import *
from grid import Grid
from snake import Snake
from apple import Apple
from AI import AI


class App:
    def __init__(self):
        pygame.init()
        # System settings
        self.surface = pygame.display.set_mode(WINDOW_RES)
        self.surface.fill(pygame.Color(CL_BLACK)) # Filling in black
        self.clock = pygame.time.Clock()
        self.image_count = 0
        self.RUNNING_AI = 1#  l f r bf lr    l f r
        self.time_prev = time.time()    # For creating a timer
        # "Game" settings
        self.speed_init = 5             # start speed of the "game"
        self.speed = self.speed_init    # current speed of the "game"
        self.snakeNewDirection = -1     # "stop" in default
        # Objects
        self.grid = Grid(self.surface)
        self.snake = Snake(self.surface)
        self.apples = []
        for i in range(1):              # count of apples
            self.apples.append(Apple(self.surface))
        self.snake.setApples(self.apples)
        self.AI = AI()


    def reset(self):
        exit()
        # Reset the Snake
        self.snake.__init__(self.surface)
        for apple in self.apples:
            apple.__init__(self.surface)

        self.time_prev = time.time()
        self.speed = self.speed_init
        self.snakeNewDirection = -1


    def run(self):
        # Main loop
        while True:
            # Input of the keyboard
            self.get_imput()

            # Player control
            if self.RUNNING_AI == 0:
                # Moving the snake with timeout (timeout = 1 / "gameSpeed")
                if (time.time() - self.time_prev) > 1 / self.speed:
                    self.time_prev = time.time()
                    self.snake.setDirection(self.snakeNewDirection)
                    self.snake.moveForward()
            # AI control
            else:
                input_layer = self.snake.get_input_layer()
                self.AI.run(input_layer)
                direction = self.AI.get_choice()
                self.snake.turn(direction)
                self.snake.moveForward()

            # Check status
            # Check for the death
            status = self.snake.getStatus()
            if status == -1:
                self.reset()
            # Check for the eating
            if status == 1:
                self.speed += 0.25

            # Drawing and saving
            self.draw()
            # self.save_screen()


    def get_imput(self):
        for event in pygame.event.get():
            # Exit
            if event.type == pygame.QUIT:
                exit() # Exit button
            if event.type == pygame.KEYDOWN:
                # Snake controll by arrow buttons (only if self.RUNNING_AI == 0)
                if event.key == pygame.K_UP:
                    self.snakeNewDirection = 0
                elif event.key == pygame.K_RIGHT:
                    self.snakeNewDirection = 1
                elif event.key == pygame.K_DOWN:
                    self.snakeNewDirection = 2
                elif event.key == pygame.K_LEFT:
                    self.snakeNewDirection = 3
                # Reset yhe "game"
                elif event.key == pygame.K_r:
                    self.reset()


    def draw(self):
        # Drawing
        self.grid.draw()
        for apple in self.apples:
            apple.draw()
        self.snake.draw()
        pygame.display.flip()
        self.clock.tick(FPS)


    def save_screen(self):
        # Saving images of the screen (slower)
        if not os.path.isdir(r"Pictures\Snake"):
            os.makedirs(r"Pictures\Snake")
        pygame.image.save(self.surface, os.path.join('Pictures\Snake',f'snake-{self.image_count}.png'))
        self.image_count += 1


if __name__ == "__main__":
    # Start there!
    app = App()
    app.run()
