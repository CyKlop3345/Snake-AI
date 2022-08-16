import pygame
import time
from pathlib import Path

from constants import *
from grid import Grid
from snake import Snake
from apple import Apple
from AI import AI


class App:
    def __init__(self):

        pygame.init()

        # Screen and urface
        self.screen = pygame.display.set_mode(RES)          # main screen
        self.surf_back = pygame.Surface((RES_workspace))    # background (grid)
        self.surf_front = pygame.Surface((RES_workspace), pygame.SRCALPHA, 32)   # front (snake, aplles)
        self.surf_front.set_colorkey((0,0,0))               # all black will be transport into transparent
        self.surf_aiVisual = pygame.Surface((RES_aiVisual)) # side surface (ai visualization)
        self.surf_aiVisual.fill(CL_BACK)


        self.clock = pygame.time.Clock()
        self.time_prev = time.time()    # For creating a timer

        # Saving settings
        self.image_path = Path.cwd() / "Pictures\Snake"
        if not self.image_path.is_dir():    # create folders
            self.image_path.mkdir()
        self.image_count = 0            # num of current image for saving

        # "Game" settings
        self.RUNNING_AI = 1
        self.speed_init = 5             # start speed of the "game"
        self.speed = self.speed_init    # current speed of the "game"
        self.snakeNewDirection = -1     # "stop" in default

        # Objects
        self.grid = Grid(self.surf_back)
        self.snake = Snake(self.surf_front)
        self.apples = []
        for i in range(1):  # count of apples
            self.apples.append(Apple(self.surf_front))
        self.snake.setApples(self.apples)
        self.AI = AI(self.surf_aiVisual)
        self.AI.set_draw_property()


    def reset(self):
        # exit()
        # Reset App
        # Objects
        self.snake.__init__(self.surf_front)
        for apple in self.apples:
            apple.__init__(self.surf_front)

        # Settings
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
                self.AI.draw_update()
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
            self.clock.tick(60)#FPS


    def get_imput(self):
        for event in pygame.event.get():
            # Exit
            if event.type == pygame.QUIT:
                exit() # Exit button
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    exit()
                # Snake controll by arrow buttons (only if self.RUNNING_AI == 0)
                elif event.key == pygame.K_UP:
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

        self.surf_front.fill((0,0,0,0))

        self.grid.draw()
        for apple in self.apples:
            apple.draw()
        self.snake.draw()

        self.surf_back.blit(self.surf_front, (0, 0))
        self.screen.blits(( (self.surf_back, (RES_aiVisual[0], 0)),
                            (self.surf_aiVisual, (0, 0)) ))

        pygame.display.update()


    def save_screen(self):
        # Saving images of the screen (slower)
        pygame.image.save(self.screen, self.image_path / f'snake-{self.image_count}.png')
        self.image_count += 1


if __name__ == "__main__":
    # Start there!
    app = App()
    app.run()
