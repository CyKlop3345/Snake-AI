import pygame
import time
from pathlib import Path

from constants import *
from grid import Grid
from snake import Snake
from apple import Apple
from AI import AI


class App:

    # Initialization
    def __init__(self):

        # Pygame settings
        pygame.init()
        self.clock = pygame.time.Clock()


        # Screen and surface
        self.screen = pygame.display.set_mode(RES)          # main screen
        self.surf_back = pygame.Surface((RES_workspace))    # background (grid)
        self.surf_front = pygame.Surface((RES_workspace), pygame.SRCALPHA, 32)   # front (snake, aplles)
        self.surf_front.set_colorkey((0,0,0))               # all black will be transport into transparent
        self.surf_aiVisual = pygame.Surface((RES_aiVisual)) # side surface (ai visualization)
        self.surf_aiVisual.fill(CL_BACK)


        # "Game" settings
        self.RUNNING_AI = 1             # 0 -- control by player
                                        # 1 -- control by AI

        self.speed_init = 5             # start speed of the "game"
        self.speed = self.speed_init    # current speed of the "game"
        self.snakeNewDirection = -1     # "stop" in default
        self.time_prev = time.time()    # timer for timeout in controll by player


        # Objects
        self.grid = Grid(self.surf_back)
        self.snake = Snake(self.surf_front)
        self.apples = []
        for i in range(1):  # count of apples
            self.apples.append(Apple(self.surf_front))
        self.snake.setApples(self.apples)
        self.AI = AI(self.surf_aiVisual)
        self.AI.set_draw_property()


        # Saving settings
        self.image_path = Path.cwd() / "Pictures\Snake"
        if not self.image_path.is_dir():    # create folders if it not exist
            self.image_path.mkdir()
        self.image_count = 0                # num of current image for saving (will paste into a filename)

    # Reinitialization
    def reset(self):
        # exit()
        # Reset App
        # Objects
        self.snake.__init__(self.surf_front)
        for apple in self.apples:
            apple.__init__(self.surf_front)


        # Settings
        # self.time_prev = time.time()
        self.speed = self.speed_init
        self.snakeNewDirection = -1



    # Run app (main loop)
    def run(self):

        # Main loop
        while True:

            # Input of the keyboard
            self.get_imput()


            # Choosing control mode
            # by Player
            if self.RUNNING_AI == 0:
                self.control_player()

            # by AI
            elif self.RUNNING_AI == 1:
                self.controll_AI()


            # Check snake status
            self.check_status()

            # Drawing
            self.draw()

            # Saving screen image
            # self.save_screen()

            #
            self.clock.tick(FPS)



    # control by Player
    def control_player(self):
        # Moving the snake with timeout (timeout = 1 / "gameSpeed")
        if (time.time() - self.time_prev) > 1 / self.speed:
            self.time_prev = time.time()
            self.snake.setDirection(self.snakeNewDirection)
            self.snake.moveForward()

    # control by AI
    def controll_AI(self):
        self.snake.calc_input_layer()
        input_layer = self.snake.get_input_layer()
        self.AI.run(input_layer)
        direction = self.AI.get_choice()
        self.AI.draw_update()
        self.snake.turn(direction)
        self.snake.moveForward()



    # Keyboard input
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

    # Check snake status (death or eating)
    def check_status(self):
        status = self.snake.getStatus()
        # Check status
        # Check for the death
        if status == -1:
            self.reset()
        # Check for the eating
        if status == 1:
            self.speed += 0.25

    # Draw all objects
    def draw(self):
        # Clear front
        self.surf_front.fill((0,0,0,0))

        # Draw back
        self.grid.draw()
        # Draw front
        for apple in self.apples:
            apple.draw()
        self.snake.draw()

        # Bliting
        self.surf_back.blit(self.surf_front, (0, 0))
        self.screen.blits(( (self.surf_back, (RES_aiVisual[0], 0)),
                            (self.surf_aiVisual, (0, 0)) ))

        # Screen updating
        pygame.display.update()

    # Save screen into a file
    def save_screen(self):
        # Saving images of the screen (slower)
        pygame.image.save(self.screen, self.image_path / f'snake-{self.image_count}.png')
        self.image_count += 1



# Start point
if __name__ == "__main__":
    # Start there!
    app = App()
    app.run()
