import pygame
import numpy as np

from constants import *

class Snake:
    def __init__(self, surface):
        # To draw from this file
        self.surface = surface
        # Snake parameters
        # Default state is [-1,-1]
        self.length = 3
        self.pos = np.array([[GRID_RES[0]//2, GRID_RES[1]//2]])
        for i in range(1, self.length):
            self.pos = np.append(self.pos, [[-1,-1]], axis=0)

        self.direction = -1
        ''' Hint: directions:
        -1 -- stop
        0  -- up
        1 -- right
        2 -- down
        3 -- left
        horizontal -- even, vertical -- odd
        '''

        self.local = None
        self.local_boards = None
        self.calc_local_cords()
        ''' Hint: directions (from world to local):
        local = [[forward   right   ]
                 [left      backward]]

        0 -- up     :   ( 0,-1) ( 1, 0)
                        (-1, 0) ( 0, 1)

        1 -- right  :   ( 1, 0) ( 0, 1)
                        ( 0,-1) (-1, 0)

        2 -- down   :   ( 0, 1) (-1, 0)
                        ( 1, 0) ( 0,-1)

        3 -- left   :   (-1, 0) ( 0,-1)
                        ( 0, 1) ( 1, 0)
        '''

        self.status = 0
        ''' Hint: status:
        0  -- nothing is happened
        1  -- finding apple
        -1 -- finding dificulty (death)
        '''

    def moveForward(self):
        # Look exception
        if self.direction == -1:
            return
        # Move secondary segments
        for i in range(self.length-1, 0, -1):
            self.pos[i][0] = self.pos[i-1][0]
            self.pos[i][1] = self.pos[i-1][1]
        # Move main segment
        self.pos[0] += self.local[0,0]
        # Check for the death
        self.checkDeath()
        self.checkEating()


    def calc_local_cords(self):
        ''' Hint in the init() '''
        if self.direction == -1:
            return
        self.local = np.array([ [ (0,-1), (1,0) ],
                                [ (-1,0), (0,1) ] ])

        self.local_boards = np.array([[-1, GRID_RES[0]],
                                      [-1, GRID_RES[1]] ])

        for i in range(self.direction):
            # turnung matrix left
            self.local = np.rot90(self.local)
            self.local_boards = np.rot90(self.local_boards)


    def draw(self):
        # Drawing all segments. Main has other color
        for pos in self.pos:
            if (pos == self.pos[0]).all():
                color = CL_HEAD
            else:
                color = CL_WHITE

            pygame.draw.rect(self.surface, color,
                            (pos[0]*CELL_SIZE +1, pos[1]*CELL_SIZE +1, CELL_SIZE-2, CELL_SIZE-2))


    def setDirection(self, direction):
        self.direction = direction
        if direction == -1:
            return
        self.calc_local_cords()


    def turn(self, turn):
        # -1 -- left
        # 1  -- right
        # 0  -- None
        if turn == 0:
            return
        self.direction += (4+turn) % 4      # Turn snake right
        # Turn matrix left
        self.local = np.rot90(self.local, turn)
        self.local_boards = np.rot90(self.local_boards, turn)


    def setApples(self, apples):
        self.apples = apples


    def getHeadPos(self):
        return self.pos[0]


    def getSegmPos(self):
        return self.pos


    def getStatus(self):
        status = self.status
        self.status = 0
        return status


    def checkDeath(self):
        # Check for the death
            # self-eating
        for i in range(1, self.length):
            if (self.pos[0] == self.pos[i]).all():
                self.status = -1
            # Barrier check
        if self.pos[0][0] < 0 or self.pos[0][0] > GRID_RES[0]-1 or self.pos[0][1] < 0 or self.pos[0][1] > GRID_RES[1]-1:
            self.status = -1


    def checkEating(self):
        for apple in self.apples:
            if (self.pos[0] == apple.getPos()).all():
                apple.randPos()
                self.pos = np.append(self.pos, [[-1,-1]], axis=0)
                self.length += 1
                self.status = 1


    def get_input_layer(self):
        if self.direction == -1:
            self.setDirection(0)

        # Reset input layer
        input_layer = [0,0,0,0,0]
        ''' Hint
        input_layer indexes:
        0 -- barrier worfard
        1 -- barrier leftside
        2 -- barrier rightside
        3 -- apple: -1 -- backward
                    0  -- none
                    1  -- forward
        4 -- apple: -1 -- left
                    0  -- none
                    1  -- right
        '''

        # Check for barrier (self segments or boundary)
        # Self segments

        for snakeSegment in self.pos:
            # Forward
            if ([self.pos[0] + self.local[0,0]] == snakeSegment).all():
                input_layer[1] = 1
            # Left
            if ([self.pos[0] + self.local[1,0]] == snakeSegment).all():
                input_layer[0] = 1
            # Right
            if ([self.pos[0] + self.local[0,1]] == snakeSegment).all():
                input_layer[2] = 1

        # Boundary
        # Forward
        if ((self.pos[0] + self.local[0,0]) == self.local_boards[0,0]).any():
            input_layer[1] = 1
        # Left
        if ((self.pos[0] + self.local[1,0]) == self.local_boards[1,0]).any():
            input_layer[0] = 1
        # Right
        if ((self.pos[0] + self.local[0,1])[0] == self.local_boards[0,1]).any():
            input_layer[2] = 1

        # Apple finding
        # finding nearest apple
        # apple_nearest
        apple_nearest_distance = 100
        for apple in self.apples:
            apple_distance = abs(self.pos[0][0] - apple.getPos()[0]) + abs(self.pos[0][1] - apple.getPos()[1])
            if apple_distance < apple_nearest_distance:
                apple_nearest_distance = apple_distance
                apple_nearest = apple
        # Identification the right side
        # Forward
        local_apple = np.zeros((2,2))
        offset = apple_nearest.getPos() - self.pos[0]
        if offset[0] < 0:
            local_apple[1,0] = 1
        elif offset[0] > 0:
            local_apple[0,1] = 1
        if offset[1] < 0:
            local_apple[0,0] = 1
        elif offset[1] > 0:
            local_apple[1,1] = 1

        for i in range(self.direction):
            # turnung matrix left
            local_apple = np.rot90(local_apple)

        # Forward
        if local_apple[0,0] == 1:
            input_layer[3] = 1
        # Backward
        elif local_apple[1,1] == 1:
            input_layer[3] = -1
        # Left
        if local_apple[1,0] == 1:
            input_layer[4] = -1
        # Right
        elif local_apple[0,1] == 1:
            input_layer[4] = 1

        return input_layer
