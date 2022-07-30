import pygame

from constants import *

class Snake:
    def __init__(self, surface):
        # To draw from this file
        self.surface = surface
        # Snake parameters
        # Default state is [-1,-1]
        self.pos = [ [GRID_RES[0]//2,GRID_RES[1]//2], [-1,-1], [-1,-1] ]
        self.length = len(self.pos)
        self.direction = -1
            # stop  -- -1
            # up    -- 0
            # right -- 1
            # down  -- 2
            # left  -- 3
            # horizontal -- even, vertical -- odd
        self.isDied = 0


    def moveForward(self):
        # Look exception
        if self.direction == -1:
            return
        # Move secondary segments
        for i in range(self.length-1, 0, -1):
            self.pos[i][0] = self.pos[i-1][0]
            self.pos[i][1] = self.pos[i-1][1]
        # Move main segment
        if self.direction == 0:
            self.pos[0][1] -= 1
        elif self.direction == 1:
            self.pos[0][0] += 1;
        elif self.direction == 2:
            self.pos[0][1] += 1;
        elif self.direction == 3:
            self.pos[0][0] -= 1;
        # Check for the death
        self.checkDeath()


    def turn(self, turnDirection):
        # -1 -- left (local)
        # +1 -- right
        # 0  -- pass
        self.direction += turnDirection


    def draw(self):
        # Drawing all segments. Main has other color
        for pos in self.pos:
            if pos == self.pos[0]:
                color = CL_HEAD
            else:
                color = CL_WHITE
            pygame.draw.rect(self.surface, color,
                            (pos[0]*CELL_SIZE +1, pos[1]*CELL_SIZE +1, CELL_SIZE-2, CELL_SIZE-2))


    def setDirection(self, direction):
        # look diference between prev and next value of direction
        # left-right or up-down will give an exception
        if (self.direction%2 == direction%2 and self.direction != -1):
            return
        # successful changing
        self.direction = direction


    def getPos(self):
        return self.pos[0]


    def getStatus(self):
        return self.isDied


    def checkDeath(self):
        # Check for the death
            # self-eating
        for i in range(1, self.length):
            if self.pos[0] == self.pos[i]:
                self.isDied = 1
            # Barrier check
        if self.pos[0][0] < 0 or self.pos[0][0] > GRID_RES[0]-1 or self.pos[0][1] < 0 or self.pos[0][1] > GRID_RES[1]-1:
            self.isDied = 1


    def eatingApple(self):
        self.pos.append([-1,-1])
        self.length += 1
