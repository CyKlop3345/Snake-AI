import pygame
from constants import *


class Snake:
    def __init__(self, surface):
        self.surface = surface

        self.pos = [ [GRID_RES[0]//2,GRID_RES[1]//2], [-1,-1], [-1,-1] ]
        self.length = len(self.pos)
        self.direction = -1  # up    -- 0
                             # right -- 1
                             # down  -- 2
                             # left  -- 3


    def setDirection(self, direction):
        if (self.direction%2 == direction%2):
            return
        self.direction = direction


    def moveForward(self):
        if self.direction == -1:
            return

        for i in range(self.length-1, 0, -1):
            self.pos[i][0] = self.pos[i-1][0]
            self.pos[i][1] = self.pos[i-1][1]

        if self.direction == 0:
            self.pos[0][1] -= 1
        elif self.direction == 1:
            self.pos[0][0] += 1;
        elif self.direction == 2:
            self.pos[0][1] += 1;
        elif self.direction == 3:
            self.pos[0][0] -= 1;


    def draw(self):
        for pos in self.pos:
            if pos == self.pos[0]:
                color = CL_HEAD
            else:
                color = CL_WHITE
            pygame.draw.rect(self.surface, color,
                            (pos[0]*CELL_SIZE +1, pos[1]*CELL_SIZE +1, CELL_SIZE-2, CELL_SIZE-2))
