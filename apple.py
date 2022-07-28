import pygame
from random import randint

from constants import *

class Apple:
    def __init__(self, surface):
        # To draw from this file
        self.surface = surface
        # Default state is [-1,-1]
        self.pos = [-1,-1]


    def getPos(self):
        return self.pos


    def randPos(self):
        self.pos[0] = randint(0, GRID_RES[0]-1)
        self.pos[1] = randint(0, GRID_RES[1]-1)


    def draw(self):
        pygame.draw.rect(self.surface, CL_RED,
                        (self.pos[0]*CELL_SIZE +1, self.pos[1]*CELL_SIZE +1, CELL_SIZE-2, CELL_SIZE-2))
