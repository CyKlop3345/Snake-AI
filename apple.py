import pygame
from random import randint
import numpy as np

from constants import *

class Apple:
    
    # Initialization
    def __init__(self, surface):
        # To draw from this file
        self.surface = surface
        # Default state is [-1,-1]
        self.pos = np.array([-1,-1])
        self.randPos()



    # Position randomisation
    def randPos(self):
        self.pos[0] = randint(0, GRID_RES[0]-1)
        self.pos[1] = randint(0, GRID_RES[1]-1)



    # Drawing aplle
    def draw(self):
        pygame.draw.rect(self.surface, CL_RED,
                        (self.pos[0]*CELL_SIZE +1, self.pos[1]*CELL_SIZE +1, CELL_SIZE-2, CELL_SIZE-2))



    # Getter
    def getPos(self):
        return self.pos
