import pygame
from constants import *


class Grid:
    def __init__(self, surface):
        self.surface = surface


    def draw(self):
        for x in range(GRID_RES[0]):
            for y in range(GRID_RES[1]):
                pygame.draw.rect(self.surface, CL_GRAY,
                                (x*CELL_SIZE +1, y*CELL_SIZE +1, CELL_SIZE-2, CELL_SIZE-2))
