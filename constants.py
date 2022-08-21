import numpy as np

# Game settings
FPS = 144



# Resolutions
RES_workspace = [1000, 1000]    # main surface resolution (grid, snake, aplles, etc)
RES_aiVisual = [500, 1000]      # surface resolution for ai work visualization
RES = [RES_workspace[0] + RES_aiVisual[0], RES_workspace[1]]



CELL_SIZE = 25                  # Size of cells in the grid
# Resolution of the grid (game board)
GRID_RES = RES_workspace[0] // CELL_SIZE, RES_workspace[1] // CELL_SIZE



# Main colors
    # Snake colors
CL_WHITE = '#ffffff'
CL_HEAD = '#e0cec3'
    # Grid
CL_BLACK = '#000000'
CL_GRAY = '#121212'
    # Apple
CL_RED = '#cd2a4b'
    # Background
CL_BACK = (20,16,16)


# AI visual
    # nodes
CL_node_inactive = np.array((96,64,64)) # 0
CL_node_active = np.array((233,233,255)) # 1

    # Connections colors
CL_conect_inactive = np.array((255,64,0)) # -1
CL_conect_active = np.array((0,191,255)) # 1
