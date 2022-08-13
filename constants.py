
FPS = 60

E_constant = 2.718
# Size of cells in the grid
CELL_SIZE = 25
RES_workspace = [1000, 1000]  # main surface resolution (grid, snake, aplles, etc)
RES_aiVisual = [0 ,1000]   # surface resolution for ai work visualization
RES = [RES_workspace[0] + RES_aiVisual[0], RES_workspace[1]]

# Count of cells in colomns and rows
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
