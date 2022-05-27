from rubik.cube import Cube

"""
Possible cube turns notations
"""
# cube edges turns
SCRAMBLE_TURNS  = ('L', 'R', 'U', 'D', 'F', 'B', 'M', 'E') # possible clock-wise turns
SCRAMBLE_TURNSi = SCRAMBLE_TURNS + tuple(ri + 'i' for ri in SCRAMBLE_TURNS) # possible clock-wise and counter clock-wise turns

# cube rotation turns
ROTATION_TURNS  = ('X', 'Y')
ROTATION_TURNSi = ROTATION_TURNS + tuple(ri + 'i' for ri in ROTATION_TURNS)

# model output turns
SKIP_TURN = '-' # no turn at all (e.g. when cube is solved)
TURNS   = SCRAMBLE_TURNS + tuple('-')  # possible output turns without counter clock-wise turns
TURNSi  = SCRAMBLE_TURNSi + tuple('-') # possible output turns with counter clock-wise turns

# cube class turns
CUBE_TURNS  = SCRAMBLE_TURNS + ROTATION_TURNS
CUBE_TURNSi = SCRAMBLE_TURNSi + ROTATION_TURNSi 

"""
Default colors of cube
"""
C_UP    = 'Y'
C_LEFT  = 'B'
C_FRONT = 'R'
C_RIGHT = 'G'
C_DOWN  = 'W'
C_BACK  = 'O'

"""
Ordered default colors of a cube
"""
DEFAULT_COLORS     = (C_UP, C_LEFT, C_FRONT, C_RIGHT, C_DOWN, C_BACK)

"""
Dictionary to get ID of each ordered default color
"""
DEFAULT_COLORS_IDX = { DEFAULT_COLORS[i] : i**2 for i in range(len(DEFAULT_COLORS)) }
DEFAULT_COLORS_IDX[None] = 0

"""
Predefined default cube with default colors
"""
DEFAULT_CUBE = Cube(C_UP*9 + (C_LEFT*3+C_FRONT*3+C_RIGHT*3+C_BACK*3)*3 + C_DOWN*9)
DEFAULT_CUBE_STR = DEFAULT_CUBE.flat_str()