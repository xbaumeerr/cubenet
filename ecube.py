import re
import typing
import numpy as np

from operator  import xor
from functools import reduce
from itertools import product

from rubik.cube   import Cube

from numpy.typing import ArrayLike
from cubetyping   import CubeLike, TurnStr

from defaults     import CUBE_TURNS, ROTATION_TURNS, SKIP_TURN, CUBE_TURNSi, ROTATION_TURNSi, SCRAMBLE_TURNSi, DEFAULT_CUBE_STR, DEFAULT_COLORS_IDX


MOVES_PATTERN = re.compile(rf'(?i)[{"".join(CUBE_TURNS) + "-"}][0-9]*[\'i]*')

class ECube(Cube):
    def __init__(self, cube_str : str = DEFAULT_CUBE_STR):
        super().__init__(cube_str)
        self._initial = self.flat_str()

    @staticmethod
    def _transform_turn_function(func : typing.Callable[[Cube], None], name : str) -> typing.Callable[[CubeLike], CubeLike]:
        """
        Generates new turn function of class Cube, 
        so the turn will be applied to copy a cube and result will be returned
        """
        def _new_turn_function(self) -> CubeLike:
            cube = self.copy()
            getattr(Cube, name)(cube)
            return cube
        setattr(_new_turn_function, "__doc__", f"Apply {name} turn to copy of the cube and return resulted cube")
        setattr(_new_turn_function, "__name__", name)
        return _new_turn_function
        
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other : CubeLike) -> bool:
        return self.flat_str() == other.flat_str()

    def to_Cube(self) -> Cube:
        """
        Get Cube class version of ECube object

        Returns
        -------
        cube : Cube
            Cube class object
        """
        return Cube(self.flat_str())

    @staticmethod
    def get_default_cube() -> CubeLike:
        """
        Get solved version of cube with default colors using side colors parameter to generate it.
        Default cube is
            YYY
            YYY
            YYY
        BBB RRR GGG WWW
        BBB RRR GGG WWW
        BBB RRR GGG WWW
            OOO
            OOO
            OOO

        Returns
        -------
        cube : ECube
            Solved version of cube.
        """
        return ECube(DEFAULT_CUBE_STR)
    
    def reset(self) -> CubeLike:
        """
        Get the initial state of cube

        Returns
        -------
        cube : ECube
            Cube in initial state
        """
        return ECube(self._initial)

    def reset_(self):
        """
        Reset cube to it's initial state
        """
        self = ECube(self._initial)

    def copy(self) -> CubeLike:
        """
        Get a copy of cube
        """
        return ECube(self.flat_str())

    @staticmethod
    def get_scramble_turns(n_turns : int, p : typing.Iterable = None, with_rotations : bool = False, custom_turns_list : typing.Iterable  = None) -> typing.Iterable[TurnStr]:
        """
        Get N random scramble turns

        Parameters
        ----------
        n_turns : int
            Amount of random rotations
        p : Iterable
            Probability distribution to pick turns
        with_rotations : bool
            Use X or Y rotations in scramble
        custom_turns_list : Iterable, optional
            If is not None, then the list will be used to get random turns

        Returns
        -------
            scramble : list
                List of turns to scramble cube 
        """
        scramble = []
        if n_turns == 0:
            return [SKIP_TURN]
        turns = CUBE_TURNSi if with_rotations else SCRAMBLE_TURNSi
        if custom_turns_list:
            turns = custom_turns_list
        pi    = np.zeros(len(turns)).astype(np.float32)
        pi[:] = 1.0/len(turns) if p is None else p
        p = pi
        for i in range(n_turns):
            scramble.append(np.random.choice(turns, 1, p=p).item())
        return scramble
    
    @staticmethod
    def get_rotation_turns(n_rotations : int, p : typing.Iterable = None, clockwise_only=True):
        turns_list = ROTATION_TURNS if clockwise_only else ROTATION_TURNSi
        return ECube.get_scramble_turns(n_rotations, p=p, custom_turns_list=turns_list)

    @staticmethod
    def reverse_turns(turns : TurnStr | typing.Iterable[TurnStr]) -> typing.Iterable[TurnStr]:
        """
        For each turn get reversed one in reversed order
        
        Parameters
        ----------
        turns : str | Iterable
            One turn or list of turns to reverse
        
        Returns
        -------
        reversed_turns : Iterable
            List of reversed turns
        """
        turns = ECube._prepare_turns(turns)
        reversed_turns = []
        if isinstance(turns, str):
            turns = [turns]
        for turn in reversed(turns):
            if turn != SKIP_TURN:
                if not turn.endswith('i'):
                    turn = turn + 'i'
                else:
                    turn = turn[:-1]
            reversed_turns.append(turn)
        return reversed_turns

    def scramble(self, n_rotations : int, reset : bool = True, p : typing.Iterable = None, with_rotations : bool = True) -> CubeLike:
        """
        Returns scrambled copy of cube by N random rotations

        Parameters
        ----------
        n_rotations : int
            Amount of random rotations
        reset : bool, optional
            If True, then the cube will be reseted to its start state before scramble
        p : Iterable
            Probability distribution to pick turns
        with_rotations : bool   
            Use X or Y rotations in scramble
        
        Returns
        -------
        cube : Cube
            Scrambled cube
        """
        cube = self.copy()
        if reset: 
            cube = self.reset()
        cube = cube.turn(ECube.get_scramble_turns(n_rotations, p, with_rotations))
        return cube
    
    def scramble_(self, n_rotations : int, reset : bool = True, p : typing.Iterable = None, with_rotations : bool = True):
        """
        Applies scramble to cube by N random rotations

        Parameters
        ----------
        n_rotations : int
            Amount of random rotations
        reset : bool, optional
            If True, then the cube will be reseted to its start state before scramble
        p : Iterable
            Probability distribution to pick turns
        with_rotations : bool
            Use X or Y rotations in scramble
        
        Returns
        -------
        cube : Cube
            Scrambled cube
        """
        if reset: 
            self.reset_()
        self.turn_(ECube.get_scramble_turns(n_rotations, p, with_rotations))

    @staticmethod
    def _prepare_turns(turns : typing.Iterable[TurnStr]) -> typing.Iterable[TurnStr]:
        """
        Prepare turns for input turns
        """
        if isinstance(turns, list):
            turns = ' '.join(turns)
        turns  = re.findall(MOVES_PATTERN, turns)
        turnsi = []
        for turn in turns:
            if turn.endswith("'"):
                turn = turn.replace("'", 'i')
            repetitions = re.findall(r'\d', turn)
            if repetitions:
                repetitions = int(repetitions[0])
                turn = ''.join(t for t in turn if not t.isdigit())
                r_turns = []
                for _ in range(repetitions):
                    r_turns.append(turn)
                turnsi.extend(r_turns)
            else:
                turnsi.append(turn)
        return turnsi

    def turn(self, turns : typing.Iterable[TurnStr] | TurnStr, reset : bool = False) -> CubeLike:
        """
        Apply turns to copy of cube and return it

        Parameters
        ----------
        turns : list | str
            Single turn or list of turns in letter notation to perform on cube in it's current state
        reset : bool, optional
            If True, then the cube will be reseted to solved state before moves

        Returns
        -------
        cube : ECube
            Cube with applied moves
        """
        turns = ECube._prepare_turns(turns)
        cube = self.copy()
        if reset: 
            cube = self.reset()
        for turn in turns:
            if turn == SKIP_TURN: continue
            getattr(Cube, turn)(cube)
        return cube
    
    def turn_(self, turns : typing.Iterable[TurnStr] | TurnStr, reset : bool = False):
        """
        Apply turns to cube itself

        Parameters
        ----------
        turns : list | str
            Single turn or list of turns in letter notation to perform on cube in it's current state
        reset : bool, optional
            If True, then the cube itself will be reseted to solved state before moves
        """
        if isinstance(turns, list):
            turns = ' '.join(turns)
        turns = re.findall(MOVES_PATTERN, turns)
        if reset: 
            self.reset_()
        for turn in turns:
            if turn == SKIP_TURN: continue
            getattr(Cube, turn)(self)
    
    def cube4D_clr(self) -> ArrayLike:
        """
        Generate 4D matrix representaion of cube.
        On each X, Y, Z coordinate you get list of colors of piece.
        The origin of coordinate space (0, 0, 0) is the cubes front left top corner piece.
        The middle piece of coordinates (1, 1, 1) is with list [None, None, None], that means it has no colors.

        Returns
        -------
        cube4D : 3x3x3x3 matrix
            Cube 4D representation with lists of colors
        """
        cube4D = np.zeros((3,3,3)).astype(object)
        coordinates_combinations = product(*[[-1,0,1]]*3)
        for x, y, z in coordinates_combinations:
            xi, yi, zi = x+1, y+1, abs(z-1)
            piece = self.get_piece(x,y,z)
            if piece is None:
                cube4D[xi, yi, zi] = [None]*3
            else:
                cube4D[xi, yi, zi] = piece.colors
        return cube4D

    def cube4D(self) -> ArrayLike:
        """
        Generate 4D matrix of cube representation.
        On each X, Y, Z coordinate you get normals (3-vectors) of piece multiplied by XOR of color ids of piece.
        
        The origin of coordinate space (0, 0, 0) is the cubes front left top corner piece.
        The middle piece of coordinates (1, 1, 1) is with ID 0 and has no colors.

        Returns
        -------
        cube4D : 3x3x3x3 matrix
            Cube 4D representation with pieces id 3-vectors.
        """
        cube4D = np.zeros((3,3,3,3))
        coordinates_combinations = product(*[[-1,0,1]]*3)
        for x, y, z in coordinates_combinations:
            xi, yi, zi = x+1, abs(y-1), abs(z-1) # translate coordinates from origin in the center of cube to left top corner
            piece  = self.get_piece(x,y,z)
            if piece is None: 
                cube4D[xi, yi, zi] = np.array([0,0,0]).astype(np.float32)
                continue
            colors_idx = [DEFAULT_COLORS_IDX[c] for c in piece.colors]
            piece_idx  = reduce(xor, colors_idx, 0)
            normal  = np.array([x,y,z]).astype(np.float32)
            normal /= np.linalg.norm(normal)
            normal *= piece_idx
            cube4D[xi, yi, zi] = normal
        cube4D /= np.linalg.norm(cube4D)
        return cube4D.astype(np.float32)
        

ECube.__doc__ = "Expand of Cube class:\n" + ECube.__base__.__doc__
ECube.__init__.__doc__ = Cube.__init__.__doc__
for turn in CUBE_TURNSi:
        old_turn_function = getattr(ECube, turn)
        setattr(ECube, turn+'_', old_turn_function)
        setattr(ECube, turn, ECube._transform_turn_function(old_turn_function, turn))