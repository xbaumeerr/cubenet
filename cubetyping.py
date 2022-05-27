import typing
from rubik.cube import Cube

ECube_type = typing.NewType('ECube_type', object)
ECube_type.__doc__ = \
    """
    ECube type
    """

CubeStr      = typing.NewType('CubeStr', str)
CubeStr.__doc__= \
    """
    String representation for the cube. The input of Cube class as string.
    """

CubeLike     = ECube_type | Cube
CubeLike.__doc__ = \
    """
    Objects of type ECube or Cube
    """

TurnStr      = typing.TypeVar('TurnStr', str, chr)
TurnStr.__doc__ = \
    """
    String representation of cube turn. The first letter of turn.
    Possible values for clockwise turns: 
        U - up, 
        D - down,
        R - right, 
        L - left,
        F - front,
        B - back.
    For counter-clockwise turns add ' sign or i at the end of the turn:
        U' or Ui - up counter-clockwise turn,
        R' or Ri - right counter-clockwise turn,
    etc.
    """

CubeNetInput = typing.TypeVar('CubeNetInput', CubeStr, CubeLike, typing.Iterable[CubeLike], typing.Iterable[CubeStr])
CubeNetInput.__doc__ = \
    """
    Type of CubeNet feed foward input.
    """

TrainingType = typing.NewType('TrainingType', str)
TrainingType.__doc__ = \
    """
    Type of training
    """