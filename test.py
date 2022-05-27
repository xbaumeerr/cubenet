import torch
import random
import numpy as np
from tqdm  import tqdm
from ecube import ECube
from model import CubeNet
from utils import Logger, validate
from self_train import generate_batch

from defaults import TURNS, TURNSi, ROTATION_TURNS


if __name__ == '__main__':
    solved_cube = ECube.get_default_cube()
    model = CubeNet().eval()

    model.load(r'checkpoints\self_cubenet_from_2_to_6\self_cubenet_from_2_to_6_50000.chk')
    model.eval()
    solved_cube = ECube.get_default_cube()

    turns = 'R U L Ui'
    solve = []
    cube = solved_cube.turn(turns)
    print(cube)
    num_turns = 0
    r = []
    while num_turns < 200 and r != ['-']:
        r = model.predict(cube)
        solve.append(r)
        cube = cube.turn(r)
        num_turns += 1
        # if cube.is_solved() and r == ['-']:
            # break
    print(solve)