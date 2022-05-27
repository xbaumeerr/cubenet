import os
import typing
import random

import torch
import numpy as np

from tqdm import tqdm
from rubik.solve import Solver
from collections import Counter
from numpy.typing import ArrayLike

from ecube import ECube
from logger import Logger
from model import CubeNet
from yparams import YParams
from defaults import SKIP_TURN
from cubetyping import CubeNetInput


def get_turns_p_reversed_distribution(turns_picking : Counter, initial_tuple : tuple) -> ArrayLike:
    """
    Get probability distribution for turns to pick in scramble. The most often used turns will be with the least probability to pick.

    Parameters
    ----------
    `turns_picking` : Counter
        Dictionary with turn letters as key and amount of picking this turn as value
    `initial_tuple` : tuple
        Tuple of keys of Counter object
    
    Returns
    -------
    `p` : ArrayLike
        Probability distribution
    """
    p = np.zeros(len(initial_tuple)).astype(np.float32)
    for i, turn in enumerate(initial_tuple):
        p[i] = turns_picking[turn]
    p = p.max() - p
    p /= p.sum()
    if np.isnan(p).any():
        p = None
    return p


def validate(
        model : CubeNet, 
        solved_cube : CubeNetInput = None, 
        validation_epochs : int = 10,
        min_scramble_turns : int = 1,
        max_scramble_turns : int = 3, 
        p_of_solved_cube   : float = 0.1,
        device : torch.DeviceObjType | str = 'cpu', 
        logger : Logger = None,
    ) -> typing.Tuple[float, ArrayLike]:
    """
    Validate CubeNet model by trying to solve scrambled cubes

    Parameters
    ----------
    `model` : CubeNet
        Model to validate
    `solved_cube` : ECube, optional
        Solved version of a cube
    `validation_epochs` : int
        Amount of epochs to evaluate
    `min_scramble_turns` : int
        Minimum amount of turns to scramble cube each epoch
    `max_scramble_turns` : int
        Minimum amount of turns to scramble cube each epoch
    `p_of_solved_cube` : float, optional
        probability of using blank scramble ['-'] instead of random
    `device` : torch.device
        Device to use
    `logger` : tqdm, optional
        logger to log to summary writer

    Returns
    -------
    `avg_model_vs_solver_coef` : float
        Average solve coefficient. Solve coef for each episode is calculated using formula:
            model_vs_solver_coef = (solver_num_turns - model_num_turns + 1)/solver_num_turns
        where model_num_turns is amount of turns used to trying to solve cube, solver_num_turns - maximum amount of turns for cube solving
        solver_num_turns is calculated using rubik.solve.Solver class
    `avg_model_vs_scramble_coef` : float
        Average of value:
            model_vs_scramble_coef = 2.0 / (1.0 + exp(model_num_turns-num_scrambles))
        where model_num_turns is amount of turns used to trying to solve cube, num_scrambles - amount of random scramble turns used to scramble cube
    """
    if solved_cube is None:
        solved_cube = ECube.get_default_cube()
    
    if logger:
        if not hasattr(logger, 'valiter'):
            logger.valiter = 0
    
    val_model = CubeNet()
    val_model = model.copy()
    val_model = val_model.to(device)
    val_model.eval()

    val_pbar   = tqdm(range(validation_epochs))
    val_logger = Logger(pbar=val_pbar)

    model_vs_solver_coefs   = []
    model_vs_scramble_coefs = []

    for epoch in val_pbar:
        num_rotations = random.randint(0,3)
        num_scrambles = random.randint(min_scramble_turns, max_scramble_turns)
        rotation = ECube.get_rotation_turns(num_rotations)
        scramble = ECube.get_scramble_turns(num_scrambles, with_rotations=False)
        skip_or_scramble = np.array([SKIP_TURN, scramble], dtype=object)
        scramble = np.random.choice(skip_or_scramble, p=[p_of_solved_cube, 1.0-p_of_solved_cube])
        
        cube     = solved_cube.copy()
        cube.turn_(rotation)
        cube.turn_(scramble)
        
        model_turns = Counter()
        solver      = Solver(cube.to_Cube())
        solver.solve()
        solver_num_turns = len(solver.moves) + 1
        
        for model_num_turns in range(1, solver_num_turns+1):
            output = val_model.forward(cube, device=device)
            turn   = CubeNet.translate(output)
            cube   = cube.turn(turn)
            model_turns.update(turn)
            if cube.is_solved():
                break
        
        model_vs_solver_coef   = (solver_num_turns - model_num_turns + 1)/solver_num_turns
        model_vs_scramble_coef = 2.0 / (1 + np.exp(model_num_turns-num_scrambles))
        model_vs_solver_coefs.append(model_vs_solver_coef)
        model_vs_scramble_coefs.append(model_vs_scramble_coef)

        mvso_coef_display = '-' if model_vs_solver_coef < 0.1 else round(model_vs_solver_coef, 4)
        mvsc_coef_display = '-' if model_vs_scramble_coef < 0.1 else round(model_vs_scramble_coef, 4)
        most_common       = f'{"two":4} most common turns {"-"*5} {model_turns.most_common(2)}' if model_vs_solver_coef < 0.1 else f'{"five":4} most common turns {model_turns.most_common(5)}'
        message = f'model vs solver coef {mvso_coef_display:6}, model vs scramble coef {mvsc_coef_display:6}, five most common turns {most_common}'
        val_logger.tqdmlog(message, add_iter_num=True)
        if logger:
            if logger.filepath:
                logger.filelog(message)
            if logger.use_tensorboard:
                logger.sw.add_scalar('Val/model vs solve coef', model_vs_solver_coef, logger.valiter)
                logger.sw.add_scalar('Val/model vs scramble coef', model_vs_scramble_coef, logger.valiter)
            logger.valiter += 1

    avg_model_vs_solver_coef   = np.array(model_vs_solver_coefs).mean()
    avg_model_vs_scramble_coef = np.array(model_vs_scramble_coefs).mean()
    
    return avg_model_vs_solver_coef, avg_model_vs_scramble_coef


def get_logf(logger):
    if logger:
        return lambda message, f=True, a=True, e=True: logger.tqdmlog(message, to_file=f, attention=a, add_iter_num=e)
    return None


def training_logger_preprocessing(params : YParams) -> Logger:
    """
    Prepare logger using training parameters

    Parameters
    ----------
    `params` : YParams
        training parameters
    
    Returns
    -------
    `logger` : Logger
        Logger object for logging during training
    """
    pbar = tqdm(range(params.load_epoch, params.num_epochs), total=params.num_epochs, initial=params.load_epoch)
    pbar.write('Do you want to clear logging file before proceeding? y/n')
    ans    = input(': ')
    clear  = 'y' in ans
    logger = Logger(
        log_dir=params.log_path, log_filename=params.model_name + '.log', 
        clear=clear, pbar=pbar, use_tensorboard=True,
        purge_step=params.load_epoch, filename_suffix=params.model_name
    )
    return logger


def save_model_step(model : CubeNet, params : YParams, epoch : int = 0, condition : bool = True, logger : Logger = None, last_loss : float = np.nan):
    """
    Save model during training.

    Parameters
    ----------
    `model` : CubeNet
        model to save
    `params` : YParams
        training parameters
    `epoch` : int
        current epoch
    `condition` : bool
        condition to save model
    `logger` : Logger
        logger to log training progress
    `last_loss` : float
        last loss for logging
    """
    if condition:
        path = os.path.join(params.save_path, f'{params.model_name}_{epoch}.chk')
        if logger:
            logger.tqdmlog(f'Saving model to {path}... Last loss {last_loss}', attention=True, add_iter_num=True, to_file=True)
        model.save(path)


def validation_step(
        model : CubeNet,
        params : YParams,
        solved_cube : ECube, 
        min_scramble_turns : int, 
        max_scramble_turns : int, 
        
        epoch : int = 0, 
        condition : bool = True, 
        device : torch.DeviceObjType = 'cpu', 
        logger : Logger = None
    ):
    """
    Validate model during training

    Parameters
    ----------
    `model` : CubeNet
        model to validate
    `params` : YParams
        training parameters
    `solved_cube` : ECube
        solved version of cube
    `min_scramble_turns` : int
        minimum amount of turns in scrambles
    `max_scramble_turns` : int
        maximum amount of turns in scrambles
    `epoch` : int, optional
        current epoch
    `condition` : bool, optional
        condition to validate model
    `device` : torch.device
        device to use during validation
    `logger` : Logger
        logger to use for logging
    """
    if condition:
        if logger:
            logger.tqdmlog(f'Starting validation...', attention=True, to_file=True, add_iter_num=True)
        avg_model_vs_solver_coef, avg_model_vs_scramble_coef = validate(
            model=model,
            solved_cube=solved_cube, 
            validation_epochs=params.validation_epochs, 
            min_scramble_turns=min_scramble_turns,
            max_scramble_turns=max_scramble_turns,
            p_of_solved_cube=params.p_of_solved_cube,
            device=device, 
            logger=logger
        )
        if logger:
            logger.tqdmlog(f'Validation average model vs solver coefficient: {avg_model_vs_solver_coef}, average model vs scramble coefficient: {avg_model_vs_scramble_coef}', attention=True, to_file=True, add_iter_num=True)
            logger.tblog('Val/avg model vs solver coef', avg_model_vs_solver_coef, epoch)
            logger.tblog('Val/avg model vs scramble coef', avg_model_vs_scramble_coef, epoch)
