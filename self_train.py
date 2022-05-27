import typing
import random
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from collections import Counter

from ecube import ECube
from model import CubeNet
from cubetyping import CubeNetInput
from defaults import SKIP_TURN, SCRAMBLE_TURNSi, TURNSi

from logger  import Logger
from yparams import YParams
from utils   import get_logf, get_turns_p_reversed_distribution, save_model_step, validation_step


def generate_batch(
        solved_cube : ECube, 
        batch_size : int, 
        main_num_turns : int, 
        other_num_turns : typing.Iterable = [], 
        other_num_turns_ratio : float = 0.0,
        p_of_solved_cube : float = 0.0, 
        p : typing.Iterable = None
    ):
    """
    Generate batch of cube scrambles. Generation process scrambles cube and reverses it last scramble turn to use it as target.

    Parameters
    ----------
    `solved_cube` : ECube
        Solved version of a cube
    `batch_size` : int
        Size of batch
    `main_num_turns` : int
        Amount of scramble turns to use the most often
    `other_num_turns` : Iterable, optional
        List of amount of scramble turns to use in ratio with main amount of scramble turns
    `other_num_turns_ration` : float, optional
        Ration of other_num_turns in batch to use
    `p` : Iterable, optional
        Probability distribution to use
    `p_of_solved_cube` : float, optional
        probability of using blank scramble ['-'] instead of random

    Returns
    -------
    `cubes` : Iterable[CubeStr]
        cube strings of scrambled cubes
    `target_values` : torch.Tensor
        matrix of vectors. each vector has only one non-zero value 1.0 for solving turn.
    """
    # calculate batch sizes
    main_turns_batch_size  = int(batch_size * (1.0 - other_num_turns_ratio))
    other_turns_batch_size = int(batch_size * other_num_turns_ratio) if other_num_turns else 0
    batch_size = main_turns_batch_size + other_turns_batch_size

    # generate random rotations of cube
    get_random_rotations = lambda _: ECube.get_rotation_turns(random.randint(0,3))
    rotations  = list(map(get_random_rotations, range(batch_size)))
    
    # generate random scrambles of cube
    get_random_scrambles = lambda _, num_turns=main_num_turns: ECube.get_scramble_turns(num_turns, p=p)
    scrambles  = list(map(get_random_scrambles, range(main_turns_batch_size)))
    if other_num_turns:
        scrambles += list(map(lambda _: get_random_scrambles(_, random.choice(other_num_turns)), range(other_turns_batch_size)))
    # replace random scramble with no-turn scramble ([SKIP_TURN])
    p_of_skip_or_scramble = [p_of_solved_cube, 1-p_of_solved_cube]
    skip_or_scramble = lambda scramble: np.random.choice(np.array([SKIP_TURN, scramble], dtype=object), p=p_of_skip_or_scramble)
    scrambles  = list(map(skip_or_scramble, scrambles))
    
    # get randomly rotated cubes
    cubes = list(map(lambda rotation: solved_cube.turn(rotation), rotations))
    # get scrambled cubes
    cubes = np.array(list(map(lambda c_s: c_s[0].turn(c_s[1]).flat_str(), zip(cubes, scrambles))))
    
    # generate target values
    target_values = torch.tensor(list(map(lambda scramble: [0.0 if turn != ECube.reverse_turns(scramble)[0] else 1.0 for turn in TURNSi], scrambles)))
    # generate shuffle indecies
    shuffle_idx   = np.random.permutation(batch_size)
    
    return cubes[shuffle_idx].tolist(), target_values[shuffle_idx]


def generate_fully_random_batch(solved_cube : ECube, batch_size : int, min_scramble_turns : int, max_scramble_turns : int, 
        p : typing.Iterable = None,
        p_of_solved_cube : float = 0.0):
    """
    Generates batch with random amount of turns scrambled cubes.
    """
    rotations  = list(map(lambda _: ECube.get_rotation_turns(random.randint(0,3)), range(batch_size)))
    scrambles  = list(map(lambda _: ECube.get_scramble_turns(random.randint(min_scramble_turns, max_scramble_turns), p=p), range(batch_size)))

    p_of_skip_or_scramble = [p_of_solved_cube, 1-p_of_solved_cube]
    skip_or_scramble = lambda scramble: np.random.choice(np.array([SKIP_TURN, scramble], dtype=object), p=p_of_skip_or_scramble)
    scrambles  = list(map(skip_or_scramble, scrambles))

    cubes = list(map(lambda rotation: solved_cube.turn(rotation), rotations))
    cubes = np.array(list(map(lambda c_s: c_s[0].turn(c_s[1]).flat_str(), zip(cubes, scrambles))))

    target_values = torch.tensor(list(map(lambda scramble: [0.0 if turn != ECube.reverse_turns(scramble)[0] else 1.0 for turn in TURNSi], scrambles)))
    
    shuffle_idx   = np.random.permutation(batch_size)

    return cubes[shuffle_idx].tolist(), target_values[shuffle_idx]


def calculate_loss(
        model : CubeNet, 
        input_values : CubeNetInput, 
        target_values : torch.Tensor, 
        optimizer : optim.Optimizer, 
        loss_function : typing.Callable, 
        device : torch.DeviceObjType | str = 'cpu'
    ) -> typing.Tuple[torch.TensorType, torch.TensorType]:
    """
    Calculate loss

    Parameters
    ----------
    `model` : CubeNet
        model to train
    `input_values` : CubeNetInput
        model input
    `target_values` : torch.Tensor
        target values that are going to be used in loss calculation
    `optimizer` : torch.Optimizer
        optimizer to use for training
    `loss_function` : Callable
        loss function
    `device` : torch.Device
        device to use
    
    Returns
    -------
    `output_values` : torch.Tensor
        output of model
    `loss` : torch.Tensor
        calculated loss    
    """
    output_values = model.forward(input_values, device=device)
    target_values = target_values.to(device)
    loss = loss_function(output_values, target_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output_values, loss


def self_train(
        model : CubeNet,
        params : YParams,
        solved_cube  : ECube,
        
        device : torch.DeviceObjType = 'cpu',
        
        logger : Logger = None,

        optimizer : optim.Optimizer = optim.Adam, 
        loss_function : typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,

    ) -> CubeNet:
    """
    Train CubeNet model using self-supervised learning. 
    The batches are generated automaticly with progressing during training amount of scramble turns.

    Parameters
    ----------
    `model` : CubeNet
        Neural network model
    `params` : YParams,
        training parameters
    `solved_cube` : ECube
        Starting cube
    `device` : torch.device, optional
        device to use in training
    `logger` : Logger, optional
        logger to log progress
    `optimizer` : torch.optim.Optimizer
        Optimizer tu use in training
    `loss_function` : torch.LossFunction
        Loss function too calculate loss

    Returns
    -------
    `model` : CubeNet
        trained model
    """

    # HISTORY COLLECTIONS
    turns_picking = Counter({turn : 0 for turn in TURNSi})
    num_turns_progression = np.linspace(params.min_scramble_turns, params.max_scramble_turns, params.num_epochs).round().astype(np.int32)
    num_turns_used = []
    if params.load_epoch > 0:
        next_num_turns = num_turns_progression[params.load_epoch]
        num_turns_used = list(range(params.min_scramble_turns, next_num_turns))

    # LOGGER LOG FUNCTION
    logf = get_logf(logger)

    # OPTIMIZER INITIALIZATION
    optimizer = optimizer(params=model.parameters(), lr=params.lr)

    # MAIN LOOP
    current_num_turns = params.min_scramble_turns
    if logger:
        logger.tblog('Training/main num turns', current_num_turns, params.load_epoch)
    last_loss = np.nan
    if logger:
        epochs_range = logger.pbar
    else:
        epochs_range = range(params.load_epoch, params.num_epochs)
    
    for epoch in epochs_range:

        # SCRAMBLE NUM TURNS EPOCH PREPROCESSING
        if num_turns_progression[epoch].item() != current_num_turns:
            if logger:
                logf(f'Changing main number of turns to {current_num_turns + 1}... Last loss {last_loss}')
                logger.tblog('Training/main num turns', current_num_turns + 1, epoch)
            num_turns_used.append(current_num_turns)
        current_num_turns = num_turns_progression[epoch].item()

        # BATCH GENERATION
        p = get_turns_p_reversed_distribution(turns_picking, SCRAMBLE_TURNSi)
        cubes, target_values = generate_batch(
            solved_cube, 
            params.batch_size, 
            main_num_turns=current_num_turns, 
            other_num_turns=num_turns_used,
            other_num_turns_ratio=params.old_num_turns_ratio, 
            p_of_solved_cube=params.p_of_solved_cube,
            p=p
        )

        # GETTING MODEL OUTPUT AND CALCULATIONG LOSS
        output_values, loss = calculate_loss(model, cubes, target_values, optimizer, loss_function, device)
        turns = CubeNet.translate(output_values)
        turns_picking.update(turns)
        last_loss = loss.detach().cpu().item()

        # LOGGING
        if logger:
            log_message = f'Last loss {np.round(last_loss,6):8}'
            logger.filelog(f'{epoch:5} | {log_message}')
            logger.tblog('Training/loss', last_loss, epoch)
            if logger.pbar:
                logger.pbar.set_postfix_str(f'loss {np.round(last_loss,6):8}, num of turns {current_num_turns}')

        # SAVING MODEL
        saving_condition = epoch % params.save_rate == 0 and params.save_path
        save_model_step(model, params, epoch, saving_condition, logger, last_loss)
        
        # VALIDATION
        validation_condition = epoch % params.validation_rate == 0 and epoch != 0
        if logger and validation_condition:
                logf(f'Current turns picking: {turns_picking.most_common()}', a=False)
        validation_step(
            model=model, 
            params=params, 
            solved_cube=solved_cube, 
            min_scramble_turns=params.min_scramble_turns, 
            max_scramble_turns=current_num_turns, 
            epoch=epoch,
            condition=validation_condition,
            device=device,
            logger=logger
        )
        
    # ENDING TRAINING
    save_model_step(model, params, epoch + 1, True, logger, last_loss)

    return model


def self_train_fully_random(
        model : CubeNet,
        params : YParams,
        solved_cube  : ECube,
        
        device : torch.DeviceObjType = 'cpu',
        
        logger : Logger = None,

        optimizer : optim.Optimizer = optim.Adam, 
        loss_function : typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,

    ) -> CubeNet:
    """
    Train CubeNet model using self-supervised learning. 
    The batches are generated automaticly with random amount of scramble turns.

    Parameters
    ----------
    `model` : CubeNet
        Neural network model
    `params` : YParams,
        training parameters
    `solved_cube` : ECube
        Starting cube
    `device` : torch.device, optional
        device to use in training
    `logger` : Logger, optional
        logger to log progress
    `optimizer` : torch.optim.Optimizer
        Optimizer tu use in training
    `loss_function` : torch.LossFunction
        Loss function too calculate loss

    Returns
    -------
    model : CubeNet
        trained model
    """
    # OPTIMIZER INITIALIZATION
    optimizer = optimizer(params=model.parameters(), lr=params.lr)
    
    # HISTORY COLLECTIONS
    turns_picking = Counter({turn : 0 for turn in TURNSi})

    # LOGGER LOG FUNCTION / VALIDATION ITERATIONS 
    if logger:
        logf = lambda message, f=True, a=True, e=True: logger.tqdmlog(message, to_file=f, attention=a, add_iter_num=e)
        logger.valiters = (params.load_epoch//params.validation_rate) * params.validation_epochs

    # MAIN LOOP
    last_loss = np.nan
    if logger:
        epochs_range = logger.pbar
    else:
        epochs_range = range(params.load_epoch, params.num_epochs)
    
    for epoch in epochs_range:

        # BATCH GENERATION
        p = get_turns_p_reversed_distribution(turns_picking, SCRAMBLE_TURNSi)
        cubes, target_values = generate_fully_random_batch(
            solved_cube=solved_cube,
            batch_size=params.batch_size,
            min_scramble_turns=params.min_scramble_turns,
            max_scramble_turns=params.max_scramble_turns,
            p_of_solved_cube=params.p_of_solved_cube,
            p=p,
        )

        # GETTING MODEL OUTPUT AND CALCULATING LOSS
        output_values, loss = calculate_loss(model, cubes, target_values, optimizer, loss_function, device)
        turns         = CubeNet.translate(output_values)
        turns_picking.update(turns)
        last_loss = loss.detach().cpu().item()

        # LOGGING
        if logger:
            log_message = f'Last loss {np.round(last_loss,6):8}'
            logger.filelog(f'{epoch:5} | {log_message}')
            logger.tblog('Training/loss', last_loss, epoch)
            if logger.pbar:
                logger.pbar.set_postfix_str(f'loss {np.round(last_loss,6):8}')

        # SAVING MODEL
        saving_condition = epoch % params.save_rate == 0 and params.save_path
        save_model_step(model, params, epoch, saving_condition, logger, last_loss)
        
        # VALIDATION
        validation_condition = epoch % params.validation_rate == 0 and epoch != 0
        if logger and validation_condition:
                logf(f'Current turns picking: {turns_picking.most_common()}', a=False)
        validation_step(
            model=model, 
            params=params, 
            solved_cube=solved_cube, 
            min_scramble_turns=params.min_scramble_turns, 
            max_scramble_turns=params.max_scramble_turns, 
            epoch=epoch,
            condition=validation_condition,
            device=device,
            logger=logger
        )
    
    save_model_step(model, params, epoch + 1, True, logger, last_loss)

    return model
