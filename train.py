import os
import typing
import argparse

import torch

import numpy as np

from ecube import ECube
from model import CubeNet
from yparams import YParams
from cubetyping import TrainingType
from utils import training_logger_preprocessing


from rl_train import rl_train
from self_train import self_train, self_train_fully_random

TRAINING_TYPE_SELF = "self"
TRAINING_TYPE_RL   = "rl"

def start_training(
        params_path : str,
        training_type : TrainingType
    ) -> typing.Tuple[CubeNet, np.ndarray]:
    """
    Train CubeNet model using self-supervised learning. The batches are generated automaticly.

    Parameters
    ----------
    `params_path` : str
        path to file with training parameters
    `training_type` : TrainingType
        type of training to use
    """
    np.seterr(invalid='ignore')

    params = YParams(params_path)

    logger = training_logger_preprocessing(params)

    if training_type == TRAINING_TYPE_SELF:
        logger.valiters = (params.load_epoch//params.validation_rate) * params.validation_epochs
    elif training_type == TRAINING_TYPE_RL:
        logger.iters    = params.load_epoch*params.num_timesteps
        logger.valiters = 0

    device = torch.device(params.device)

    solved_cube = ECube.get_default_cube()

    # MODEL INITITALIZATION
    models = {}
    if training_type == TRAINING_TYPE_SELF:
        model = CubeNet()
        model  = model.to(device)
        models['model'] = model
    elif training_type == TRAINING_TYPE_RL:
        policy_model = CubeNet().to(device)
        target_model = CubeNet().to(device)
        models['policy_model'] = policy_model
        models['target_model'] = target_model
    
    # SHOW PARAMETERS
    logger.tqdmlog('='*100, add_iter_num=False)
    params.display(lambda message: logger.tqdmlog(message, attention=False, add_iter_num=False))
    logger.tqdmlog('='*100, add_iter_num=False)
    

    # SAVE FOLDER PREPROCESSING / LOADING MODEL
    if params.save_path is not None:
        if not os.path.exists(params.save_path):
            os.makedirs(params.save_path)
    if params.load_path:
        logger.tqdmlog(f"Loading model from file path {params.load_path}...", attention=True, to_file=True)
        if training_type == TRAINING_TYPE_SELF:
            models['model'].load(params.load_path)
        elif training_type == TRAINING_TYPE_RL:
            models['policy_model'].load(params.load_path)
    if training_type == TRAINING_TYPE_RL:
        models['target_model'] = models['policy_model'].copy()
        models['policy_model'].train()
        models['target_model'].eval() 

    # TRAIN
    if training_type == TRAINING_TYPE_SELF:
        if params.fully_random:
            model = self_train_fully_random(models['model'], params, solved_cube, device, logger)
        else:
            model = self_train(models['model'], params, solved_cube, device, logger)
    elif training_type == TRAINING_TYPE_RL:
        model = rl_train(policy_model, target_model, params, solved_cube, device, logger)
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params_path', type=str, required=True,
        help='path to file with training parameters')
    parser.add_argument('--training_type', type=str, required=True, choices=[TRAINING_TYPE_RL, TRAINING_TYPE_SELF], 
        help=f'type of training to use: {TRAINING_TYPE_SELF} - self supervised learning, {TRAINING_TYPE_RL} - reinforcement learning')

    args = parser.parse_args()

    start_training(args.params_path, args.training_type)
