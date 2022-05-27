import random
import typing
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from ecube import ECube
from model import CubeNet

from logger  import Logger
from yparams import YParams
from utils import get_logf, get_turns_p_reversed_distribution, save_model_step, validation_step

from collections import Counter, namedtuple
from defaults import SKIP_TURN, SCRAMBLE_TURNSi, TURNSi


Experience = namedtuple(
    'Experience',
    ('current_cube_str', 'next_cube_str', 'next_done', 'reward')
)


class ReplayMemory:
    """
    Memory used to keep experiences of training for
    replay experience methods of reinforcement learning

    Parameters
    ----------
    `capacity` : int
        Maximum size of memory; maximum amount of experiences it can hold.

    Attributes
    ----------
    `capacity` : int
        The maximum size of memory
    `memory` : list
        List of experiences in memory
    """
    def __init__(self, capacity : int):
        self.capacity   = capacity
        self.memory     = []
        self._push_count = 0

    def push(self, experience : Experience):
        """
        Add new experience to memory. If memory is full, then new pushed experience
        will override old one.

        Parameters
        ----------
        `experience` : namedtuple
            Experience to push in memory.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self._push_count % self.capacity] = experience
        self._push_count += 1
    
    def sample(self, batch_size : int) -> list:
        """
        Get random batch of experiences.

        Parameters
        ----------
        `batch_size` : int
            Amount of experience to get from memory.
        """
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size) -> bool:
        """
        Is memory can provide batch of random experiences with this size.

        Parameters
        ----------
        `batch_size` : int
            The size of batch that needs to be sampled.

        Returns
        -------
        `can_provide` : bool
            Is memory can provide batch of random experiences with this size.
        """
        return len(self.memory) >= batch_size


def get_eps(episode : int, eps_start : float, eps_end : float, eps_decay : float) -> float:
    """
    Get epsilone value at current timestep of training.

    Parameters
    ----------
    `timestep` : int
        Current timestep of training
    
    Returns
    -------
    `eps` : int
        The value of epsilone
    """
    return eps_end + (eps_start - eps_end) * np.exp(-1. * episode * eps_decay)


def step(model : CubeNet, cube : ECube, eps : float, device : torch.DeviceObjType) -> typing.Tuple[torch.Tensor, int, ECube, int, bool]:
    """
    Make a step: generate turn, apply this turn to cube in current state, get reward

    Parameters
    ----------
    `model` : CubeNet
        Learning model
    `cube` : ECube
        Current cube state
    `eps` : float
        Epsilone value to use for greedy strategy
    `device` : torch.device
        Device to use
    
    Returns
    -------
    `q_values` : torch.Tensor
        Output of model for current cube
    `turn_id` : int
        Choosed by e-greedy strategy and applied id of turn
    `next_cube` : ECube
        Next state of cube
    `reward` : int
        Reward for choosen turn
    `next_done` : bool
        Is the cube in a solved state
    """
    turn_id  = random.randint(0, len(TURNSi)-1)
    q_values = None
    q_values = model.forward(cube, device=device)
    if eps <= random.random():
        turn_id = model.translate(q_values.detach().cpu(), return_id=True).item()
    next_cube = cube.turn(TURNSi[turn_id])
    next_done = cube.is_solved()
    reward = 1 if next_done else -1
    return q_values, turn_id, next_cube, reward, next_done


def get_loss(
        experiences : typing.Iterable[Experience],
        policy_model : CubeNet,
        target_model : CubeNet,
        gamma : float = 0.999,
        loss_function : typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        device : torch.DeviceObjType = 'cpu'
    ) -> torch.Tensor:
    """
    Calculate loss using experiences batch

    Parameters
    ----------
    `experiences` : Iterable[Experience]
        List of experiences to use in batch
    `policy_model` : CubeNet
        Learnable model to use for current q-values calculation
    `target_model` : CubeNet
        Evaluation model to use for target q-values calculations
    `gamma` : float
        Parameter to use in target q-values calculation
    `loss_function` : torch.LossFunction
        Loss function to use
    `device` : torch.device
        Device to use for loss calculation
    
    Returns
    -------
    `loss` : torch.Tensor
        Loss value
    """
    current_cube_strs, next_cube_strs, next_dones, rewards = Experience(*zip(*experiences))
    next_dones_tensor = torch.tensor(next_dones, dtype=torch.bool).to(device)
    rewards_tensor    = torch.tensor(rewards,    dtype=torch.float32).to(device)
    
    current_full_q_values = policy_model(current_cube_strs, device=device)
    current_q_values = current_full_q_values.gather(dim=1, index=CubeNet.translate(current_full_q_values, return_id=True)).squeeze(1)

    batch_size = current_q_values.shape[0]
    target_q_values = torch.zeros(batch_size).to(device)
    non_final_cubes = [ECube(next_cube_strs[i]) for i,next_done in enumerate(next_dones) if not next_done]
    if non_final_cubes:
        non_final_q_values = target_model(non_final_cubes, device)
        non_final_target_q_values = non_final_q_values.max(dim=1)[0]
        target_q_values[next_dones_tensor == False] = non_final_target_q_values
    target_q_values = target_q_values * gamma + rewards_tensor

    loss = loss_function(current_q_values, target_q_values)
    return loss
    

def rl_train(
        policy_model : CubeNet,
        target_model : CubeNet,
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
    `policy_model` : CubeNet
        learning model
    `target_model` : CubeNet
        model used to calculate loss
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

    logf = get_logf(logger)
        
    # HISTORY COLLECTIONS
    avg_loss = []
    avg_timesteps      = []
    avg_timesteps_coef = []
    avg_q_values       = []
    reward_sums        = []
    turns_id           = []
    turns_picking = Counter({turn : 0 for turn in TURNSi})
    min_scrambles = params.min_scramble_turns
    max_scrambles = params.max_scramble_turns
    if params.min_scramble_turns == 1:
        min_scrambles = 2
    if params.max_scramble_turns == 1:
        max_scrambles = 3
    num_turns_progression = np.linspace(min_scrambles, max_scrambles, params.num_epochs).round().astype(np.int32)

    # MEMORY INITIALIZATION
    memory = ReplayMemory(params.memory_capacity)
    
    # OPTIMIZER INITIALIZATION
    optimizer = optimizer(params=policy_model.parameters(), lr=params.lr)

    # MAIN LOOP
    if logger:
        epochs_range = logger.pbar
    else:
        epochs_range = range(params.load_epoch, params.num_epochs)
    
    for episode in epochs_range:

        # TURNS PROBABILITY DISTRIBUTION
        p = get_turns_p_reversed_distribution(turns_picking, SCRAMBLE_TURNSi)
        
        # CUBE SCRAMBLING
        current_num_turns = num_turns_progression[episode].item() if params.max_scramble_turns != 1 else 1
        rotation = ECube.get_rotation_turns(random.randint(0,3))
        scramble = ECube.get_scramble_turns(current_num_turns, p=p)
        scrambled_or_solved = np.array([[SKIP_TURN], scramble], dtype=object)
        scramble = np.random.choice(scrambled_or_solved, p=[params.p_of_solved_cube, 1.0 - params.p_of_solved_cube])
        reversed_scramble = ECube.reverse_turns(scramble)
        current_cube = solved_cube.turn(rotation + scramble)
        
        # EPISODE HISTORY INITIALIZATION
        episode_loss = []
        episode_timesteps = []
        episode_q_values  = []
        episode_rewards   = []            
        last_loss = np.nan

        # EPISODE LOOP
        current_num_timesteps = 1 if current_num_turns == 1 else params.num_timesteps
        for t in range(current_num_timesteps):

            # GETTING NEXT STATE
            current_done = current_cube.is_solved()
            current_eps = get_eps(episode, params.eps_start, params.eps_end, params.eps_decay)
            q_values, current_turn_id, next_cube, reward, next_done = step(policy_model, current_cube, current_eps, device=device)
            current_turn = TURNSi[current_turn_id] 
            turns_picking.update(current_turn)

            # UPDATING REWARD
            if turns_id:
                reward += np.abs(current_turn_id - np.array(turns_id).mean())/(len(TURNSi)-1)
        
            # SAVING EXPERIENCE
            experience = Experience(
                current_cube_str=current_cube.flat_str(),
                next_cube_str=next_cube.flat_str(),
                next_done=next_done,
                reward=reward
            )
            memory.push(experience)

            # LEARNING STEP
            current_batch_size = 1 if current_num_turns == 1 else params.batch_size
            if memory.can_provide_sample(current_batch_size):
                experiences = memory.sample(current_batch_size)

                # LOSS CALCULATION
                if current_num_turns == 1:
                    target_q_values = torch.zeros(len(TURNSi))
                    target_q_values[TURNSi.index(reversed_scramble[-1])] = 1.0
                    loss = loss_function(q_values.squeeze(0), target_q_values.to(device))
                else:
                    loss = get_loss(experiences, policy_model, target_model, gamma=params.gamma, loss_function=loss_function, device=device)

                # OPTIMIZER STEP
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                last_loss = loss.detach().cpu().item()
                episode_loss.append(loss.detach().cpu().item())

            # LOGGING
            episode_rewards.append(reward)
            turns_id.append(current_turn_id)
            episode_q_values.append(q_values.detach().cpu().numpy()) 

            if logger:
                log_message = f't {t+1:3}/{current_num_timesteps:5}, ' \
                            + f'last loss {np.round(last_loss,6):8}, ' \
                            + f'avg loss {np.nan if not episode_loss else np.array(episode_loss).mean().round(6):8}, ' \
                            + f'rewards sum {np.array(episode_rewards).sum().round(4):6} '
                logger.filelog(log_message)
                logger.tblog('All/last_loss', last_loss, logger.iters)
                logger.tblog('All/avg_loss', np.nan if not episode_loss else np.array(episode_loss).mean(), logger.iters)
                logger.tblog('All/rewards_sum', np.array(episode_rewards).sum(), logger.iters)
                logger.pbar.set_description(log_message)
                logger.iters += 1

            # ENDING EPISODE
            if next_done:
                break
            
            # CURRENT STATE
            current_cube = next_cube.copy()
    
        # HISTORY LOGGING
        avg_q_values.append(np.array(episode_q_values).mean(axis=0))
        reward_sums.append(np.array(episode_rewards).sum())
        episode_timesteps.append(t+1)
        avg_timesteps.append(np.array(episode_timesteps).mean())
        avg_timesteps_coef.append(1.0 - avg_timesteps[-1]/current_num_timesteps)
        avg_loss.append(np.array(episode_loss if episode_loss else [np.nan]).mean())

        if logger:
            logger.pbar.set_postfix_str(f'last avg loss {avg_loss[-1].round(6):8}, '
                                + f'last avg solve coef {avg_timesteps_coef[-1].round(4):6}, '
                                + f'last reward sums {reward_sums[-1].round(4):6}, '
                                + f'current eps {current_eps.round(4):6}')
            logger.tblog('Episode/avg_loss', avg_loss[-1], episode)
            logger.tblog('Episode/avg_solve_coef', avg_timesteps_coef[-1], episode)
            logger.tblog('Episode/reward_sums', reward_sums[-1], episode)
            logger.tblog('Episode/eps', current_eps, episode)

        # COPYING POLICY MODEL TO TARGET
        if episode % params.target_update_rate == 0:
            if logger:
                logf(f'{episode} Copying policy model to target model... Last loss {last_loss}', a=False)
            target_model = policy_model.copy()
        
        # SAVING MODEL
        saving_condition = episode % params.save_rate == 0 and params.save_path
        if saving_condition:
            target_model = policy_model.copy()
        save_model_step(target_model, params, episode, saving_condition, logger, last_loss)
        
        # VALIDATION
        validation_condition = episode % params.validation_rate == 0 and episode != 0
        if logger and validation_condition:
                logf(f'Current turns picking: {turns_picking.most_common()}', a=False)
        validation_step(
            model=policy_model, 
            params=params, 
            solved_cube=solved_cube, 
            min_scramble_turns=params.min_scramble_turns, 
            max_scramble_turns=params.max_scramble_turns, 
            epoch=episode,
            condition=validation_condition,
            device=device,
            logger=logger
        )
        
        # UPDATING SCRAMBLE TURNS AMOUNT
        if params.max_scramble_turns > 1 and episode + 1 != params.num_epochs and num_turns_progression[episode + 1] != current_num_turns:
            if logger:
                logf(f'Changing number of turns to {current_num_turns + 1} and copying policy model... Last loss {last_loss}')
            target_model = policy_model.copy()
            
    # ENDING TRAINING
    target_model = policy_model.copy()
    save_model_step(target_model, params, episode + 1, True, logger, last_loss)

    return target_model
