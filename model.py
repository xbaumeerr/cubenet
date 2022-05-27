import typing

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ecube import ECube
from rubik.cube import Cube

from collections.abc import Iterable
from cubetyping import CubeNetInput, TurnStr

from defaults import TURNSi


class CubeNet(nn.Module):
    """
    Neural network to solve rubik's cube 3x3x3
    eLU activation function is used in this network. You can change it using set_activation_function method.
    """
    def __init__(self):
        super().__init__()
            
        self._conv3d = nn.Conv3d(9, 9, kernel_size=3, padding=1)
        self._conv2d = nn.Conv2d(1, 3, kernel_size=(1,3))
        self._pool   = nn.MaxPool2d((1,3))
        self._line1  = nn.Linear(3*3*3*3, 26)
        self._actf   = F.elu
        self._line2  = nn.Linear(26, len(TURNSi))
    
    def _prepare_cube(self, cube_s : CubeNetInput) -> ECube | typing.Iterable[ECube]:
        """
        Prepare cube or list of cubes for processing.

        Parameters
        ----------
        `cube_s` : CubeStr | CubeLike | Iterable[CubeStr] | Iterable[CubeLike]
            Representation of a cube or list of cubes.

        Returns
        -------
        `cube` : ECube | Iterable[ECube]
            Prepared cube or list of cubes
        """
        if isinstance(cube_s, (ECube, Cube)):
            return ECube(cube_s.flat_str())
        if isinstance(cube_s, str):
            return ECube(cube_s)
        if isinstance(cube_s, Iterable):
            cube_s = list(cube_s)
            for i,c in enumerate(cube_s):
                cube_s[i] = self._prepare_cube(c)
        return cube_s


    def preprocessing(self, cube_s : CubeNetInput) -> torch.Tensor:
        """
        Generate input for network with cube or batch of cubes in its current state

        Parameters
        ----------
        `cube_s` : CubeNetInput
            Representation of a cube or list of cubes

        Returns
        -------
        `inpt` : torch.Tensor
            Input of size (N,9,1,3,3,3) for the neural network, where N is batch size (1 if cube where single)
        """
        cube_s = self._prepare_cube(cube_s)
        if isinstance(cube_s, ECube):
            cube_s = [cube_s,]
        x = [cube.cube4D() for cube in cube_s]
        x = np.array(x)
        x = torch.from_numpy(x)
        x_slices = x                      # slices of cube from left to right 3x3 matricies 
        y_slices = x.permute(0, 2,3,1,4)  # slices of cube from bottom to top
        z_slices = x.permute(0, 3,1,2,4)  # slices of cube from back to front
        inpt     = torch.concat((x_slices,y_slices,z_slices), dim=1) # make N batches with 9 channels each
        return inpt

    def forward(self, cube_s : CubeNetInput, device='cpu') -> torch.Tensor:
        """
        Feed forward network

        Parameters
        ----------
        `cube_s` : CubeNetInput
            Cube or list of cubes to solve
        `device` : torch.device, optional
            Device to use in feed forward
        
        Returns
        -------
        `y` : torch.Tensor
            Output of size (N,20) of the network with q-value for each possible turn and batch size N
        """
        cube_s = self._prepare_cube(cube_s)
        x = self.preprocessing(cube_s)  # -> (N,9,3,3,3) 
        x = x.to(device)
        x = self._conv3d(x)             # -> (N,9,3,3,3)    // for each slice from left to right, top to bottom and front to back
        x = x.view(-1,9*3*3,3)          # -> (N,81,3)       // get each transformed 3-vector of each slice
        x = x.unsqueeze(1)              # -> (N,1,81,3)     // add channel layer
        x = self._conv2d(x)             # -> (N,3,81,1)     // get convolution of each 3-vector with 3 filters
        x = x.permute(0,2,1,3)          # -> (N,81,3,1)
        x = x.squeeze(3)                # -> (N,81,3)       // set shape to (BATCHES, EACH 3-VECTORS OF EACH SLICE, 3 CONVOLUTION VALUES)
        x = self._pool(x)               # -> (N,81,1)       // max pooling of filters
        x = x.flatten(1)                # -> (N,81)
        x = self._line1(x)              # -> (N,26)
        x = self._actf(x)               # -> (N,26)         // activation function
        x = self._line2(x)              # -> (N,17)         // get q-values for each turn
        return x

    @staticmethod
    def translate(y : torch.Tensor, detach : bool = True, return_id : bool = False) -> TurnStr | int:
        """
        Get turn letter from network output

        Parameters
        ----------
        `y` : torch.Tensor
            Output of CubeNet network
        `detach` : bool, optional
            Detach output tensor from gradient calculation
        `return_id` : bool, optional
            If true, index of the move will be returned instead of move letter
        
        Returns
        -------
        `move` : TurnStr | int
            Predicted move to apply to cube
        """
        if detach:
            y = y.detach()
        move_idx = y.argmax(dim=1).unsqueeze(1)
        if return_id:
            return move_idx
        turns = [TURNSi[move_id.item()] for move_id in move_idx]
        return turns

    def predict(self, cube_s : CubeNetInput, detach : bool = True, with_grad : bool = False, return_id : bool = False, device : torch.DeviceObjType = 'cpu') -> TurnStr | int:
        """
        Predict the next move from current cube state or list of cube states to solution
        using neural network

        Parameters
        ----------
        `cube_s` : CubeNetInput
            Cube or list of cubes as input
        `detach` : bool, optional
            Detach output tensor from gradient calculation
        `with_grad` : bool, optional
            Use gradiante calculation during prediction
        `return_id` : bool, optional
            If true, index of the move will be returned instead of move letter
        `device` : torch.device
            Device to use

        Returns
        -------
        `move` : TurnStr | int
            Predicted move
        """
        cube_s = self._prepare_cube(cube_s)
        if not with_grad:
            with torch.no_grad():
                return CubeNet.translate(self.forward(cube_s, device), detach=detach, return_id=return_id)
        else:
            return CubeNet.translate(self.forward(cube_s, device), detach=detach, return_id=return_id)
    

    def set_activation_function(self, activation_function : typing.Callable[[torch.Tensor], torch.Tensor]):
        """
        Set different activation function to use

        Parameters
        ----------
        `activation_function` : Callable(torch.Tensor) -> torch.Tensor
        """
        self._actf = activation_function

    def save(self, path : str):
        """
        Save neural network and cube parameters to path

        Parameters
        ----------
        `path` : str
            Path to save neural network
        """
        torch.save(self.state_dict(), path)            
    
    def load(self, path : str):
        """
        Load model

        Parameters
        ----------
        `path` : str
            Path to file with saved model parameters
        """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def copy(self, other : nn.Module = None) -> nn.Module:
        """
        Copy CubeNet model

        Parameters
        ----------
        `other` : CubeNet, optional
            if parameter passes, then it will get the copy
        
        Returns
        -------
        `model_copy` : CubeNet
            returns copy of model
        """
        model = CubeNet() if other is None else other
        model.load_state_dict(self.state_dict())
        if isinstance(model, CubeNet):
            model.set_activation_function(self._actf)
        return model
        