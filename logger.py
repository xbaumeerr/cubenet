import os
import tqdm
import typing

from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    Class for logging

    Parameters
    ----------
    `log_dir` : str
        directory for logging
    `log_filename` : str
        name of file to save logging
    `pbar` : tqdm, optinal
        save tqdm progress bar as object attribute
    `clear` : bool
        whether clear logging file on start
    `use_tensorboard` : bool
        whether initialize SummaryWriter for tensorboard logging
    `purge_step` : int
        purge_step parameter for SummaryWriter
    `filename_suffix` : str
        filename_suffix parameter for SummaryWriter
    """
    def __init__(self, log_dir : str = '', log_filename : str = '', clear : bool = False, pbar : tqdm = None, use_tensorboard : bool = False, purge_step : int = None, filename_suffix : str = ''):
        self.pbar     = pbar
        self.path     = log_dir
        self.filename = log_filename
        self.filepath = os.path.join(log_dir, log_filename)
        self.use_tensorboard = use_tensorboard
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if self.filepath and clear:
            open(self.filepath, 'w').close()
        if use_tensorboard:
            self.sw = SummaryWriter(log_dir, purge_step=purge_step, filename_suffix=self.filename)
    
    def tqdmlog(self, message : str, pbar : tqdm = None, to_file : bool = False, attention : bool = False, add_iter_num : bool = False):
        """
        Log message using tqdm progress bar

        Parameters
        ----------
        `message` : str
            Message to log
        `pbar` : tqdm, optional
            tqdm progress bar to write
        `to_file` : bool, optional
            where save log message in file
        `attention` : bool, optional
            whether add '=' sign as attention to message
        `add_iter_num` : bool, optional
            whether add prefix as iteration number to message
        """
        pbar = pbar if self.pbar is None else self.pbar
        if to_file:
            self.filelog(message)
        if pbar is not None:
            if add_iter_num:
                message = f'{pbar.n:5} | {message}'
            if attention:
                pbar.write('='*len(message))
            pbar.write(message)
            if attention:
                pbar.write('='*len(message))
    
    def filelog(self, message : str, filepath : str = None):
        """
        Write message to file

        Parameters
        ----------
        `message` : str
            message to log
        `filepath`: str
            Path to file to add save message
        """
        filepath = filepath if filepath is not None else self.filepath
        if filepath:
            with open(filepath, 'a') as f:
                f.write(message + '\n')
    
    def tblog(self, tag : str, value : typing.Any, step : int, sw : SummaryWriter = None):
        """
        Log value to tensorboard using SummaryWriter.

        Parameters
        ----------
        `tag` : str
            see `add_scalar` summary writer method
        `value` : Any
            see `add_scalar` summary writer method
        `step` : int
            see `add_scalar` summary writer method
        `sw` : SummaryWriter, optional
            SummaryWriter object for logging
        """
        if sw:
            sw_ = sw
        elif self.use_tensorboard:
            sw_ = self.sw
        if sw_:
            sw_.add_scalar(tag, value, step)