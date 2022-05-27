import yaml


class YParams:
    """
    Class to save parameters from yaml config file
    Parameters of the config file will be saved as object attributes

    Parameters
    ----------
    `filepath` : str
        Path to config file with parameters
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self._load_params()
    
    def _load_params(self):
        """
        Load parameters from config file
        """
        self.kw = {}
        with open(self.filepath) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        for param_name, param_value in params.items():
            self.kw[param_name] = param_value
            setattr(self, param_name, param_value)
    
    def display(self, log_function=print):
        for param_name, param_value in self.kw.items():
            log_function(f'{param_name}: {param_value}')