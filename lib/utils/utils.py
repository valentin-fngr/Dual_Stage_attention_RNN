import os 
from easydict import EasyDict as edict
import yaml
import json 

import torch    


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader, node: yaml.Node):
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())


def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config



def mape(y_true, y_pred): 
    """
    MAPE score
    """
    epsilon = 1e-10  # To prevent division by zero
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
