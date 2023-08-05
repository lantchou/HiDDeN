from typing import List
import numpy as np
import torch.nn as nn
import torch
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.rotate import Rotate
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.mirror import Mirror
from noise_layers.translate import Translate
from noise_layers.shear import Shear


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: List[nn.Module], device):
        super(Noiser, self).__init__()
        self.noise_layers: List[nn.Module] = [Identity()]
        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))
                elif layer == 'Mirror':
                    self.noise_layers.append(Mirror())
                else:
                    raise ValueError(f'Unknown layer string in Noiser.__init__().')
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        return random_noise_layer(encoded_and_cover)

