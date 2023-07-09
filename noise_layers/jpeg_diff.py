import random
import torch.nn as nn
from DiffJPEG import DiffJPEG
from noise_layers.jpeg_compression import save_images

class JpegDiff(nn.Module):
    """
    Noise layer which applies a differentiable JPEG approximation to the noised image within a 0-100 QF range.
    """
    def __init__(self, device):
        super(JpegDiff, self).__init__()
        self.device = device

    def forward(self, noised_and_cover):
        qf = random.randint(1, 100)
        _, _, height, width = noised_and_cover[0].shape
        jpeg = DiffJPEG(height=height, width=width, differentiable=True, quality=qf, device=self.device)
        images = (noised_and_cover[0] + 1) / 2  # to [0, 1] range
        jpeg_images = jpeg(images)
        noised_and_cover[0] = jpeg_images * 2 - 1  # back to [-1, 1] range
        return noised_and_cover
