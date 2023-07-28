import random
from typing import List
import torch.nn as nn
import torchvision.transforms.functional as TF


class GaussianBlur(nn.Module):
    """
    Apply gaussian blur to image with kernel aize randomly selected from given kernel_sizes
    """

    def __init__(self, kernel_sizes: List[int]):
        super(GaussianBlur, self).__init__()
        # assert that all kernel sizes are odd with message
        assert all([kernel_size % 2 == 1 for kernel_size in kernel_sizes]), \
            "All kernel sizes must be odd"
        self.kernel_sizes = kernel_sizes

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        kernel_size = random.choice(self.kernel_sizes)
        noised_and_cover[0] = TF.gaussian_blur(
            noised_image, [kernel_size, kernel_size])

        return noised_and_cover

    def __repr__(self):
        return f"GaussianBlur(kernel_sizes={self.kernel_sizes})"
