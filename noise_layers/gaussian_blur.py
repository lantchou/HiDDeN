import torch.nn as nn
import torchvision.transforms.functional as TF
import random

SIGMAS = [1, 3, 5, 7, 9]


class GaussianBlur(nn.Module):
    """
    Apply gaussian blur with random sigma to image.
    """

    def __init__(self, sigmas=None):
        super(GaussianBlur, self).__init__()
        if sigmas is None:
            sigmas = SIGMAS
        self.sigmas = sigmas

    def forward(self, noised_and_cover):
        sigma = random.choice(self.sigmas)

        # apply gaussian blur with kernel size `sigma` and sigma `sigma`.
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = TF.gaussian_blur(noised_image, [sigma, sigma], sigma)

        return noised_and_cover
