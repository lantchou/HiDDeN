import torch.nn as nn
import torchvision.transforms.functional as TF

class GaussianBlur(nn.Module):
    """
    Apply gaussian blur with given sigma to image.
    """

    def __init__(self, sigma):
        super(GaussianBlur, self).__init__()
        self.sigma = sigma

    def forward(self, noised_and_cover):
        # apply gaussian blur with kernel size `sigma` and sigma `sigma`.
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = TF.gaussian_blur(noised_image, [self.sigma, self.sigma], self.sigma)

        return noised_and_cover

    def __repr__(self):
        return f"GaussianBlur(sigma={self.sigma})"
