import torch.nn as nn
import torchvision.transforms.functional as TF
import random

class Rotate(nn.Module):
    """
    Rotate image with randomized angle.
    """

    def __init__(self, min_angle, max_angle):
        super(Rotate, self).__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle

    def forward(self, noised_and_cover):
        angle = random.uniform(self.min_angle, self.max_angle)
        if random.random() < 0.5:
            angle = -angle

        noised_image = noised_and_cover[0]
        noised_and_cover[0] = TF.rotate(noised_image, angle)

        return noised_and_cover
