import torch.nn as nn
import torchvision.transforms.functional as TF
import random

MAX_ROTATION_ANGLE = 20

class Rotate(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """

    def __init__(self, max_rotation_angle=MAX_ROTATION_ANGLE):
        super(Rotate, self).__init__()
        self.max_rotation_angle = max_rotation_angle

    def forward(self, noised_and_cover):
        angle = random.uniform(0, self.max_rotation_angle)
        if random.random() < 0.5:
            angle = -angle

        noised_image = noised_and_cover[0]
        noised_and_cover[0] = TF.rotate(noised_image, angle)

        return noised_and_cover
