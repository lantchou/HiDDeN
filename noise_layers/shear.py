import random
import torch.nn as nn
import torchvision.transforms.functional as TF


class Shear(nn.Module):
    """
    Apply a shear to the image with a random angle within a given range.
    """

    min_angle: int
    max_angle: int

    def __init__(self, min_angle=2, max_angle=30):
        super(Shear, self).__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle

    def forward(self, noised_and_cover):
        angle = random.uniform(self.min_angle, self.max_angle)
        if random.random() < 0.5:
            angle = -angle

        noised_and_cover[0] = TF.affine(noised_and_cover[0], angle=0, translate=[0, 0], scale=1, shear=angle, interpolation=TF.InterpolationMode.BILINEAR)
        return noised_and_cover
