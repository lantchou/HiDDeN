import random
import torch.nn as nn
import torchvision.transforms.functional as TF


class Translate(nn.Module):
    """
    Translate the image in a random direction for both axes within a given range.
    """

    min_translate_ratio: float
    max_translate_ratio: float

    def __init__(self, min_translate_ratio=0.05, max_translate_ratio=0.5):
        super(Translate, self).__init__()
        self.min_translate_ratio = min_translate_ratio
        self.max_translate_ratio = max_translate_ratio

    def forward(self, noised_and_cover):

        translate_ratio_x = random.uniform(self.min_translate_ratio, self.max_translate_ratio)
        translate_ratio_y = random.uniform(self.min_translate_ratio, self.max_translate_ratio)
        dy = int(noised_and_cover[0].shape[2] * translate_ratio_y)
        dx = int(noised_and_cover[0].shape[3] * translate_ratio_x)

        noised_image = noised_and_cover[0]
        noised_and_cover[0] = TF.affine(noised_image, angle=0, translate=[dx, dy], scale=1, shear=0)

        return noised_and_cover
