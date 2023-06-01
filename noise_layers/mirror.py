import torch.nn as nn
import torchvision.transforms.functional as TF


class Mirror(nn.Module):
    """
    Mirror the image.
    """

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = TF.hflip(noised_image)

        return noised_and_cover
