import random
import torch.nn as nn
from DiffJPEG import DiffJPEG


class JpegDiff(nn.Module):
    """
    Noise layer which applies a differentiable JPEG approximation to the noised image within a 0-100 QF range.
    """

    def __init__(self, device, height=128, width=128, qf: int | tuple[int, int] = (70, 100)):
        super(JpegDiff, self).__init__()
        self.device = device
        if isinstance(qf, int):
            self.jpeg = DiffJPEG(height=height, width=width, differentiable=True, quality=qf, device=self.device)
            self.qf_range = None
        else:
            self.qf_range = qf

    def forward(self, noised_and_cover):
        # get jpeg noise layer instance
        _, _, height, width = noised_and_cover[0].shape
        if self.qf_range is not None:
            qf = random.randint(self.qf_range[0], self.qf_range[1])
            jpeg = DiffJPEG(height=height, width=width, differentiable=True, quality=qf, device=self.device)
        else:
            jpeg = self.jpeg

        # apply jpeg noise layer
        images = (noised_and_cover[0] + 1) / 2  # to [0, 1] range
        jpeg_images = jpeg(images)
        noised_and_cover[0] = jpeg_images * 2 - 1  # back to [-1, 1] range

        return noised_and_cover
