from util import load_model
import torch
import torchvision
from PIL import Image
import random
import os
import numpy as np
import torchvision.transforms.functional as TF
import math
import cv2


def torch_to_cv2(image):
    image = image.cpu().detach().numpy()
    image = (image + 1) / 2  # Revert the [-1, 1] normalization
    image = np.clip(image * 255, 0, 255).astype(np.uint8)  # Map to [0, 255]
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def small_crop_figure():

    device = torch.device("cpu")
    options_file = "./runs/hidden-identity-2023.03.04--14-28-11/options-and-config.pickle"
    checkpoint_file = "./runs/hidden-identity-2023.03.04--14-28-11/checkpoints/hidden-identity--epoch-400.pyt"
    hidden_net, _, _ = load_model(options_file,
                                  checkpoint_file,
                                  device)

    folder = "../data/test/"
    path = os.path.join(folder, random.choice(os.listdir(folder)))
    image_size = 256
    image = Image.open(path)
    image = torchvision.transforms.ToTensor()(image).to(device).unsqueeze_(0)
    image = image * 2 - 1
    image = TF.resize(image, [image_size, image_size])

    crop_ratio = 0.1
    crop_size = math.floor(image_size * crop_ratio)
    image_crop = TF.center_crop(image, [crop_size, crop_size])

    message_length = 30
    message = np.random.choice([0, 1], (1, message_length))
    message = torch.Tensor(message).to(device)

    image_enc = hidden_net.eval_encode_on_batch(image, message)

    image_enc_crop = TF.center_crop(image_enc, [crop_size, crop_size])

    message_dec = hidden_net.eval_decode_on_batch(image_enc_crop)

    message_detached = message.detach().cpu().numpy()
    msg_dec_rounded = message_dec.detach().cpu().numpy().round().clip(0, 1)
    msg_error_count = np.sum(
        np.abs(msg_dec_rounded - message_detached))
    msg_error_rate = msg_error_count / message_length

    print("message error rate: ", msg_error_rate)

    image_cv2 = torch_to_cv2(image.squeeze_(0))
    cv2.imwrite(
        "./figures-output/small-crop-example/small_crop_original.png", image_cv2)

    image_crop_cv2 = torch_to_cv2(image_crop.squeeze_(0))
    cv2.imwrite(
        "./figures-output/small-crop-example/small_crop_original_cropped.png", image_crop_cv2)

    image_enc_cv2 = torch_to_cv2(image_enc.squeeze_(0))
    cv2.imwrite(
        "./figures-output/small-crop-example/small_crop_encoded.png", image_enc_cv2)

    image_enc_cv2_crop = torch_to_cv2(image_enc_crop.squeeze_(0))
    cv2.imwrite(
        "./figures-output/small-crop-example/small_crop_encoded_cropped.png", image_enc_cv2_crop, )


if __name__ == "__main__":
    small_crop_figure()
