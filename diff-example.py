import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


def torch_to_cv2(image):
    image = image.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.uint8)
    return image


def rgb2yuv(image_rgb, image_yuv_out):
    """ Transform the image from rgb to yuv """
    image_yuv_out[:, 0, :, :] = 0.299 * image_rgb[:, 0, :, :].clone() + 0.587 * image_rgb[:, 1, :,
                                                                                          :].clone() + 0.114 * image_rgb[:, 2, :,
                                                                                                                         :].clone()
    image_yuv_out[:, 1, :, :] = -0.14713 * image_rgb[:, 0, :, :].clone() + -0.28886 * image_rgb[:, 1, :,
                                                                                                :].clone() + 0.436 * image_rgb[:,
                                                                                                                               2, :,
                                                                                                                               :].clone()
    image_yuv_out[:, 2, :, :] = 0.615 * image_rgb[:, 0, :, :].clone() + -0.51499 * image_rgb[:, 1, :,
                                                                                             :].clone() + -0.10001 * image_rgb[:,
                                                                                                                               2, :,
                                                                                                                               :].clone()


def yuv2rgb(image_yuv, image_rgb_out):
    """ Transform the image from yuv to rgb """
    image_rgb_out[:, 0, :, :] = image_yuv[:, 0, :,
                                          :].clone() + 1.13983 * image_yuv[:, 2, :, :].clone()
    image_rgb_out[:, 1, :, :] = image_yuv[:, 0, :, :].clone() + -0.39465 * image_yuv[:, 1, :,
                                                                                     :].clone() + -0.58060 * image_yuv[:, 2, :,
                                                                                                                       :].clone()
    image_rgb_out[:, 2, :, :] = image_yuv[:, 0, :,
                                          :].clone() + 2.03211 * image_yuv[:, 1, :, :].clone()


original_images = [cv2.imread(os.path.join("./figures-output/combined-vs-rivagan", filename))
                   for filename in sorted(os.listdir("./figures-output/combined-vs-rivagan"))
                   if not filename.startswith(".") and filename.startswith("original")]

combined_images = [cv2.imread(os.path.join("./figures-output/combined-vs-rivagan", filename))
                   for filename in sorted(os.listdir("./figures-output/combined-vs-rivagan"))
                   if not filename.startswith(".") and filename.startswith("combined")]

rivagan_images = [cv2.imread(os.path.join("./figures-output/combined-vs-rivagan", filename))
                  for filename in sorted(os.listdir("./figures-output/combined-vs-rivagan"))
                  if not filename.startswith(".") and filename.startswith("rivagan")]

i = 0
for img_original, img_combine, img_rivagan in zip(original_images, combined_images, rivagan_images):
    # amp = 15
    # diff = diff * amp
    # diff_avg = np.mean(diff, axis=2)
    # diff_avg_pil = Image.fromarray(diff_avg, mode="L")
    # diff_avg_pil.save(f"figures-output/combined-vs-rivagan/diff-{i + 1}.png")

    # img_yuv = cv2.cvtColor(img_combine, cv2.COLOR_BGR2YUV)
    # wm_img_yuv = cv2.cvtColor(img_rivagan, cv2.COLOR_BGR2YUV)
    # diff = np.abs(img_yuv - wm_img_yuv)
    # _, h, w = diff.shape
    # # diff[1] = np.zeros((h, w))
    # # diff[2] = np.zeros((h, w))
    # diff[:, :, 1] = np.zeros_like(diff[:, :, 1])  # Set U channel to zero
    # diff[:, :, 2] = np.zeros_like(diff[:, :, 2])  # Set V channel to zero
    # print(diff)
    # diff = (diff * 15).clip(0, 255)
    # diff_bgr = cv2.cvtColor(diff, cv2.COLOR_YUV2BGR)

    # cv2.imwrite(f"figures-output/combined-vs-rivagan/diff-{i + 1}.png", diff_bgr)
    # img_yuv = cv2.cvtColor(img_combine, cv2.COLOR_BGR2YUV)
    # wm_img_yuv = cv2.cvtColor(img_rivagan, cv2.COLOR_BGR2YUV)
    #
    # diff_y = np.abs(img_yuv[:, :, 0] - wm_img_yuv[:, :, 0])  # Calculate Y channel difference
    # diff_y = np.clip(diff_y, 0, 255)
    #
    # diff_rgb = np.zeros_like(img_combine)  # Initialize a zero-filled RGB image
    # diff_rgb[:, :, 0] = diff_y  # Assign Y channel difference to the R channel

    # # convert to grayscale
    # diff_rgb = cv2.cvtColor(diff_rgb, cv2.COLOR_YUV2BGR)
    # diff_rgb = cv2.cvtColor(diff_rgb, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(f"figures-output/combined-vs-rivagan/diff-{i + 1}.png", diff_rgb)

    # amplification = 15
    # img_combine_tensor = TF.to_tensor(img_combine)
    # img_rivagan_tensor = TF.to_tensor(img_rivagan)
    # diff = (img_combine_tensor - img_rivagan_tensor).abs()
    # diff = diff * amplification
    # diff = diff.clamp(0, 1)
    # diff_avg = torch.mean(diff, dim=0)
    # diff_avg_pil = TF.to_pil_image(diff_avg, "L")
    # diff_avg_pil.save(f"./figures-output/combined-vs-rivagan/diff-{i + 1}.png")

    img_original_tensor = TF.to_tensor(img_original).unsqueeze_(0)
    img_combine_tensor = TF.to_tensor(img_combine).unsqueeze_(0)
    img_rivagan_tensor = TF.to_tensor(img_rivagan).unsqueeze_(0)

    image_size = 512
    img_original_tensor = TF.resize(
        img_original_tensor, (image_size, image_size))
    img_combine_tensor = TF.resize(
        img_combine_tensor, (image_size, image_size))
    img_rivagan_tensor = TF.resize(
        img_rivagan_tensor, (image_size, image_size))

    img_original_yuv = torch.empty_like(img_original_tensor)
    img_combine_yuv = torch.empty_like(img_combine_tensor)
    img_rivagan_yuv = torch.empty_like(img_rivagan_tensor)
    rgb2yuv(img_original_tensor, img_original_yuv)
    rgb2yuv(img_combine_tensor, img_combine_yuv)
    rgb2yuv(img_rivagan_tensor, img_rivagan_yuv)

    diff_combined = torch.abs(img_original_yuv - img_combine_yuv)
    _, _, h, w = diff_combined.shape
    diff_combined[:, 1] = torch.zeros((h, w))
    diff_combined[:, 2] = torch.zeros((h, w))
    amplification = 15
    diff_combined = (diff_combined * 15).clip(0, 255)
    yuv2rgb(diff_combined, diff_combined)

    diff_rivagan = torch.abs(img_original_yuv - img_rivagan_yuv)
    _, _, h, w = diff_rivagan.shape
    diff_rivagan[:, 1] = torch.zeros((h, w))
    diff_rivagan[:, 2] = torch.zeros((h, w))
    diff_rivagan = (diff_rivagan * amplification).clip(0, 255)
    yuv2rgb(diff_rivagan, diff_rivagan)

    diff_combined_pil = TF.to_pil_image(diff_combined.squeeze_(0))
    diff_rivagan_pil = TF.to_pil_image(diff_rivagan.squeeze_(0))
    diff_combined_pil = TF.to_grayscale(diff_combined_pil, num_output_channels=1)
    diff_rivagan_pil = TF.to_grayscale(diff_rivagan_pil, num_output_channels=1)
    diff_combined_pil.save(f"./figures-output/combined-vs-rivagan/diff-combined-{i + 1}.png")
    diff_rivagan_pil.save(f"./figures-output/combined-vs-rivagan/diff-rivagan-{i + 1}.png")

    i += 1
