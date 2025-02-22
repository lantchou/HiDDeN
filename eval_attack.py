import argparse
import torch
import os
import time
from PIL import Image
from torch.functional import Tensor
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from typing import List, Tuple
import numpy as np
import csv
import math
import io
import random
from typing import Union, Sequence

import matplotlib.pyplot as plt

from model.hidden import Hidden
from util import load_model
from noise_layers.crop import get_random_rectangle_inside
from test_model import random_crop

from noise_layers.jpeg_compression import rgb2yuv
from noise_layers.jpeg_compression import yuv2rgb

RESULTS_FILENAME = "results.txt"
GRAPH_FILENAME = "graph.png"


def main():
    if torch.has_mps:
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(
        description='Evaluate performance of model against attack')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True,
                        type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12,
                        type=int, help='The batch size.')
    parser.add_argument('--input-folder', '-i', required=True, type=str,
                        help='Folder with input images')
    parser.add_argument("--attack", "-a", required=True, help="Attack type")
    parser.add_argument("--name", "-n", required=False, help="Experiment name")
    parser.add_argument('--save-images', '-s', action=argparse.BooleanOptionalAction,
                        default=False, help='Save attacked encoded images')
    parser.add_argument('--image-size', default=None, type=int)

    args = parser.parse_args()

    images, filenames = load_images(args.input_folder, device, args.image_size)
    _, _, height, width = images.shape

    hidden_net, hidden_config, train_options = load_model(
        args.options_file, args.checkpoint_file, device)

    if args.name:
        train_options.experiment_name = args.name

    # get last part of folder path
    folder_name = os.path.basename(os.path.normpath(args.input_folder))

    results_dir = os.path.join(
        "eval-attack-results",
        args.attack,
        train_options.experiment_name,
        folder_name,
        time.strftime('%Y.%m.%d--%H-%M-%S'))
    results_path = os.path.join(results_dir, RESULTS_FILENAME)
    os.makedirs(results_dir)
    csv_header = ["Image", "No attack"]
    error_rates_no_attack, _, _, _ = eval(images, hidden_net, args.batch_size,
                                          hidden_config.message_length, lambda img: img, device)
    error_rates_all: List[List[float]] = [error_rates_no_attack]
    if args.attack == "rotate":
        angles = [2, 5, 10, 15, 20, 25, 30, 35, 45, 50, 60, 70, 80, 90]
        avg_error_per_angle = []
        for angle in angles:
            # randomly switch sign of angle
            if random.random() < 0.5:
                angle = -angle
            error_rates, error_avg, ssim_avg, attack_images = eval(images, hidden_net,
                                                                   args.batch_size, hidden_config.message_length,
                                                                   lambda img: TF.rotate(
                                                                       img, angle, interpolation=TF.InterpolationMode.BILINEAR),
                                                                   device)

            avg_error_per_angle.append(error_avg)

            error_rates_all.append(error_rates)
            csv_header.append(f"{angle} degree rotation")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"{angle}-degrees"))

            print_and_write(f"Results {angle} degree rotation", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)
            print_and_write(
                f"\t Average SSIM = {ssim_avg * 100:.5f}%\n", results_path)

        save_graph(results_dir, angles, avg_error_per_angle,
                   "Rotation angle (Degrees)")
    elif args.attack == "crop":
        crop_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        avg_error_per_ratio = []
        for crop_ratio in crop_ratios:
            crop_width = math.floor(width * crop_ratio)
            crop_height = math.floor(height * crop_ratio)
            crop_top = random.randrange(0, height - crop_height)
            crop_left = random.randrange(0, width - crop_width)
            error_rates, error_avg, ssim_avg, attack_images = eval(images, hidden_net,
                                                                   args.batch_size, hidden_config.message_length,
                                                                   lambda img: TF.crop(
                                                                       img,
                                                                       crop_top,
                                                                       crop_left,
                                                                       crop_height,
                                                                       crop_width),
                                                                   device, False)

            avg_error_per_ratio.append(error_avg)

            error_rates_all.append(error_rates)
            csv_header.append(f"{crop_ratio * 100}% crop")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"ratio-{crop_ratio}"))

            print_and_write(
                f"Results for {crop_ratio * 100}% crop", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)

        crop_ratios_percent = [crop_ratio * 100 for crop_ratio in crop_ratios]
        save_graph(results_dir, crop_ratios_percent,
                   avg_error_per_ratio, "Crop ratio (%)", True)
    elif args.attack == "jpeg":
        qfs = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        avg_error_per_qf = []
        for qf in qfs:
            error_rates, error_avg, ssim_avg, attack_images = eval(images, hidden_net,
                                                                   args.batch_size, hidden_config.message_length,
                                                                   lambda img: jpeg_compress(
                                                                       img, qf, device),
                                                                   device)

            avg_error_per_qf.append(error_avg)

            error_rates_all.append(error_rates)
            csv_header.append(f"QF = {qf}")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"qf-{qf}"))

            print_and_write(
                f"Results for JPEG compression with QF = {qf}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)
            print_and_write(
                f"\t Average SSIM = {ssim_avg * 100:.5f}%\n", results_path)

        save_graph(results_dir, qfs, avg_error_per_qf,
                   "Quality factor (QF)", True)
    elif args.attack == "resize":
        # scales = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2]
        scales = [0.5, 0.6, 0.75, 0.90, 1.25, 1.5, 1.75, 2]
        avg_error_per_scale = []
        for scale in scales:
            resize_height = math.floor(height * scale)
            resize_width = math.floor(width * scale)
            error_rates, error_avg, ssim_avg, attack_images = eval(images, hidden_net,
                                                                   args.batch_size, hidden_config.message_length,
                                                                   lambda img: TF.resize(img,
                                                                                         [resize_height, resize_width], interpolation=TF.InterpolationMode.BILINEAR),
                                                                   device, False)

            avg_error_per_scale.append(error_avg)

            error_rates_all.append(error_rates)
            csv_header.append(f"Resize with scale = {scale}")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"scale-{scale}"))

            print_and_write(
                f"Results for resize with scale = {scale}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)

        save_graph(results_dir, scales, avg_error_per_scale,
                   "Resize scale (x and y)")
    elif args.attack == "shear":
        # angles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75]
        angles = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75]
        avg_error_per_angle = []
        for angle in angles:

            # randomly flip angle
            if random.random() < 0.5:
                angle = -angle
            error_rates, error_avg, ssim_avg, attack_images = eval(images,
                                                                   hidden_net,
                                                                   args.batch_size,
                                                                   hidden_config.message_length,
                                                                   lambda img: TF.affine(
                                                                       img, 0, [0, 0], 1, angle, interpolation=TF.InterpolationMode.BILINEAR),
                                                                   device)

            avg_error_per_angle.append(error_avg)

            error_rates_all.append(error_rates)
            csv_header.append(f"Shear of angle = {angle}")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"{angle}-degrees"))

            print_and_write(
                f"Results for shear of angle = {angle}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)
            print_and_write(
                f"\t Average SSIM = {ssim_avg * 100:.5f}%\n", results_path)

        save_graph(results_dir, angles, avg_error_per_angle,
                   "Shear angle (Degrees)")
    elif args.attack == "translate":
        # distance ratios (images of size [w x h] will be translated with distances [w * dr, h * dr]).
        # drs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        drs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
        avg_error_per_dr = []
        for dr in drs:
            dx = math.floor(width * dr)
            dy = math.floor(height * dr)

            # randomly flip distances
            if random.random() < 0.5:
                dx = -dx

            if random.random() < 0.5:
                dy = -dy

            error_rates, error_avg, ssim_avg, attack_images = eval(images,
                                                                   hidden_net,
                                                                   args.batch_size,
                                                                   hidden_config.message_length,
                                                                   lambda img: TF.affine(
                                                                       img, 0, [dx, dy], 1, [0, 0]),
                                                                   device)

            avg_error_per_dr.append(error_avg)

            error_rates_all.append(error_rates)
            csv_header.append(f"Translation with ratio = {dr}")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"ratio-{dr}"))

            print_and_write(
                f"Results for translation with ratio = {dr}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)
            print_and_write(
                f"\t Average SSIM = {ssim_avg * 100:.5f}%\n", results_path)

        drs_percent = [dr * 100 for dr in drs]
        save_graph(results_dir, drs_percent, avg_error_per_dr,
                   "Translation ratio (Percentage)")
    elif args.attack == "mirror":
        error_rates, error_avg, ssim_avg, attack_images = eval(images, hidden_net,
                                                               args.batch_size, hidden_config.message_length,
                                                               lambda img: TF.hflip(
                                                                   img),
                                                               device)

        error_rates_all.append(error_rates)
        csv_header.append(f"Mirrored")

        if args.save_images:
            save_images(attack_images, filenames, os.path.join(
                results_dir, f"mirrored"))

        print_and_write(f"Results for mirrored", results_path)
        print_and_write(
            f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)
        print_and_write(
            f"\t Average SSIM = {ssim_avg * 100:.5f}%\n", results_path)
    elif args.attack == "blur":
        kernel_sizes = [3, 5, 7, 9]
        avg_error_per_ks = []
        for ks in kernel_sizes:
            error_rates, error_avg, ssim_avg, attack_images = eval(images, hidden_net,
                                                                   args.batch_size, hidden_config.message_length,
                                                                   lambda img: TF.gaussian_blur(
                                                                       img, [ks, ks]),
                                                                   device)

            avg_error_per_ks.append(error_avg)

            error_rates_all.append(error_rates)
            csv_header.append(f"Kernel size = {ks}")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"kernel-size-{ks}"))

            print_and_write(
                f"Results for blur with kernel size = {ks}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)
            print_and_write(
                f"\t Average SSIM = {ssim_avg * 100:.5f}%\n", results_path)

        save_graph(results_dir, kernel_sizes, avg_error_per_ks, "Kernel size")
    elif args.attack == "cropout":
        keep_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        avg_error_per_keep_ratio = []
        for keep_ratio in keep_ratios:
            error_rates, error_avg, ssim_avg, attack_images = eval(images, hidden_net,
                                                                   args.batch_size, hidden_config.message_length,
                                                                   lambda img: cropout(
                                                                       img,
                                                                       images,
                                                                       keep_ratio),
                                                                   device)

            avg_error_per_keep_ratio.append(error_avg)

            error_rates_all.append(error_rates)
            csv_header.append(f"Keep ratio = {keep_ratio * 100}%")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"keep-ratio-{keep_ratio * 100:g}-percent"))

            print_and_write(
                f"Results for cropout with keep ratio = {keep_ratio}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)
            print_and_write(
                f"\t Average SSIM = {ssim_avg * 100:.5f}%\n", results_path)

        keep_ratios_percent = [keep_ratio * 100 for keep_ratio in keep_ratios]

        save_graph(results_dir, keep_ratios_percent,
                   avg_error_per_keep_ratio, "Keep ratio (%)", True)
    elif args.attack == "dropout":
        _, _, _, watermark_images = eval(images, hidden_net,
                                         args.batch_size, hidden_config.message_length,
                                         lambda img: img,
                                         device)
        keep_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        avg_error_per_keep_ratio = []
        for keep_ratio in keep_ratios:
            error_rates, error_avg, ssim_avg, attack_images = eval(images, hidden_net,
                                                                   args.batch_size, hidden_config.message_length,
                                                                   lambda img: dropout(
                                                                       img,
                                                                       images,
                                                                       keep_ratio),
                                                                   device)

            avg_error_per_keep_ratio.append(error_avg)

            error_rates_all.append(error_rates)
            csv_header.append(f"Keep ratio = {keep_ratio * 100}%")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"keep-ratio-{keep_ratio * 100:g}-percent"))

            print_and_write(
                f"Results for dropout with keep ratio = {keep_ratio}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)
            print_and_write(
                f"\t Average SSIM = {ssim_avg * 100:.5f}%\n", results_path)

        keep_ratios_percent = [keep_ratio * 100 for keep_ratio in keep_ratios]

        save_graph(results_dir, keep_ratios_percent,
                   avg_error_per_keep_ratio, "Keep ratio (%)", True)
    elif args.attack == "identity":
        # TODO do different arg parser flow for this case?
        error_rates, error_avg, ssim_avg, watermark_images = eval(images, hidden_net,
                                                                  args.batch_size, hidden_config.message_length,
                                                                  lambda img: img,
                                                                  device)

        if args.save_images:
            save_images(watermark_images,
                        filenames,
                        os.path.join(results_dir, "images"))

        print_and_write(f"Results for encoding without attack", results_path)
        print_and_write(
            f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)
        print_and_write(
            f"\t Average SSIM = {ssim_avg * 100:.5f}%\n", results_path)

        save_images_with_diffs(images, watermark_images,
                               os.path.join(results_dir, "diffs"))

    # transpose list of lists
    csv_rows = list(map(list, zip(filenames, *error_rates_all)))
    write_error_rates(csv_header, csv_rows, os.path.join(
        results_dir, "error-rates.csv"))


def write_error_rates(csv_header, csv_rows, path):
    with open(path, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)


def save_images(images, filenames, folder):
    os.makedirs(folder)
    for img, filename in zip(images, filenames):
        path = os.path.join(folder, filename)
        img = (img + 1) / 2  # restore to [0, 1] range
        save_image(img, path)


def save_images_with_diffs(images: torch.Tensor, watermark_images: List[torch.Tensor], folder):
    os.makedirs(folder)
    image_count = 6
    image_size = 512
    for i in range(image_count):
        random_index = random.randint(0, len(images) - 1)
        img = images[random_index].clip(-1, 1)
        wm_img = watermark_images[random_index].clip(-1, 1)

        img = (img + 1) / 2  # restore to [0, 1] range
        wm_img = (wm_img + 1) / 2  # restore to [0, 1] range

        img = TF.resize(img, (image_size, image_size))
        wm_img = TF.resize(wm_img, (image_size, image_size))

        # diff = torch.abs(img - wm_img)
        # diff = diff * 20
        # diff_mean = torch.mean(diff, dim=0)
        # diff_pil = TF.to_pil_image(diff_mean, mode="L")
        # diff_pil = TF.equalize(diff_pil)

        img_yuv = torch.empty_like(img)
        wm_img_yuv = torch.empty_like(wm_img)
        rgb2yuv(img, img_yuv)
        rgb2yuv(wm_img, wm_img_yuv)
        diff = torch.abs(img_yuv - wm_img_yuv) * 255
        _, h, w = diff.shape
        diff[1] = torch.ones((h, w)) * 0
        diff[2] = torch.ones((h, w)) * 0
        diff = (diff * 15).clip(0, 255)
        diff = (diff / 255)
        yuv2rgb(diff, diff)

        diff = diff.clip(0, 1)

        img_pil = TF.to_pil_image(img, mode="RGB")
        wm_img_pil = TF.to_pil_image(wm_img, mode="RGB")
        # rgb diff to grayscale
        diff_pil = TF.to_pil_image(diff, mode="RGB")
        diff_pil = TF.to_grayscale(diff_pil, num_output_channels=1)

        # save images
        img_pil.save(os.path.join(folder, f"original-{i + 1}.png"))
        wm_img_pil.save(os.path.join(folder, f"watermarked-{i + 1}.png"))
        diff_pil.save(os.path.join(folder, f"diff-{i + 1}.png"))


def rgb2yuv(image_rgb, image_yuv_out):
    """ Transform the image from rgb to yuv """
    image_yuv_out[0, :, :] = 0.299 * image_rgb[0, :, :].clone() + 0.587 * image_rgb[1, :,
                                                                                    :].clone() + 0.114 * image_rgb[2, :,
                                                                                                                   :].clone()
    image_yuv_out[1, :, :] = -0.14713 * image_rgb[0, :, :].clone() + -0.28886 * image_rgb[1, :,
                                                                                          :].clone() + 0.436 * image_rgb[2, :,
                                                                                                                         :].clone()
    image_yuv_out[2, :, :] = 0.615 * image_rgb[0, :, :].clone() + -0.51499 * image_rgb[1, :,
                                                                                       :].clone() + -0.10001 * image_rgb[
        2, :,
        :].clone()


def yuv2rgb(image_yuv, image_rgb_out):
    """ Transform the image from yuv to rgb """
    image_rgb_out[0, :, :] = image_yuv[0, :,
                                       :].clone() + 1.13983 * image_yuv[2, :, :].clone()
    image_rgb_out[1, :, :] = image_yuv[0, :, :].clone() + -0.39465 * image_yuv[1, :,
                                                                               :].clone() + -0.58060 * image_yuv[2, :,
                                                                                                                 :].clone()
    image_rgb_out[2, :, :] = image_yuv[0, :,
                                       :].clone() + 2.03211 * image_yuv[1, :, :].clone()


def print_and_write(s: str, path: str):
    print(s)
    with open(path, "a") as f:
        f.write(s + "\n")


def eval(images, hidden_net: Hidden, batch_size, message_length, attack, device, do_ssim=True):
    ssim = StructuralSimilarityIndexMeasure(data_range=2).to(device)
    image_count = images.shape[0]
    messages = np.random.choice([0, 1],
                                (image_count, message_length))
    messages = torch.Tensor(messages).to(device)
    attack_images = []
    error_rates = []
    error_count = 0
    ssim_sum = 0
    for i in range(0, image_count, batch_size):
        end = min(i + batch_size, image_count)
        batch_imgs = images[i:end].clip(-1, 1)
        batch_msgs = messages[i:end]
        batch_imgs_enc = hidden_net.eval_encode_on_batch(
            batch_imgs, batch_msgs)

        batch_imgs_enc_att = attack((batch_imgs_enc + 1) / 2) * 2 - 1
        batch_msgs_dec = hidden_net.eval_decode_on_batch(batch_imgs_enc_att)

        for img, enc_img_att, msg, msg_dec in zip(batch_imgs, batch_imgs_enc_att, batch_msgs, batch_msgs_dec):
            msg_detached = msg.detach().cpu().numpy()
            msg_dec_rounded = msg_dec.detach().cpu().numpy().round().clip(0, 1)
            msg_error_count = np.sum(
                np.abs(msg_dec_rounded - msg_detached))
            msg_error_rate = msg_error_count / message_length
            error_rates.append(msg_error_rate)
            error_count += msg_error_count

            if do_ssim:
                ssim_sum += ssim(enc_img_att.unsqueeze_(0), img.unsqueeze_(0))

        for att_img in batch_imgs_enc_att:
            attack_images.append(att_img)

    error_avg = error_count / (image_count * message_length)
    ssim_avg = ssim_sum / image_count
    return error_rates, error_avg, ssim_avg, attack_images


def load_images(folder: str, device, image_size) -> Tuple[Tensor, List[str]]:
    images = []
    filenames: List[str] = []
    for item in sorted(os.listdir(folder)):
        path = os.path.join(folder, item)
        if os.path.isfile(path) and not item.startswith("."):
            filenames.append(item)
            img = Image.open(path).convert("RGB")
            if image_size is not None:
                if img.width >= image_size and img.height >= image_size:
                    img = random_crop(img, image_size, image_size)
                else:
                    img = img.resize((image_size, image_size))

            images.append(TF.to_tensor(img).unsqueeze_(0))

    images_tensor = torch.cat(images).to(device)
    images_tensor = images_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    return images_tensor, filenames


def jpeg_compress(images: torch.Tensor, qf: int, device) -> Tensor:
    jpeg_images = []
    for image in images:
        image = (image.clip(-1, 1) + 1) / 2  # to [0, 1] range
        pil_image: Image.Image = TF.to_pil_image(image, "RGB")
        out = io.BytesIO()
        pil_image.save(out, "JPEG", quality=qf)
        out.seek(0)
        image_jpeg = TF.to_tensor(Image.open(out)) * \
            2 - 1  # back to [-1, 1] range
        jpeg_images.append(image_jpeg.unsqueeze_(
            0))  # for some reason torchvision.transforms removes a dimension, and adding one again fixes it

    return torch.cat(jpeg_images).to(device)


def cropout(watermark_images: torch.Tensor, cover_images: torch.Tensor, keep_ratio: float):
    cropout_mask = torch.zeros_like(watermark_images)
    h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=watermark_images,
                                                                 height_ratio_range=(
                                                                     keep_ratio, keep_ratio),
                                                                 width_ratio_range=(keep_ratio, keep_ratio))
    cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

    return watermark_images * cropout_mask + cover_images * (1 - cropout_mask)


def dropout(watermark_images: torch.Tensor, cover_images: torch.Tensor, keep_ratio: float):
    mask = np.random.choice([0.0, 1.0], watermark_images.shape[2:], p=[
                            1 - keep_ratio, keep_ratio])
    mask_tensor = torch.tensor(
        mask, device=watermark_images.device, dtype=torch.float32)
    mask_tensor = mask_tensor.expand_as(watermark_images)
    return watermark_images * mask_tensor + cover_images * (1 - mask_tensor)


def save_graph(folder: str,
               params: Sequence[Union[int, float]],
               bit_errors: List[float],
               xlabel: str,
               desc=False):
    bit_accuracies = [(1 - bit_error) * 100 for bit_error in bit_errors]
    plt.plot(params, bit_accuracies, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel("Bit accuracy (%)")
    plt.ylim(bottom=48, top=102)

    if desc:
        plt.gca().invert_xaxis()

    plt.grid()
    plt.savefig(os.path.join(folder, "graph.png"))

    # save data to csv
    with open(os.path.join(folder, "graph.csv"), "w") as f:
        f.write("param,bit_accuracy\n")
        for param, bit_accuracy in zip(params, bit_accuracies):
            f.write(f"{param},{bit_accuracy}\n")


if __name__ == "__main__":
    main()
