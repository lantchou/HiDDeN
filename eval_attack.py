import argparse
import torch
import os
import time
from PIL import Image
from torch.functional import Tensor
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from typing import List, Tuple
import numpy as np
import csv
import math
import io
import random

from model.hidden import Hidden
from utils import load_model


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
    parser.add_argument('--save-images', '-s', action=argparse.BooleanOptionalAction,
                        default=True, help='Save attacked encoded images')

    args = parser.parse_args()

    images, filenames = load_images(args.input_folder, device)
    _, _, height, width = images.shape

    hidden_net, hidden_config, train_options = load_model(
        args.options_file, args.checkpoint_file, device)

    results_dir = os.path.join(
        "eval-attack-results",
        train_options.experiment_name,
        args.attack,
        time.strftime('%Y.%m.%d--%H-%M-%S'))
    os.makedirs(results_dir)
    csv_header = ["Image", "No attack"]
    error_rates_no_attack, _, _ = eval(images, hidden_net, args.batch_size,
                                       hidden_config.message_length, lambda img: img, device)
    error_rates_all: List[List[float]] = [error_rates_no_attack]
    if args.attack == "rotate":
        angles = [2, 5, 10, 20]
        for angle in angles:
            # randomly switch sign of angle
            if random.random() < 0.5:
                angle = -angle
            error_rates, error_avg, attack_images = eval(images, hidden_net,
                                                         args.batch_size, hidden_config.message_length,
                                                         lambda img: TF.rotate(
                                                             img, angle),
                                                         device)

            error_rates_all.append(error_rates)
            csv_header.append(f"{angle} degree rotation")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"{angle}-degrees"))

            print(f"Results {angle} degree rotation")
            print(f"\t Average bit error = {error_avg:.5f}\n")
    elif args.attack == "crop":
        crop_ratios = [0.9, 0.7, 0.5, 0.3, 0.1]
        for crop_ratio in crop_ratios:
            crop_width = math.floor(width * crop_ratio)
            crop_height = math.floor(height * crop_ratio)
            error_rates, error_avg, attack_images = eval(images, hidden_net,
                                                         args.batch_size, hidden_config.message_length,
                                                         lambda img: TF.crop(
                                                             img, height - crop_height, width - crop_width, crop_height,
                                                             crop_width),
                                                         device)

            error_rates_all.append(error_rates)
            csv_header.append(f"{crop_ratio * 100}% crop")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"ratio-{crop_ratio}"))

            print(f"Results for {crop_ratio * 100}% crop")
            print(f"\t Average bit error = {error_avg:.5f}\n")
    elif args.attack == "jpeg":
        qfs = [100, 80, 60, 40, 20, 10]
        for qf in qfs:
            error_rates, error_avg, attack_images = eval(images, hidden_net,
                                                         args.batch_size, hidden_config.message_length,
                                                         lambda img: jpeg_compress(
                                                             img, qf, device),
                                                         device)

            error_rates_all.append(error_rates)
            csv_header.append(f"QF = {qf}")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"qf-{qf}"))

            print(f"Results for JPEG compression with QF = {qf}")
            print(f"\t Average bit error = {error_avg:.5f}\n")
    elif args.attack == "resize":
        scales = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2]
        for scale in scales:
            resize_height = math.floor(height * scale)
            resize_width = math.floor(width * scale)
            error_rates, error_avg, attack_images = eval(images, hidden_net,
                                                         args.batch_size, hidden_config.message_length,
                                                         lambda img: TF.resize(img, [resize_height, resize_width]),
                                                         device)

            error_rates_all.append(error_rates)
            csv_header.append(f"Resize scale = {scale}")

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"scale-{scale}"))

            print(f"Results for resize with scale = {scale}")
            print(f"\t Average bit error = {error_avg:.5f}\n")
    elif args.attack == "reflect":
        error_rates, error_avg, attack_images = eval(images, hidden_net,
                                                     args.batch_size, hidden_config.message_length,
                                                     lambda img: TF.hflip(img),
                                                     device)

        error_rates_all.append(error_rates)
        csv_header.append(f"Reflected")

        if args.save_images:
            save_images(attack_images, filenames, os.path.join(
                results_dir, f"reflected"))

        print(f"Results for reflected")
        print(f"\t Average bit error = {error_avg:.5f}\n")
    elif args.attack == "identity":
        # identity
        # TODO do different arg parser flow for this case?
        error_rates, error_avg, attack_images = eval(images, hidden_net,
                                                     args.batch_size, hidden_config.message_length,
                                                     lambda img: img,
                                                     device)
        error_rates_all.append(error_rates)

        if args.save_images:
            save_images(attack_images, filenames,
                        os.path.join(results_dir, "images"))

        return  # no csv writing needed

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


def eval(images, hidden_net: Hidden, batch_size, message_length, attack, device):
    image_count = images.shape[0]
    messages = np.random.choice([0, 1],
                                (image_count, message_length))
    messages = torch.Tensor(messages).to(device)
    attack_images = []
    error_rates = []
    error_count = 0
    for i in range(0, image_count, batch_size):
        end = min(i + batch_size, image_count)
        batch_imgs = images[i:end].clip(-1, 1)
        batch_msgs = messages[i:end]
        batch_imgs_enc = hidden_net.eval_encode_on_batch(
            batch_imgs, batch_msgs)
        batch_imgs_enc_att = attack(batch_imgs_enc)
        batch_msgs_dec = hidden_net.eval_decode_on_batch(batch_imgs_enc_att)

        for msg, msg_dec in zip(batch_msgs, batch_msgs_dec):
            msg_detached = msg.detach().cpu().numpy()
            msg_dec_rounded = msg_dec.detach().cpu().numpy().round().clip(0, 1)
            msg_error_count = np.sum(
                np.abs(msg_dec_rounded - msg_detached))
            msg_error_rate = msg_error_count / message_length
            error_rates.append(msg_error_rate)
            error_count += msg_error_count

        for att_img in batch_imgs_enc_att:
            attack_images.append(att_img)

    error_avg = error_count / (image_count * message_length)
    return error_rates, error_avg, attack_images


def load_images(folder: str, device) -> Tuple[Tensor, List[str]]:
    images = []
    filenames: List[str] = []
    for item in sorted(os.listdir(folder)):
        path = os.path.join(folder, item)
        if os.path.isfile(path) and not item.startswith("."):
            filenames.append(item)
            img = Image.open(path).convert("RGB")
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
        image_jpeg = TF.to_tensor(Image.open(out)) * 2 - 1  # back to [-1, 1] range
        jpeg_images.append(image_jpeg.unsqueeze_(
            0))  # for some reason torchvision.transforms removes a dimension, and adding one again fixes it

    return torch.cat(jpeg_images).to(device)


if __name__ == "__main__":
    main()
