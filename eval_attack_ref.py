'''Attack evaluation for reference watermarking scheme'''
import argparse
import os
import time
from PIL import Image
import torchvision.transforms.functional as TF
from typing import List, Tuple, Callable
import numpy as np
import math
import io
import random

import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder

from eval_attack import save_graph

RESULTS_FILENAME = "results.txt"
GRAPH_FILENAME = "graph.png"
MESSAGE_LENGTH = 32


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate performance of reference model against attack')
    parser.add_argument('--input-folder', '-i', required=True, type=str,
                        help='Folder with input images')
    parser.add_argument("--attack", "-a", required=True, help="Attack type")
    parser.add_argument('--image-size', '-s', default=None, type=int)
    parser.add_argument(
        '--save-images', action=argparse.BooleanOptionalAction, default=False,)

    args = parser.parse_args()

    images, filenames = load_images(args.input_folder, args.image_size)
    height, width, _ = images[0].shape

    # get last part of folder path
    folder_name = os.path.basename(os.path.normpath(args.input_folder))

    results_dir = os.path.join(
        "eval-attack-results",
        args.attack,
        "rivagan",
        folder_name,
        time.strftime('%Y.%m.%d--%H-%M-%S'))
    results_path = os.path.join(results_dir, RESULTS_FILENAME)
    graph_path = os.path.join(results_dir, GRAPH_FILENAME)
    os.makedirs(results_dir)
    if args.attack == "rotate":
        angles = [2, 5, 10, 20, 30, 45, 60, 90]
        avg_error_per_angle = []
        for angle in angles:
            # randomly switch sign of angle
            if random.random() < 0.5:
                angle = -angle

            error_avg, attack_images = eval(
                images, lambda img: rotate(img, angle))

            avg_error_per_angle.append(error_avg)

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"{angle}-degrees"))

            print_and_write(f"Results {angle} degree rotation", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)

        save_graph(graph_path, angles, avg_error_per_angle,
                   "Rotation angle (Degrees)")
    elif args.attack == "crop":
        crop_ratios = [0.9, 0.7, 0.5, 0.3, 0.1]
        avg_error_per_ratio = []
        for crop_ratio in crop_ratios:
            crop_width = math.floor(width * crop_ratio)
            crop_height = math.floor(height * crop_ratio)
            crop_top = random.randrange(0, height - crop_height)
            crop_left = random.randrange(0, width - crop_width)
            error_avg, attack_images = eval(images,
                                            lambda img: img[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width])

            avg_error_per_ratio.append(error_avg)

            print_and_write(
                f"Results for {crop_ratio * 100}% crop", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)

        crop_ratios_percent = [crop_ratio * 100 for crop_ratio in crop_ratios]
        save_graph(graph_path, crop_ratios_percent,
                   avg_error_per_ratio, "Crop ratio (%)", True)
    elif args.attack == "jpeg":
        qfs = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        avg_error_per_qf = []
        for qf in qfs:
            error_avg, attack_images = eval(
                images, lambda img: jpeg_compress(img, qf))

            avg_error_per_qf.append(error_avg)

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"qf-{qf}"))

            print_and_write(
                f"Results for JPEG compression with QF = {qf}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)

        save_graph(graph_path, qfs, avg_error_per_qf,
                   "Quality factor (QF)", True)
    elif args.attack == "resize":
        scales = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2]
        # scales = [1.25, 1.5, 1.75, 2]
        avg_error_per_scale = []
        for scale in scales:
            resize_height = math.floor(height * scale)
            resize_width = math.floor(width * scale)
            error_avg, attack_images = eval(images, lambda img: cv2.resize(
                img,
                (resize_width, resize_height),
                interpolation=cv2.INTER_NEAREST))

            avg_error_per_scale.append(error_avg)

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"scale-{scale}"))

            print_and_write(
                f"Results for resize with scale = {scale}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)

        save_graph(graph_path, scales, avg_error_per_scale,
                   "Resize scale (x and y)")
    elif args.attack == "shear":
        angles = [2, 5, 10, 15, 20, 25, 30, 45, 60]
        avg_error_per_angle = []
        for angle in angles:

            # randomly flip angle
            if random.random() < 0.5:
                angle = -angle
            # perform cv2 shear
            error_avg, attack_images = eval(images,
                                            lambda img: shear(img, angle))

            avg_error_per_angle.append(error_avg)

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"{angle}-degrees"))

            print_and_write(
                f"Results for shear of angle = {angle}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)

        save_graph(graph_path, angles, avg_error_per_angle,
                   "Shear angle (Degrees)")
    elif args.attack == "translate":
        # distance ratios (images of size [w x h] will be translated with distances [w * dr, h * dr]).
        drs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        avg_error_per_dr = []
        for dr in drs:
            dx = math.floor(width * dr)
            dy = math.floor(height * dr)

            # randomly flip distances
            if random.random() < 0.5:
                dx = -dx

            if random.random() < 0.5:
                dy = -dy

            error_avg, attack_images = eval(
                images, lambda img: translate(img, dx, dy))

            avg_error_per_dr.append(error_avg)

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"ratio-{dr}"))

            print_and_write(
                f"Results for translation with ratio = {dr}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)

        drs_percent = [dr * 100 for dr in drs]
        save_graph(graph_path, drs_percent, avg_error_per_dr,
                   "Translation ratio (Percentage)")
    elif args.attack == "mirror":
        error_avg, attack_images = eval(images,
                                        lambda img: cv2.flip(img, 1))

        if args.save_images:
            save_images(attack_images, filenames, os.path.join(
                results_dir, f"mirrored"))

        print_and_write(f"Results for mirrored", results_path)
        print_and_write(
            f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)
    elif args.attack == "blur":
        kernel_sizes = [3, 5, 7, 9]
        avg_error_per_ks = []
        for ks in kernel_sizes:
            assert ks % 2 == 1, "Kernel size must be odd"

            # apply gaussian blur with kernel size ks and same sigma as torchtransforms
            sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
            error_avg, attack_images = eval(images,
                                            lambda img: cv2.GaussianBlur(img, (ks, ks), sigma))

            avg_error_per_ks.append(error_avg)

            if args.save_images:
                save_images(attack_images, filenames, os.path.join(
                    results_dir, f"kernel-size-{ks}"))

            print_and_write(
                f"Results for blur with kernel size = {ks}", results_path)
            print_and_write(
                f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)

        save_graph(graph_path, kernel_sizes, avg_error_per_ks, "Kernel size")
    elif args.attack == "identity":
        error_avg, images = eval(images, lambda img: img)
        print_and_write("Results for encoding without attack", results_path)
        print_and_write(
            f"\t Average bit accuracy = {(1 - error_avg) * 100:.5f}%", results_path)

        if args.save_images:
            save_images(images,
                        filenames,
                        os.path.join(results_dir, f"images"))


def save_images(images, filenames, folder):
    os.makedirs(folder)
    for img, filename in zip(images, filenames):
        path = os.path.join(folder, filename)
        cv2.imwrite(path, img)


def print_and_write(s: str, path: str):
    print(s)
    with open(path, "a") as f:
        f.write(s + "\n")


# attack param is a function
def eval(images: List[np.ndarray], attack: Callable[[np.ndarray], np.ndarray]) -> Tuple[float, List[np.ndarray]]:
    attack_images = []
    error_count = 0.
    encoder = WatermarkEncoder()
    encoder.loadModel()
    decoder = WatermarkDecoder('bits', MESSAGE_LENGTH)
    for img in images:
        # apply watermark
        msg = [random.randint(0, 1) for _ in range(MESSAGE_LENGTH)]
        encoder.set_watermark('bits', msg)
        img_enc: np.ndarray = encoder.encode(img, "rivaGan")

        # apply attack
        img_enc_att = attack(img_enc)
        attack_images.append(img_enc_att)

        # decode and count errors
        msg_dec = decoder.decode(img_enc_att, "rivaGan")
        error_count += sum([abs(msg[i] - msg_dec[i])
                            for i in range(MESSAGE_LENGTH)])

    error_avg = error_count / (len(images) * MESSAGE_LENGTH)
    return error_avg, attack_images


def random_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    height, width, _ = img.shape
    x_start = 0 if crop_size == width else np.random.randint(
        0, width - crop_size)
    y_start = 0 if crop_size == height else np.random.randint(
        0, height - crop_size)
    return img[y_start:y_start+height, x_start:x_start+width, :]


# Load images and apply watermark to them.
def load_images(folder: str, image_size: int | None) -> Tuple[List[np.ndarray], List[str]]:
    images = []
    filenames: List[str] = []

    for item in sorted(os.listdir(folder)):
        path = os.path.join(folder, item)
        if os.path.isfile(path) and not item.startswith("."):
            filenames.append(item)
            img: np.ndarray = cv2.imread(path)
            height, width, _ = img.shape
            if image_size is not None:
                if height >= image_size and width >= image_size:
                    img = random_crop(img, image_size)
                else:
                    img = cv2.resize(img, (image_size, image_size))

            images.append(img)

    return images, filenames


def jpeg_compress(image: np.ndarray, qf: int) -> np.ndarray:
    pil_image: Image.Image = Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    out = io.BytesIO()
    pil_image.save(out, "JPEG", quality=qf)
    out.seek(0)
    image_jpeg = Image.open(out)
    return cv2.cvtColor(np.array(image_jpeg), cv2.COLOR_RGB2BGR)


def rotate(img, angle):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img,
                                   rotation_matrix,
                                   (width, height))
    return rotated_image


def shear(image, angle):
    image_tensor = TF.to_tensor(image)
    sheared_images = TF.affine(image_tensor,
                               angle=0,
                               translate=[0, 0],
                               scale=1,
                               shear=angle,
                               interpolation=TF.InterpolationMode.BILINEAR)
    # return image as cv2 numpy array with shape (height, width, channels)
    return np.array(TF.to_pil_image(sheared_images))


def translate(image, dx, dy):
    height, width, _ = image.shape
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    translated_image = cv2.warpAffine(
        image, translation_matrix, (width, height))
    return translated_image


if __name__ == "__main__":
    main()
