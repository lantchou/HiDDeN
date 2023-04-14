import argparse
import os
import csv
import numpy as np
import torch
import torch.nn
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from PIL import Image
import torchvision.transforms.functional as TF
import utils
from model.hidden import *

TEST_IMAGES_FOLDER = "../data/test/"
TEST_RESULTS_FOLDER = "./test-results"

def main():
    if torch.has_mps:
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True,
                        type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12,
                        type=int, help='The batch size.')
    parser.add_argument('--image-folder', '-i', required=True, type=str,
                        help='Folder with test images')
    parser.add_argument('--image-size', '-s', type=int,
                        default=128, help='Image size')
    parser.add_argument('--test-size', '-t', type=int,
                        default=len(os.listdir(TEST_IMAGES_FOLDER)), help='Test size')
    parser.add_argument('--resize', '-r', action=argparse.BooleanOptionalAction,
                        default=False, help='Resize if true, else crop')
    args = parser.parse_args()

    hidden_net, hidden_config, train_options = utils.load_model(
        args.options_file, args.checkpoint_file, device)

    images, filenames = load_test_images(
        args.image_size, device, args.test_size, args.resize)
    test_size = len(images)

    messages = np.random.choice([0, 1],
                                (test_size, hidden_config.message_length))
    messages = torch.Tensor(messages).to(device)

    ssim = StructuralSimilarityIndexMeasure(data_range=2).to(device)
    encoded_images, ssim_avg, error_avg = save_results(
        images, filenames, messages, hidden_net, args, hidden_config.message_length,
        train_options.experiment_name, ssim)

    print(f"Average SSIM = {ssim_avg:.5f}")
    print(f'Average bit error = {error_avg:.5f}')

    encoded_images = torch.cat(encoded_images).cpu()
    random_indeces = np.random.choice(test_size, min(test_size, 8))
    utils.save_images(images[random_indeces].cpu(),
                      encoded_images[random_indeces],
                      f'{train_options.experiment_name}-{args.image_size}-{"resize" if args.resize else "crop"}.png',
                      '.')


def load_test_images(img_size, device, size, resize=True):
    images = []
    filenames = []
    for item in sorted(os.listdir(TEST_IMAGES_FOLDER)):
        if len(images) == size:
            break

        path = TEST_IMAGES_FOLDER + item
        filenames.append(item)
        if os.path.isfile(path):
            img = Image.open(path).convert("RGB")
            if resize:
                img = img.resize((img_size, img_size))
                np_img = np.array(img)
            else:
                # crop instead of resizing
                if img.size[0] < img_size or img.size[1] < img_size:
                    img.close()
                    continue

                np_img = random_crop(img, img_size, img_size)
            img.close()
            images.append(TF.to_tensor(np_img).unsqueeze_(0))

    print(f"Loaded in {len(images)} testing images")

    images_tensor = torch.cat(images).to(device)
    images_tensor = images_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    return images_tensor, filenames


def random_crop(img, width, height):
    img_width, img_height = img.size
    img = np.array(img)
    assert img_width >= width
    assert img_height >= height
    x_start = 0 if img_width == width else np.random.randint(0, img_width - width)
    y_start = 0 if img_height == height else np.random.randint(0, img_height - height)
    img = img[y_start:y_start+height, x_start:x_start+width, :]
    return img


def save_results(images, filenames, messages, hidden_net, args, message_length, experiment_name, ssim):
    csv_filename = f'{experiment_name}-{args.image_size}-{"resize" if args.resize else "crop"}.csv'
    csv_path = os.path.join(TEST_RESULTS_FOLDER, csv_filename)
    if not os.path.isdir(TEST_RESULTS_FOLDER):
        os.mkdir(TEST_RESULTS_FOLDER)
    image_count = images.shape[0]
    with open(csv_path, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "SSIM", "Bit error rate"])  # header

        encoded_images = []
        error_count = 0
        ssim_sum = 0
        for i in range(0, image_count, args.batch_size):
            end = min(i + args.batch_size, image_count)
            batch_imgs = images[i:end]
            batch_msgs = messages[i:end]
            batch_imgs_enc = hidden_net.eval_encode_on_batch(
                batch_imgs, batch_msgs)
            batch_msgs_dec = hidden_net.eval_decode_on_batch(batch_imgs_enc)

            j = 0
            for img, enc_img, msg, msg_dec in \
                    zip(batch_imgs, batch_imgs_enc, batch_msgs, batch_msgs_dec):
                encoded_images.append(enc_img)

                img_ssim = ssim(enc_img.unsqueeze_(
                    0), img.unsqueeze_(0))
                ssim_sum += img_ssim

                msg_detached = msg.detach().cpu().numpy()
                msg_dec_rounded = msg_dec.detach().cpu().numpy().round().clip(0, 1)
                msg_error_count = np.sum(
                    np.abs(msg_dec_rounded - msg_detached))
                msg_error_rate = msg_error_count / message_length
                error_count += msg_error_count

                writer.writerow(
                    [filenames[i + j], img_ssim.item(), msg_error_rate])
                j += 1

        ssim_avg = ssim_sum / image_count
        error_avg = error_count / (image_count * message_length)
        return encoded_images, ssim_avg, error_avg


if __name__ == '__main__':
    main()
