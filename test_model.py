import argparse
import os
import csv
import numpy as np
import torch
import torch.nn
from torchmetrics import StructuralSimilarityIndexMeasure
from PIL import Image
import torchvision.transforms.functional as TF
import utils
from model.hidden import *
from noise_layers.noiser import Noiser


TEST_IMAGES_FOLDER = "../data/test/"



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

    hidden_net, hidden_config, train_options = load_model(
        args.options_file, args.checkpoint_file, device)

    test_size = args.test_size
    images = load_test_images(args.image_size, device, args.resize, test_size)

    messages = np.random.choice([0, 1],
                                (test_size, hidden_config.message_length))
    messages = torch.Tensor(messages).to(device)

    encoded_images, ssim_avg, error_avg = save_results(
        images, messages, hidden_net, args, hidden_config.message_length,
        train_options.experiment_name)

    print(f"Average SSIM = {ssim_avg:.5f}")
    print(f'Average bit error = {error_avg:.5f}')

    encoded_images = torch.cat(encoded_images).cpu()
    random_indeces = np.random.choice(test_size, min(test_size, 8))
    utils.save_images(images[random_indeces].cpu(),
                      encoded_images[random_indeces],
                      f'{train_options.experiment_name}-{args.image_size}-{"resize" if args.resize else "crop"}.png',
                      '.')


def load_model(options_file, checkpoint_file, device):
    train_options, hidden_config, noise_config = utils.load_options(
        options_file)
    noiser = Noiser(noise_config, device)

    checkpoint = torch.load(checkpoint_file, device)
    hidden_net = Hidden(hidden_config, device, noiser, None, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)

    return hidden_net, hidden_config, train_options


def load_test_images(img_size, device, resize=True, size=1000):
    images = []
    for index, item in enumerate(os.listdir(TEST_IMAGES_FOLDER)):
        if index == size:
            break

        path = TEST_IMAGES_FOLDER + item
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
    return images_tensor


def random_crop(img, width, height):
    img_width, img_height = img.size
    img = np.array(img)
    assert img_width >= width
    assert img_height >= height
    x_start = np.random.randint(0, img_width - width)
    y_start = np.random.randint(0, img_height - height)
    img = img[y_start:y_start+height, x_start:x_start+width, :]
    return img


def save_results(images, messages, hidden_net, args, message_length, experiment_name):
    csv_filename = f'{experiment_name}-{args.image_size}-{"resize" if args.resize else "crop"}.csv'
    with open(csv_filename, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SSIM", "Bit error rate"])  # header

        ssim = StructuralSimilarityIndexMeasure(data_range=2)
        encoded_images = []
        error_count = 0
        ssim_sum = 0
        for i in range(0, args.test_size, args.batch_size):
            end = min(i + args.batch_size, args.test_size)
            batch_imgs = images[i:end]
            batch_msgs = messages[i:end]
            _, (batch_imgs_enc, _, batch_msgs_dec) = hidden_net.validate_on_batch(
                [batch_imgs, batch_msgs])

            for img, enc_img, msg, msg_dec in \
                    zip(batch_imgs, batch_imgs_enc, batch_msgs, batch_msgs_dec):
                encoded_images.append(enc_img)

                img_ssim = ssim(enc_img.unsqueeze_(0).cpu(), img.unsqueeze_(0).cpu())
                ssim_sum += img_ssim

                msg_detached = msg.detach().cpu().numpy()
                msg_dec_rounded = msg_dec.detach().cpu().numpy().round().clip(0, 1)
                msg_error_count = np.sum(np.abs(msg_dec_rounded - msg_detached))
                msg_error_rate = msg_error_count / message_length
                error_count += msg_error_count

                writer.writerow([img_ssim.item(), msg_error_rate])

        ssim_avg = ssim_sum / args.test_size
        error_avg = error_count / (args.test_size * message_length)
        return encoded_images, ssim_avg, error_avg


if __name__ == '__main__':
    main()
