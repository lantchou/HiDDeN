import torch
import torch.nn
import argparse
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import utils
from model.hidden import *
from noise_layers.noiser import Noiser


TEST_IMAGES_FOLDER = "../data/test/"


def random_crop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


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
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--image-folder', '-i', required=True, type=str,
                        help='Folder with test images')
    parser.add_argument('--source-image', '-s', type=str, help='Source image.')
    args = parser.parse_args()

    hidden_net, hidden_config = load_model(args.options_file, args.checkpoint_file, device)

    # image_pil = Image.open(args.source_image)
    # image_pil = image_pil.resize((hidden_config.H, hidden_config.W))
    # image = np.array(image_pil)
    # image_tensor = TF.to_tensor(image).to(device)
    # image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    # image_tensor.unsqueeze_(0)
    images = load_test_images((hidden_config.H, hidden_config.W), device)
    image_count = images.shape[0]
    messages = np.random.choice([0, 1],
                                (image_count, hidden_config.message_length))
    messages = torch.Tensor(messages).to(device)
    
    encoded_images = []
    error_count = 0
    for i in range(0, image_count, args.batch_size):
        end = min(i + args.batch_size, image_count)
        batch_imgs = images[i:end]
        batch_msgs = messages[i:end]
        _, (batch_imgs_enc, _, batch_msgs_dec) = hidden_net.validate_on_batch([batch_imgs, batch_msgs])

        for img in batch_imgs_enc:
            encoded_images.append(img.unsqueeze_(0))

        batch_msgs_detached = batch_msgs.detach().cpu().numpy()
        batch_msgs_dec_rounded = batch_msgs_dec.detach().cpu().numpy().round().clip(0, 1)
        error_count += np.sum(np.abs(batch_msgs_dec_rounded - batch_msgs_detached))

    error_avg = error_count / (image_count * hidden_config.message_length)
    print(f'Average bit error count: {error_avg:.3f}')

    encoded_images = torch.cat(encoded_images).cpu()
    random_indeces = np.random.choice(image_count, 8)
    utils.save_images(images[random_indeces].cpu(),
                      encoded_images[random_indeces],
                      'test',
                      '.')


def load_model(options_file, checkpoint_file, device):
    _, hidden_config, noise_config = utils.load_options(options_file)
    noiser = Noiser(noise_config, device)

    checkpoint = torch.load(checkpoint_file, device)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)

    return hidden_net, hidden_config


def load_test_images(img_size, device):
    images = []
    for item in os.listdir(TEST_IMAGES_FOLDER):
        path = TEST_IMAGES_FOLDER + item
        if os.path.isfile(path):
            img = Image.open(path).convert("RGB")
            img = img.resize(img_size) 
            img = TF.to_tensor(img)
            images.append(img.unsqueeze_(0))

    # image_tensor = torch.Tensor(images)
    images_tensor = torch.cat(images).to(device)
    images_tensor = images_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    return images_tensor

if __name__ == '__main__':
    main()
