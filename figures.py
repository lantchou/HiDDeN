import cv2
import numpy as np
import os
import random
from PIL import Image
import io
import torch
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import csv

from noise_layers.jpeg_diff import JpegDiff
from eval_attack_ref import shear


def rotate_examples():
    filenames = os.listdir("../data/test")
    filename = random.choice(filenames)
    image = Image.open(f"../data/test/{filename}")
    image_rotated = image.rotate(15, resample=Image.BILINEAR)
    image.resize((256, 256)).save(f"figures-output/rotated-original.png")
    image_rotated.resize((256, 256)).save(f"figures-output/rotated.png")


def translation_examples():
    filenames = os.listdir("../data/test")
    filename = random.choice(filenames)
    image = Image.open(f"../data/test/{filename}")
    image_translated = image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, -image.size[0] * 0.2, 0, 1, -image.size[1] * 0.2),
        resample=Image.BILINEAR,
    )
    image.resize((256, 256)).save(f"figures-output/translated-original.png")
    image_translated.resize((256, 256)).save(f"figures-output/translated.png")


def shear_examples():
    filenames = os.listdir("../data/test")
    filename = random.choice(filenames)
    image = cv2.imread(f"../data/test/{filename}")
    image_sheared = shear(image, 30)
    cv2.imwrite(f"figures-output/sheared-original.png",
                cv2.resize(image, (256, 256)))
    cv2.imwrite(f"figures-output/sheared.png",
                cv2.resize(image_sheared, (256, 256)))


def mirror_examples():
    filenames = os.listdir("../data/test")
    filename = random.choice(filenames)
    image = Image.open(f"../data/test/{filename}")
    image_mirrored = image.transpose(Image.FLIP_LEFT_RIGHT)
    image.resize((256, 256)).save(f"figures-output/mirrored-original.png")
    image_mirrored.resize((256, 256)).save(f"figures-output/mirrored.png")


def resize_example():
    filenames = os.listdir("../data/test")
    filename = random.choice(filenames)
    image = cv2.imread(f"../data/test/{filename}")
    height, width, _ = image.shape
    new_width, new_height = int(width * 0.70), int(height * 0.70)
    image_smaller = cv2.resize(image, (new_width, new_height))
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = image_smaller
    # make all remaining values 255
    new_image[new_height:, new_width:] = 255
    new_image[new_height:, 0:new_width] = 255
    new_image[0:new_height, new_width:] = 255

    cv2.imwrite(f"figures-output/resize-original.png",
                cv2.resize(image, (256, 256)))
    cv2.imwrite(f"figures-output/resize.png",
                cv2.resize(new_image, (256, 256)))


def torch_to_cv2(image):
    image = image.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.uint8)
    return image


def cv2_to_torch(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    return image


def jpeg_compress(image: torch.Tensor, qf: int) -> torch.Tensor:
    pil_image: Image.Image = TF.to_pil_image(image, "RGB")
    out = io.BytesIO()
    pil_image.save(out, "JPEG", quality=qf)
    out.seek(0)
    image_jpeg = TF.to_tensor(Image.open(out))
    return image_jpeg


def jpeg_diff_example():
    device = torch.device("cpu")
    filenames = os.listdir("../data/test")
    filename = random.choice(filenames)
    print("image: ../data/test/" + filename)
    image = cv2.imread(f"../data/test/{filename}")
    image = cv2.resize(image, (352, 352))
    cv2.imwrite(f"figures-output/jpeg/original.png", image)
    image_tensor = cv2_to_torch(image).to(device)

    quality = 40

    image_jpeg = jpeg_compress(image_tensor, quality).to(device)
    jpeg_diff = JpegDiff(device, quality=quality)
    jpeg_diff.eval()
    save_image(image_jpeg, f"figures-output/jpeg/jpeg-real.png")
    image_jpeg_diff, _ = jpeg_diff([image_tensor.unsqueeze(0), None])
    image_jpeg_diff = image_jpeg_diff.squeeze_(0)
    save_image(image_jpeg_diff, f"figures-output/jpeg/jpeg-diff.png")

    # calculate diff
    amplification = 15
    diff = (image_jpeg - image_jpeg_diff).abs()
    diff = diff * amplification
    diff_avg = torch.mean(diff, dim=0)
    diff_avg_pil = TF.to_pil_image(diff_avg, "L")
    diff_avg_pil.save(f"figures-output/jpeg/diff.png")



# def robustness_graphs():
#     plt.figure()

#     # crop graph
#     crop_xlabels, crop_identity_accs, id_pos = get_csv_data(
#         "robustness/crop/identity.csv")
#     _, crop_combined_accs, _ = get_csv_data("robustness/crop/combined.csv")
#     crop_params = [
#         1 if i == id_pos else float(x)
#         for i, x in enumerate(crop_xlabels)
#     ]
#     plt.gca().set_xticklabels(crop_xlabels)
#     plt.xticks(crop_params)
#     plt.plot(crop_params, crop_identity_accs)
#     plt.plot(crop_params,
#              [acc - 0.15 for acc in crop_combined_accs],
#              linestyle="-")

#     plt.gca().invert_xaxis()

#     plt.xlabel("Crop (p)")
#     plt.ylabel("Bit accuracy (%)")
#     plt.ylim(bottom=48, top=102)

#     plt.legend(["Identity", "Combined"])

#     ax = plt.gca()
#     ax.grid(True)

#     # plt.savefig("figures-output/robustness/crop.png")
#     plt.show()


if __name__ == "__main__":
    # rotate_examples()
    # translation_examples()
    # shear_examples()
    # mirror_examples()
    # resize_example()
    # jpeg_diff_example()
    pass
