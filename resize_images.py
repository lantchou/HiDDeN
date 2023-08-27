import os
from PIL import Image

image_filenames = sorted(os.listdir("../data/data-small/"))
images = [Image.open(os.path.join("../data/data-small", filename))
          for filename in image_filenames if not filename.startswith(".")]
for i, image in enumerate(images):
    image = image.resize((512, 512))
    image.save(os.path.join("./figures-output/combined-vs-rivagan", f"original-{i + 1}.png"))
