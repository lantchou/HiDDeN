import os
import cv2
import numpy as np
import sys

def calculate_flicker_index(frame_folder):
    frame_files = sorted(os.listdir(frame_folder))
    frame_paths = [os.path.join(frame_folder, filename) for filename in frame_files if not filename.endswith(".")]

    # Calculate flicker index
    flicker_index = 0
    for i in range(len(frame_paths) - 1):
        frame1 = cv2.imread(frame_paths[i])
        frame2 = cv2.imread(frame_paths[i + 1])
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # calculate height and width
        frame1 = frame1.astype(np.float32)
        frame2 = frame2.astype(np.float32)
        frame1 = frame1 / 255
        frame2 = frame2 / 255
        frame_diff = np.abs(frame1 - frame2)
        frame_diff = frame_diff.sum()
        height, width = frame1.shape
        frame_diff = frame_diff / (height * width)
        flicker_index += frame_diff

    flicker_index = flicker_index / (len(frame_paths) - 1)
    return flicker_index

if __name__ == "__main__":
    # read input folder from stdin args
    input_folder = sys.argv[1]
    flicker_index = calculate_flicker_index(input_folder)
    print("Flicker Index:", flicker_index)
