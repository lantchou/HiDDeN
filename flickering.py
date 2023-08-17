import os
import cv2
import numpy as np

def calculate_flicker_index(frame_folder):
    frame_files = sorted(os.listdir(frame_folder))
    frame_paths = [os.path.join(frame_folder, filename) for filename in frame_files]
    print(frame_paths)


    # Calculate flicker index
    flicker_index = 0
    for i in range(len(frame_paths) - 1):
        frame1 = cv2.imread(frame_paths[i])
        frame2 = cv2.imread(frame_paths[i + 1])
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame1 = cv2.resize(frame1, (128, 128))
        frame2 = cv2.resize(frame2, (128, 128))
        # calculate height and width
        frame1 = frame1.astype(np.float32)
        frame2 = frame2.astype(np.float32)
        frame1 = frame1 / 255
        frame2 = frame2 / 255
        frame_diff = np.abs(frame1 - frame2)
        frame_diff = frame_diff / 255
        frame_diff = frame_diff.sum()
        height, width = frame1.shape
        frame_diff = frame_diff / (height * width)
        flicker_index += frame_diff

    flicker_index = flicker_index / (len(frame_paths) - 1)
    return flicker_index

if __name__ == "__main__":
    input_folder = input("Input folder: ")
    flicker_index = calculate_flicker_index(input_folder)
    print("Flicker Index:", flicker_index)
