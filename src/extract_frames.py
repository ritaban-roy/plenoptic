import cv2
import os
import glob
from skimage import io
from skimage.transform import resize
import numpy as np

def gamma_correction(x):
    x = np.where(x <= 0.0404482, x/12.92, ((x+0.055)/1.055)**2.4)
    return x

def extract_frames():
    video_file = '../data/video_2.mp4'
    output_dir = '../data/video_2/'
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if(frame_count % 2 == 0):
            frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
            cv2.imwrite(frame_filename, frame)

        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved {frame_count} frames in the '{output_dir}' directory.")

def frames_to_numpy():
    frame_files = sorted(glob.glob('../data/video_2/*.png'))
    frames = []
    for frame_file in frame_files:
        img = io.imread(frame_file)/255.0
        img = resize(img, (img.shape[0]//1.5, img.shape[1]//1.5), anti_aliasing=True)
        #img = gamma_correction(img)
        #img = (img[:, :, 0]*0.2126 + img[:, :, 1]*0.7152 + img[:, :, 2]*0.0722).squeeze()
        frames.append(img)
    frames = np.array(frames)
    np.savez('video_2.npz', frames)

def frames_to_numpy_gray():
    frame_files = sorted(glob.glob('../data/video_2/*.png'))
    frames = []
    for frame_file in frame_files:
        img = io.imread(frame_file)/255.0
        img = resize(img, (img.shape[0]//1.5, img.shape[1]//1.5), anti_aliasing=True)
        img = gamma_correction(img)
        img = (img[:, :, 0]*0.2126 + img[:, :, 1]*0.7152 + img[:, :, 2]*0.0722).squeeze()
        frames.append(img)
    frames = np.array(frames)
    np.savez('short_video_2_gray.npz', frames)

if __name__ == '__main__':
    extract_frames()
    frames_to_numpy()
    frames_to_numpy_gray()
        
