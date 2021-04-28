import os, random, math, glob, sys
from tqdm import tqdm
import numpy as np
# from PIL import Image
import tifffile as tiff
import pickle as pkl

def compute_mean_and_std(root, CHANNEL_NUM = 3, amount = 0.1, selection = False, ext = 'npy'):

    types = ('*.png', '*.jpg', '*.tif', '*.npy')
    training_images = []
    for files in types:
        training_images.extend(glob.glob(root + '/' + files)) 
    # print(glob.glob(root + '/' + files))
    # print(len(training_images))
    # print(training_images[0], training_images[2783], training_images[2784], training_images[-1])

    if selection:
        training_images = random.sample(training_images, math.ceil(len(training_images)*amount))

    pixel_num = 0
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for i in tqdm(training_images):
        if ext == 'npy':
            im = np.load(i).transpose(1,2,0)
            # print(im.shape)
        else:
            im = tiff.imread(i)

        im = im/255.0

        pixel_num += (im.size/CHANNEL_NUM)
        channel_sum += np.sum(im, axis = (0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum/pixel_num
    bgr_std = np.sqrt(channel_sum_squared/pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    stats = [rgb_mean, rgb_std]
    # with open(root + os.sep + 'rgb_stats.pkl', 'wb') as f:
        # pkl.dump(stats, f) 

    return stats

def load_rgb_mean_std(root):
    try:
        stats = []
        
        with open(root + os.sep + 'rgb_stats.pkl', 'rb') as f:
            stats = pkl.load(f)
        
        mean_ = stats[0]
        std_ = stats[1]
    except:
        mean_, std_ = compute_mean_and_std(root = root, amount = 0.3, selection = False)

    return mean_, std_


if __name__ == '__main__':
    root = sys.argv[1]
    # print(root)
    # load_rgb_mean_std(root)
    stats = compute_mean_and_std(root, CHANNEL_NUM = 3, amount = 0.1, selection = False, ext = 'npy')
    # with open(root + os.sep + 'rgb_stats.pkl', 'wb') as f:
    #     pkl.dump(stats, f) 
    print(stats)