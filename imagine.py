#########################           IMPORTS           ##########################
from PIL import Image
import pandas as pd
import numpy as np
import math
########################           FUNCTIONS           #########################
def make_kernel(radius, sigma):
    ''' used from clemisch on stack overflow 
    their original code can be viewed here:
    https://stackoverflow.com/a/43346070'''
    ax = np.linspace(-radius, radius, 2 * radius + 1)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def add_padding(img: np.array, pad: int):
    pad_img = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad))
    p_shape = pad_img.shape
    # filling corners
    pad_img[0: pad, 0: pad] = img[0][0]
    pad_img[p_shape[0] - pad: p_shape[0], 0: pad] = img[img.shape[0] - 1][0]
    pad_img[0: pad, p_shape[1] - pad: p_shape[1]] = img[0][img.shape[1] - 1]
    pad_img[p_shape[0] - pad: p_shape[0], p_shape[1] - pad: p_shape[1]] = img[img.shape[0] - 1][img.shape[1] - 1]
    # filling sides
    pad_img[pad:p_shape[0]-pad, 0:pad] = np.repeat(img[:, 0, np.newaxis], pad, axis=1)
    pad_img[0:pad, pad:p_shape[1]-pad] = np.repeat(img[np.newaxis, 0, :], pad, axis=0)
    pad_img[pad:p_shape[0]-pad, p_shape[1]-pad:p_shape[1]] = np.repeat(img[:, img.shape[1]-1, np.newaxis], pad, axis=1)
    pad_img[p_shape[0]-pad:p_shape[0], pad:p_shape[1]-pad] = np.repeat(img[np.newaxis, img.shape[0]-1, :], pad, axis=0)

    
    pad_img[pad: p_shape[0] - pad, pad: p_shape[1] - pad] = img
    return pad_img

def gaussian_blur(in_img: np.array):
    rad = 2
    out_img = np.zeros(in_img.shape)
    pad_img = add_padding(in_img, rad)
    kernel = make_kernel(rad, sigma=1)
    for h in range(in_img.shape[0]):
        for w in range(in_img.shape[1]):
            # move through kernel
            kernel_sum = 0
            for i in range(-rad, rad + 1):
                for j in range(-rad, rad + 1):
                    kernel_sum += pad_img[h + i + rad, w + j + rad] * kernel[i + rad, j + rad]
            out_img[h, w] = kernel_sum

    return out_img

def sobel_filter(in_img: np.array):
    x_img, y_img = np.zeros(in_img.shape), np.zeros(in_img.shape)
    pad_img = add_padding(in_img, 1)
    g_x = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    g_y = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for h in range(in_img.shape[0]):
        for w in range(in_img.shape[1]):
            # move through the kernel
            x_sum, y_sum = 0, 0
            for i in range(3):
                for j in range(3):
                    x_sum += pad_img[h + i - 1, w + j - 1] * g_x[i, j]
                    y_sum += pad_img[h + i - 1, w + j - 1] * g_y[i, j]
            x_img[h, w], y_img[h, w] = x_sum, y_sum
    # finding magnitudes
    g = np.sqrt(np.square(x_img) + np.square(y_img))
    g = np.uint8(255 * (g / np.max(g)))
    # Finding directions
    x_img += 0.01
    th = np.divide(y_img, x_img, out=np.zeros_like(y_img), where=x_img!=0)
    th = ((360 / (2 * 3.1415)) * np.arctan(th))
    theta = np.zeros(th.shape)
    theta[th < -67.5] = 90
    theta[th >= -67.5] = 45
    theta[th >= -22.5] = 0
    theta[th >= 22.5] = 135
    theta[th >= 67.5] = 90
    return g, theta

def magnitude_thresholding(in_img: np.array, th_img: np.array):
    out_img = np.zeros(in_img.shape)
    pad_img = add_padding(in_img, 1)
    for h in range(in_img.shape[0]):
        for w in range(in_img.shape[1]):
            intensity = in_img[h, w]
            if th_img[h][w] == 0:
                if intensity >= pad_img[h+1][w] and intensity >= pad_img[h+1][w+2]:
                    out_img[h][w] = intensity
            elif th_img[h][w] == 45:
                if intensity >= pad_img[h][w+2] and intensity >= pad_img[h+2][w-1]:
                    out_img[h][w] = intensity
            elif th_img[h][w] == 90:
                if intensity >= pad_img[h][w+1] and intensity >= pad_img[h+2][w+1]:
                    out_img[h][w] = intensity
            elif th_img[h][w] == 135:
                if intensity >= pad_img[h][w] and intensity >= pad_img[h+2][w+2]:
                    out_img[h][w] = intensity
    return out_img

def double_threshold(in_img: np.array, min: float, max: float):
    min_thresh, max_thresh = min, max
    out_img = np.zeros(in_img.shape)

    out_img[in_img > 255 * min_thresh] = 127
    out_img[in_img > 255 * max_thresh] = 255

    for w in range(out_img.shape[1]):
        if out_img[0, w] == 255:
            out_img[0, w] = 127
    
    return out_img

def hysteresis(in_img: np.array):
    out_img = np.zeros(in_img.shape)
    pad_img = add_padding(in_img, 1)
    for h in range(in_img.shape[0]):
        for w in range(in_img.shape[1]):
            if in_img[h][w] == 127:
                near_strong = False
                for i in range(h, h+3):
                    for j in range(w, w+3):
                        if pad_img[i][j] == 255:
                            near_strong = True
                if not near_strong:
                    out_img[h][w] = 0
                else: out_img[h][w] = 255
            elif in_img[h][w] == 255:
                out_img[h][w] = 255
    return out_img

def line_thickener(in_img: np.array, radius: int):
    pad_img = add_padding(in_img, radius)
    for h in range(in_img.shape[0]):
        for w in range(in_img.shape[1]):
            if in_img[h, w] == 0:
                for i in range(h, h + 2 * radius + 1):
                    for j in range(w, w + 2 * radius + 1):
                        pad_img[i, j] = 0
    out_img = pad_img[radius:pad_img.shape[0] - radius, radius:pad_img.shape[1] - radius]
    return out_img

def gamma_correct(in_img: np.array, gamma: float):
    out_img = np.copy(in_img)
    out_img = (out_img / 255) ** (1 / gamma)
    return out_img * 255

def dither(in_img: np.array, coeff: float):
    out_img = np.zeros(in_img.shape)
    noise = np.array(Image.open(r'noise.png'))
    noise = np.delete(noise, [1, 2, 3], 2).reshape(noise.shape[0], noise.shape[1])
    if in_img.shape > noise.shape:
        tile_grid = (math.ceil(in_img.shape[0] / noise.shape[0]), math.ceil(in_img.shape[1] / noise.shape[1]))
        noise = np.tile(noise, tile_grid)
    
    for h in range(in_img.shape[0]):
        for w in range(in_img.shape[1]):
            if in_img[h, w] > noise[math.floor(coeff * h), math.floor(coeff * w)]:
                out_img[h, w] = 255
            else:
                out_img[h, w] = 0
    return out_img

def color_filter(in_img: np.array, color_1, color_2):
    out_img = np.zeros((in_img.shape[0], in_img.shape[1], 3))
    for h in range(in_img.shape[0]):
        for w in range(in_img.shape[1]):
            if in_img[h, w] == 255: 
                out_img[h, w] = color_1
            else: out_img[h,w] = color_2
    return out_img

def make_image(in_img: np.array):
    out_img = Image.fromarray(np.uint8(np.stack([in_img, in_img, in_img], axis=2)))
    return out_img

######################           OPENING IMAGE           #######################
IMAGE_ADDRESS = r'example.jpg'
img = np.array(Image.open(IMAGE_ADDRESS))
if img.shape[2] == 4:
    img = np.delete(img, 3, 2)

#####################           BLACK AND WHITE           ######################
print('# turning to greyscale...')
bw_img = np.uint8(img.mean(axis=2))

######################           EDGE DETECTION           ######################
print('# detecting edges...')
# Gaussian blur
print('*   gaussian blurring')
gb_img = gaussian_blur(bw_img)
# intensity gradient
print('*   finding intensity gradient')
sf_img, th_img = sobel_filter(gb_img)
# magnitude thresholding
print('*   magnitude thresholding')
mt_img = magnitude_thresholding(sf_img, th_img)
# double thresholding
print('*   double thresholding')
dt_img = double_threshold(mt_img, 0.1, 0.2)
# hysteresis
print('*   hysteresis')
hy_img = 255 - hysteresis(dt_img)
# line thickening
print('*   thickening lines')
tl_img = line_thickener(hy_img, 1)

########################           DITHERING           #########################
print('# dithering')
print('*   adjusting brightness...')
gc_img = gamma_correct(bw_img, 1.0)
print('*   blue noise dithering...')
di_img = dither(gc_img, coeff = 0.4)

########################           COMBINING           #########################
combined_img = di_img - (255 - tl_img)
combined_img[combined_img < 0] = 0

colored = np.uint8(color_filter(combined_img, [245, 242, 225], [25, 26, 36]))

final = Image.fromarray(colored)
final = final.save('final_img.png')
print('# finished')
