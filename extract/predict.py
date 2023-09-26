import os
import numpy as np
from PIL import Image
from skimage.filters import gaussian
from skimage.color import rgb2lab, lab2rgb
from sklearn import cluster
from skimage.io import imread


def read_img(fpath):
    img = Image.open(fpath).convert('RGB')
    return img


def get_hue(a_values, b_values, eps=1e-8):
    """Compute hue angle"""
    return np.degrees(np.arctan(b_values / (a_values + eps)))


def mode_hist(x, bins='sturges'):
    """Compute a histogram and return the mode"""
    hist, bins = np.histogram(x, bins=bins)
    mode = bins[hist.argmax()]
    return mode


def clustering(x, n_clusters=5, random_state=2021):
    model = cluster.KMeans(n_clusters, random_state=random_state)
    model.fit(x)
    return model.labels_, model


def get_scalar_values(skin_smoothed_lab, labels, topk=3, bins='sturges'):
    # gather values of interest
    hue_angle = get_hue(skin_smoothed_lab[:, 1], skin_smoothed_lab[:, 2])
    skin_smoothed = lab2rgb(skin_smoothed_lab)

    # concatenate data to be clustered (L, h, and RGB for visualization)
    data_to_cluster = np.vstack([skin_smoothed_lab[:, 0], hue_angle,
                                 skin_smoothed[:, 0], skin_smoothed[:, 1], skin_smoothed[:, 2]]).T

    # Extract skin pixels for each mask (by clusters)
    n_clusters = len(np.unique(labels))
    masked_skin = [data_to_cluster[labels == i, :] for i in range(n_clusters)]
    n_pixels = np.asarray([np.sum(labels == i) for i in range(n_clusters)])

    # get scalar values per cluster
    keys = ['lum', 'hue', 'red', 'green', 'blue']
    res = {}

    for i, key in enumerate(keys):
        res[key] = np.array([mode_hist(part[:, i], bins=bins)
                             for part in masked_skin])

    # only keep top3 in luminance and avarage results
    idx = np.argsort(res['lum'])[::-1][:topk]
    total = np.sum(n_pixels[idx])

    res_topk = {}
    for key in keys:
        res_topk[key] = np.average(res[key][idx], weights=n_pixels[idx])
        res_topk[key+'_std'] = np.sqrt(np.average((res[key][idx]-res_topk[key])**2, weights=n_pixels[idx]))
    return res_topk


def get_skin_values(img, mask, n_clusters=5):
    # smoothing
    img_smoothed = gaussian(img, sigma=(1, 1), truncate=4, multichannel=True)

    # get skin pixels (shape will be Mx3) and go to Lab
    skin_smoothed = img_smoothed[mask]
    skin_smoothed_lab = rgb2lab(skin_smoothed)

    res = {}

    # L and hue
    hue_angle = get_hue(skin_smoothed_lab[:, 1], skin_smoothed_lab[:, 2])
    data_to_cluster = np.vstack([skin_smoothed_lab[:, 0], hue_angle]).T
    labels, model = clustering(data_to_cluster, n_clusters=n_clusters)
    tmp = get_scalar_values(skin_smoothed_lab, labels)
    res['lum'] = tmp['lum']
    res['hue'] = tmp['hue']
    res['lum_std'] = tmp['lum_std']
    res['hue_std'] = tmp['hue_std']

    # also extract RGB for visualization purposes
    res['red'] = tmp['red']
    res['green'] = tmp['green']
    res['blue'] = tmp['blue']
    res['red_std'] = tmp['red_std']
    res['green_std'] = tmp['green_std']
    res['blue_std'] = tmp['blue_std']

    return res


def main():
    attrs = ['lum', 'hue']
    res = {}
    for attr in attrs:
        res[attr] = []
        res[attr+'_std'] = []

    # reading images
    fimg = 'samples/00000.png'
    fmask = 'samples/00000_mask.png'

    img_original = imread(fimg)
    mask = imread(fmask)

    # get values
    tmp = get_skin_values(np.asarray(img_original),
                          np.asarray(mask) == 1)
    for attr in attrs:
        res[attr].append(tmp[attr])
        res[attr+'_std'].append(tmp[attr+'_std'])

    print(res)


if __name__ == "__main__":
    main()
