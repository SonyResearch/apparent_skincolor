import os
import cv2
import argparse
import itertools as it
import numpy as np
import pandas as pd
from tqdm import tqdm


np.random.seed(2022)

# Image size with preserved ratio
WIDTH = 859
HEIGHT = 611

# Number of pairs for every combination
N_SAMPLES = 500

# We are interested in the following gender and ethnicities
genders = ['F', 'M']
ethnicities = ['W', 'B', 'A', 'I', 'L']


def get_img_resize(fpath, width=WIDTH, height=HEIGHT):
    img = cv2.imread(fpath)[:, :, ::-1]
    img_resize = cv2.resize(img, (width, height))
    return img_resize


def save_img(top_fpaths, bottom_fpaths, save_fpaths,
             data_dir, save_dir):
    """Given lists of image pairs, save their concatenated version"""
    for top_fpath, bottom_fpath, save_fpath in tqdm(zip(top_fpaths, bottom_fpaths, save_fpaths)):
        top_img = get_img_resize(os.path.join(data_dir, top_fpath))
        bottom_img = get_img_resize(os.path.join(data_dir, bottom_fpath))

        merged_img = np.ones((3000, WIDTH, 3), dtype=np.uint8) * 255
        merged_img[:HEIGHT, :WIDTH, :] = top_img
        merged_img[-HEIGHT:, :WIDTH, :] = bottom_img

        cv2.imwrite(os.path.join(save_dir, save_fpath), merged_img[:, :, ::-1])


def collect_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='CFD/CFDv3/Images',
                        help='path to CFD dataset')
    parser.add_argument('--save_dir', type=str,
                        default='500_aspect_ratio/',
                        help='path where to save the image pairs')
    opt = vars(parser.parse_args())
    return opt


def main():
    opt = collect_arg()

    product = list(it.product(ethnicities, genders))
    combinations = list(it.combinations(product, 2))

    base_df = pd.read_csv('cfd_meta.csv', index_col=0)

    res = {}

    res['file_name'] = []
    res['file_top'] = []
    res['file_bottom'] = []
    res['combination'] = []

    count = 0
    for i in range(len(combinations)):
        # get the top and bottom values
        top = combinations[i][0]
        bottom = combinations[i][1]

        cond_top = (base_df['EthnicitySelf'] == top[0]) & (base_df['GenderSelf'] == top[1])
        cond_bottom = (base_df['EthnicitySelf'] == bottom[0]) & (base_df['GenderSelf'] == bottom[1])

        # get the filenames for corresponding top and bottom values
        fnames = list(it.product(base_df.loc[cond_top].index.values,
                                 base_df.loc[cond_bottom].index.values))
        np.random.shuffle(fnames)

        # limit to the number of samples
        top_fpaths = [fname[0] for fname in fnames[:N_SAMPLES]]
        bottom_fpaths = [fname[1] for fname in fnames[:N_SAMPLES]]

        # file path plumbing
        txt_combination = top[0] + top[1] + bottom[0] + bottom[1]
        save_fpaths = [(txt_combination + '_image_%d.png' % i)
                       for i in range(count, count + N_SAMPLES)]

        save_img_dir = os.path.join(opt['save_dir'], 'images')
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)

        # saving images
        save_img(top_fpaths, bottom_fpaths, save_fpaths,
                 data_dir=opt['data_dir'], save_dir=save_img_dir)

        count += N_SAMPLES

        res['file_name'].extend(save_fpaths)
        res['file_top'].extend(top_fpaths)
        res['file_bottom'].extend(bottom_fpaths)
        res['combination'].extend([txt_combination for i in range(N_SAMPLES)])

    # save annotations
    df = pd.DataFrame(res)
    df = df.set_index('file_name')
    df.to_csv(os.path.join(opt['save_dir'], 'cfd_all_pairs.csv'))


if __name__ == '__main__':
    main()
