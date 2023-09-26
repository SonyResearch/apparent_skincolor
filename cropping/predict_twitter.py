import subprocess
import os
import sys
import platform
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# path to the Twitter repository
TWITTER_CROP_REPO = 'image-crop-analysis'

# path to the images coming from generate.py
DATA_DIR = '500_aspect_ratio/images'
EXP_DF = pd.read_csv('500_aspect_ratio/cfd_all_pairs.csv')
SAVE_PATH = 'results_500_aspect_ratio.csv'

# Twitter API plumbing
BIN_MAPS = {"Darwin": "mac", "Linux": "linux"}
HOME_DIR = Path(TWITTER_CROP_REPO).expanduser()
sys.path.append(str(HOME_DIR / "src"))
bin_dir = HOME_DIR / Path("./bin")
bin_path = bin_dir / BIN_MAPS[platform.system()] / "candidate_crops"
model_path = bin_dir / "fastgaze.vxm"
from crop_api import ImageSaliencyModel, is_symmetric, parse_output, reservoir_sampling


def twitter(data_dir, file_names):
    """Use Twitter cropping API to predict which image (top or bottom) survives"""
    res = {}
    res['sal_x_twitter'] = []
    res['sal_y_twitter'] = []
    res['top_left_x_twitter'] = []
    res['top_left_y_twitter'] = []
    res['crop_width_twitter'] = []
    res['crop_height_twitter'] = []

    for fname in file_names:
        fpath = os.path.join(data_dir, fname)

        cmd = f"{str(bin_path)} {str(model_path)} '{Path(fpath).absolute()}' show_all_points"
        output = subprocess.check_output(cmd, shell=True)
        dict_output_img = parse_output(output)

        # The crops are formatted as (x_top_left,y_top_left,width,height)
        # Row order: aspectRatios = [0.56, 1.0, 1.14, 2.0, img_h / img_w]

        x, y = dict_output_img['salient_point'][0]
        res['sal_x_twitter'].append(x)
        res['sal_y_twitter'].append(y)

        x, y, w, h = dict_output_img['crops'][0]
        res['top_left_x_twitter'].append(x)
        res['top_left_y_twitter'].append(y)
        res['crop_width_twitter'].append(w)
        res['crop_height_twitter'].append(h)

    survived = []
    for element in np.asarray(res['top_left_y_twitter']) < 887:
        if element:
            survived.append('top')
        else:
            survived.append('bottom')

    res['survived_member_twitter'] = survived
    return res


def main():
    model = ImageSaliencyModel(crop_binary_path=bin_path, crop_model_path=model_path)
    res = twitter(DATA_DIR, EXP_DF['file_name'])
    df = pd.DataFrame(res, index=EXP_DF['file_name'].values)
    df.to_csv(SAVE_PATH)


if __name__ == '__main__':
    main()
