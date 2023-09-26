# Apparent Skin Color

**Beyond Skin Tone: A Multidimensional Measure of Apparent Skin Color**  
William Thong, Przemyslaw Joniak, Alice Xiang  
ICCV 2023

[[Paper](https://arxiv.org/abs/2309.05148)]

## Skin color score extraction

* Run [extract/predict.py](extract/predict.py) to extract skin color scores (i.e., perceptual lightness L* and hue angle h*)
  * Only PIL, scikit-image and scikit-learn are needed
  * Perceptual lightness L* is associated with skin tone (values below 60 are considered to be dark tones, while values above are light tones)
  * Hue angle h* is associated with skin huw (values below 55 are considered to red hues, while values above are yellow hues)
* We also provide the results of this script in [extract/results](extract/results) for the CelebAMask-HQ, FFHQ and CFD datasets
* When masks are not available, we extract them with [DeepLabV3](https://github.com/royorel/FFHQ-Aging-Dataset#optional-arguments) trained on CelebAMask-HQ

## Experiments

### Saliency-based image cropping

We evaluate the fairness of the open-source image cropping model from Twitter by comparing a pair of facial images.
In an ideal scenario, the cropping model should have an equal preference for both faces. 

Experiments are reported in Section 4.2.1 of the main paper, and are adapted from [[Birhane et al, WACV 2022](https://github.com/vinayprabhu/Saliency_Image_Cropping/tree/main)]. To reproduce them, follow these steps:

* Download the CFD and CFD-INDIA dataset [[CFD](https://www.chicagofaces.org/download/)]
* Clone and install the following repository [[twitter-research/image-crop-analysis](https://github.com/twitter-research/image-crop-analysis)]
* Run [cropping/generate.py](cropping/generate.py) to generate image pairs
  * This will generate 22,500 image pairs, equally distributed wrt self-reported gender and ethnicity labels, from the whole dataset of 739 individuals listed in [[cropping/cfd_meta.csv](cropping/cfd_meta.csv)]
* Run [cropping/predict_twitter.py](cropping/predict_twitter.py) apply the Twitter cropping algorithm on the image pairs

### Face verification

We evaluate the fairness of face verification models.

Experiments are reported in Section 4.2.2 of the main paper.

* Clone the following repository to have access to the models [[serengil/deepface](https://github.com/serengil/deepface)]
* For the dataset, we rely on the LFW version prepared by [scikit-learn](https://scikit-learn.org/)
* Run [face-verification/lfw.py](face-verification/lfw.py) to get the predictions for the ArcFace, FaceNet and Dlib models

###  Skin color causal effect in attribute prediction

We evaluate the causal effect of skin color in several (non)-commercial models.
To achieve this, we modify the skin color of images in CelebAMask-HQ by moving in the latent space that are meaningful to edit the skin tone and the skin hue independently.

* Download the CelebAMask-HQ dataset [[switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)]
* We rely on the [[yuval-alaluf/stylegan3-editing](https://github.com/yuval-alaluf/stylegan3-editing)] to edit images
* We provide the decision boundaries for perceptual lightness and hue angle in the [attribute](attribute) folder, which can be used to edit any images with [InterfaceGAN](https://github.com/yuval-alaluf/stylegan3-editing/tree/main/editing/interfacegan)


## Citation

If you find this repository useful for your research, please consider citing our work:
```
@inproceedings{thong2023skincolor,
  title={Beyond Skin Tone: A Multidimensional Measure of Apparent Skin Color},
  author={Thong, William and Joniak, Przemyslaw and Xiang, Alice},
  booktitle={ICCV},
  year={2023}
}
```
