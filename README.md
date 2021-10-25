# Adversarial Batch Normalization (AdvBN)

This repository provides the official PyTorch implementation of the NeurIPS 2021 paper:
> Encoding Robustness to Image Style via Adversarial Feature Perturbations      
> Author: *Manli Shu, Zuxuan Wu, Micah Goldblum, Tom Goldstein*      

> Abstract: Adversarial training is the industry standard for producing models that are robust to small adversarial perturbations.  However, machine learning practitioners need models that are robust to other kinds of changes that occur naturally, such as changes in the style or illumination of input images. 
Such changes in input distribution have been effectively modeled as shifts in the mean and variance of deep image features. 
We adapt adversarial training by directly perturbing feature statistics, rather than image pixels, to produce models that are robust to various unseen distributional shifts. 
We explore the relationship between these perturbations and distributional shifts by visualizing adversarial features.
Our proposed method, Adversarial Batch Normalization (AdvBN), is a single network layer that generates worst-case feature perturbations during training.
By fine-tuning neural networks on adversarial feature distributions, we observe improved robustness of networks to various unseen distributional shifts, including style variations and image corruptions. 
In addition, we show that our proposed adversarial feature perturbation can be complementary to existing image space data augmentation methods, leading to improved performance. 


## Overview
This repository contains implementations for:   
* Training using AdvBN on ImageNet: `train_imagenet.py`    
* Evaluating models on ImageNet and its variants: `test_imagenet.py`    
* Creating the ImageNet-AdvBN test dataset: `make_test_data.py`

## Environment
The code is tested on:    
* pytorch == 1.7.1
* torchvision == 0.8.2
* pillow == 7.1.2   
More dependencies can be found at `./requirements.txt`     

Hardware requirements:     
* The default training setting requires 4 GPUs (with >= 11GB GPU memory each). 


## Training
1. Set the `${IMAGENET}` argument in `./scripts/train_imagenet.sh` to the path to your ImageNet dataset where subdirectory `train` and `val` can be found.
2. Modify the batchsize and other arguments in `./configs/res50_configs.yml`.
3. Run:
    ```
    bash ./scripts/train.sh
    ```
    Pre-trained models are available for [download](https://drive.google.com/drive/folders/1dtL1I244ZXdcWy4wvcOoKP-4PYQ4KCV6?usp=sharing). 

## Testing
### Datasets
* [**ImageNet-C**](https://github.com/hendrycks/robustness): This dataset contains distorted images with 15 categories of common image corruption applied, each with 5 levels of severity. More details can be found in its [official repository](https://github.com/hendrycks/robustness).
* [**ImageNet-Instagram**](https://arxiv.org/abs/1912.13000): it is composed of ImageNet images processed with a total of 20 different Instagram aesthetic image filters. Filters are applied separately, and the dataset contains 20 sub-datasets, each corresponding to one type of image filter. The dataset can be downloaded [here](https://drive.google.com/file/d/1rmOFrwa5kaxjhqw2Kgh-qDaeiFlt-u0B/view?usp=sharing).
* [**ImageNet-Sketch**](https://github.com/HaohanWang/ImageNet-Sketch): a dataset of black and white sketches, including 50,000 images in total, falling into 1,000 ImageNet categories, with 50 images per category. Details concerning the construction of this dataset can be found in its [official repository](https://github.com/HaohanWang/ImageNet-Sketch).     
* [**Stylized-ImageNet**](https://github.com/rgeirhos/Stylized-ImageNet): it consists of style transfered versions of images from the ImageNet dataset. Codes for generating this dataset can be found at [official repository](https://github.com/rgeirhos/Stylized-ImageNet). 


Command line args: 
```
--statedict: path to the pretrained model.
--pathX: path to dataset X, X = "I"(imagenet)/"S"(stylized imagenet)/"K(ImageNet-Sketch)/"C"(ImageNet-C)/"T"(ImageNet-Ins.).
--set: which dataset to test (default='ISKCT').
```
1. Specify the paths to each dataset in `./scripts/test.sh`.
2. Specify the paths to the model checkpoint to be tested. 
3. Run: 
```
bash ./scrips/test.sh
```


## Creating ImageNet-AdvBN dataset
1. Download the pretrained [VGG-19 network](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing) (credit to [this repo](https://drive.google.com/file/d/1UDcA2zdncYJRgDSsP9SupI-e61huxf9P/view?usp=sharing)) and our ImageNet trained [decoder](https://drive.google.com/file/d/1UDcA2zdncYJRgDSsP9SupI-e61huxf9P/view?usp=sharing).
2. Modify related paths and configurations in `./scripts/make_data.sh`.
3. Run:
    ```
    bash ./scripts/make_data.sh
    ```
    Our ImageNet-AdvBN dataset is also available for [download](https://drive.google.com/file/d/1fHOw5_uHJ6Yebp5twAo7hut9MmYSd2vC/view?usp=sharing). 


## Performance

|        |ImageNet (acc.)|ImageNet-C (mCE.) ↓|ImageNet-Instagram (acc.)|ImageNet-Sketch (acc.)|Stylized-ImageNet (acc.)|       
| ---    |   :---:       |   :---:         |       :---:             |       :---:             |     :---:              | 
|Standard Training|    76.1%      |    76.7%        |     67.2%               |      24.1%              |      7.4%              |
|AdvBN   |    77.0%      |    72.7%        |     69.5%               |      27.9%              |      11.9%             |     


## Citation
If you find the code or our method useful, please consider citing: 
```
@InProceedings{shu2021advbn,
    author    = {Manli Shu and Zuxuan Wu and Micah Goldblum and Tom Goldstein},
    title     = {Encoding Robustness to Image Style via Adversarial Feature Perturbations},
    booktitle = {NeurIPS},
    year      = {2021},
}
```
