# Scaling MLPs :fire:
​
![enter image description here](https://lh3.googleusercontent.com/pw/AJFCJaXewwJEp0hSooj47PsD2wedGBqjfTNPr4hnIMfypkTmF6dvVk35OL6wYIDNf-FADkZjOSxs9NYK7uwVPi3zdQcJIQg-jDZJuLTReIErvdG3j4jWkVGCEp0VGvkFNXfRFGQNTQacYUqAfBUWxaj7I9ZB=w2976-h1472-s-no)
## Overview
This repository contains the code accompanying our [paper](https://arxiv.org/abs/2306.13575)  *Scaling MLPs: A Tale of Inductive Bias*. In this work we explore the limits of the *multi-layer perceptron*, or short MLP, when subjected to higher amounts of compute. More precisely, we study architectures with the following block form:
![](https://lh3.googleusercontent.com/pw/AIL4fc_3gvNmHfrvhN38zgU2OMTHqG-4w0zMY6of3S7Gi0EoV498btfYB2H7NnYUlpm8d0Va7COQAigFYZ9BCEI93qIqkV4_CKLKtdED6VQ8p-uJrKb6zD0yRfoe2yaMRdFFZeyPXaiFGWkJEurH-wvNGMY1=w1426-h154-s-no?authuser=0)
​
**Why?** We argue that such an MLP has minimal inductive bias (compared to convolutional networks, vision transformers, MLPMixers etc.) and thus offers an interesting test bed to explore whether simply scaling compute can make even the simplest models work (to some degree). The importance of inductive bias has recently been questioned due to vision transformers and MLPMixers eclipsing the more structured convolutional models on standard benchmarks.
​
Moreover, MLPs still remain to be the main protagonists in ML theory works but surprisingly, very little is known about their empirical performance at scale! We aim to close this gap here and provide the community with very performant MLPs to analyse!
​
## Pretrained Models
​
We publish our models pre-trained on ImageNet21k for various number of epochs at an image resolution of $64\times 64$ [here]([https://drive.google.com/drive/folders/17pbKnQgftxkGW5zZGuUvN1C---DesqOW?usp=sharing](https://drive.google.com/drive/folders/17pbKnQgftxkGW5zZGuUvN1C---DesqOW?usp=sharing)). 
​
Fine-tuning those models for $50$ epochs should give you roughly the following down-stream performances (check *Fine-tuning* section for hyper-parameter details)
|                  | #Params | CIFAR10 | CIFAR100 | STL10 | TinyImageNet | ImageNet1k |
| ---------------- | ------- | ------- | -------- | ----- | ------------ | ---------- |
| **B_6-Wi_512**   | 24M     | 87.6%   | 69.8%    | 79.1% | 52.3%        | 33.1%      |
| **B_12-Wi_512**  | 37M     | 90.8%   | 73.7%    | 82.9% | 58.5%        | 37.1%      |
| **B_6-Wi_1024**  | 74M     | 91.5%   | 76.4%    | 85.0% | 62.7%        | 40.0%      |
| **B_12-Wi_1024** | 124M    | 93.6%   | 79.2%    | 88.9% | 69.9%        | 42.5%      |
​

Make sure that you also download the config.txt file and place in together in the same folder as the corresponding checkpoint.
## Environment Setup
​
For installing the *FFCV* dataloading framework, we refer to the original [repository](https://github.com/libffcv/ffcv). To install the remaining packages, activate the FFCV environment and run 
>`pip install -r requirements.txt`
​
## Creating .beton Files
In order to use the efficiency of MLPs to the fullest, we need a more optimised data loading framework than the standard one provided by *torch*. This is because the data transfer from CPU to GPU otherwise becomes the bottleneck of training, not the gradient computation!! 
To ensure a faster data transfer, we use the *FFCV* framework, which requires converting your dataset first to the **beton** format. This can be achieved by specifying your dataset as a torchvision.dataset object. 

If your dataset is implemented in the torchvision.datasets library, simply add the corresponding lines of code to the `get_dataset` function in `dataset_to_beton.py`. We provide implementations for *CIFAR10* and *CIFAR100*. 

If you have your dataset in the standard hierarchical subfolder structure, i.e. your dataset consists of subfolders each corresponding to a separate class, you can simply specify the `dataset_path` argument in `create_beton` in order to obtain the *.beton* file.
​

Conversion to *.beton* accepts a resolution parameter `res`, specifying the resolution of the images. We recommend using `-- res 64` for very large datasets such as *ImageNet21k* in order to keep the computational requirements manageable for users with less resources.
​
 
Downloading and converting the trainset of CIFAR10 to the *.beton* format can for instance be achieved by running
>`python3 data_utils/dataset_to_beton.py --dataset_name cifar10 --mode train --res 32`

Converting a subfolder-structured dataset can be converted to the *.beton* format at resolution 64 by running
>`python3 data_utils/dataset_to_beton.py --data_path path/to/folders --mode train --res 64`
## Pre-training
​
**ImageNet21k.** Due to legal reasons, we cannot provide the *ImageNet21k* in the .beton format directly. We recommend applying [here](https://www.image-net.org/download.php) to download it but in case you cannot get access, you can use the torrent [here](https://academictorrents.com/details/8ec0d8df0fbb507594557bce993920442f4f6477). Similarly for *ImageNet1k*. Once you have downloaded the dataset, we recommend pre-processing it as detailed in this [repository](https://arxiv.org/abs/2104.10972) to remove faulty images and classes with only very little examples. Then produce the *.beton* as outlined above. 
​
​
**Pre-training.** For pre-training the `B_12-Wi_1024` *BottleneckMLP* on *ImageNet21k* at resolution $64 \times 64$, you can use the following command:
>`python3 train.py --dataset imagenet21 --model BottleneckMLP --architecture B_12-Wi_1024 --batch_size 16384 --resolution 64` 
​

For more specific configurations, we encourage the user to check out all available flags in `train.py`. In case you run into memory issues, try to reduce the batch-size. We remark however that smaller batch sizes tend to lead worse results, check-out our paper where we highlight this effect. During training, the parameters will automatically be saved to the `checkpoints`folder. 
## Fine-tuning
​
You can fine-tune our pre-trained checkpoints or your own using the script `finetune.py`.  For instance, the following command fine-tunes the model specified in the path argument on CIFAR10, provided you have converted the CIFAR10 dataset to the *.beton* format:
> `python3 finetune.py --checkpoint_path path/to/checkpoint --dataset cifar10 --batch_size 2048 --epochs 50 --data_resolution 32 --lr 0.01 --optimizer sgd --smooth 0.3 --no-augment --mixup 0.0 --mode finetune --weight_decay 0.0001`
​

You can also train a linear layer on top by specifying the flag `--mode linear`instead.

