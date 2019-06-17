# Unifying the Video and Question Attentions for Open-Ended Video Question Answering

Table of Contents
=================
<!--ts-->
* [Introduction](#introduction)
* [Datasets](#datasets)
* [Methods](#methods)
	 * [Compared Algorithms](#compared-algorithms)
	 * [Results](#results)
* [Dependency](#dependency)
* [Usage](#usage)
* [Reference](#reference)
* [License](#license)
<!--te-->

## Introduction
videoqa is the dataset and the algorithms used in [**Unifying the Video and Question Attentions for Open-Ended Video Question Answering**](https://ieeexplore.ieee.org/abstract/document/8017608/)

## Datasets
- [[./dataset/file_map.tsv][file_map]]: contains the Tumblr urls of the videos 
- [[./dataset/QA.tsv][QA]]: contains the question-answer pairs
- [[./dataset/split][Split]]: contains the dataset split in the paper

## Methods

### Compared Algorithms
+ [E-SA] (https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14906/14319)
+ [SS-VQA] (https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14906/14319) 
+ Mean-VQA: a designed baseline where imageQA is performed on each frame
### Results

[[./examples/117791.gif]] 
- Question: What is a boy combing his hair with?
- Groundtruth: with his fingers
- Prediction: with his hands

[[./examples/076306.gif]]
- Question: What runs up a fence?
- Groundtruth: a cat
- Prediction: a cat

.[[/examples/112935.gif]]
- Question: What is a young girl in a car adjusting?
- Groundtruth: her dark glasses
- Prediction: her hair

## Dependency
- [[https://github.com/Theano][Theano]]
- [[https://github.com/mila-udem/blocks][Blocks]]
- Python >= 3.4
## Usage
``` python main.py ```
## Reference
If you use the code or our dataset, please cite our paper 

@article{xue2017unifying,

  title={Unifying the Video and Question Attentions for Open-Ended Video Question Answering},

  author={Xue, Hongyang and Zhao, Zhou and Cai, Deng},

  journal={IEEE Transactions on Image Processing},

  year={2017},

  publisher={IEEE}

}
