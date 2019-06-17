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
- [file_map](https://github.com/ZJULearning/videoqa/tree/master/dataset/file_map.tsv): contains the Tumblr urls of the videos 
- [QA](https://github.com/ZJULearning/videoqa/tree/master/dataset/QA.tsv): contains the question-answer pairs
- [Split](https://github.com/ZJULearning/videoqa/tree/master/dataset/split): contains the dataset split in the paper

## Methods

### Compared Algorithms
+ [E-SA] (https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14906/14319)
+ [SS-VQA] (https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14906/14319) 
+ Mean-VQA: a designed baseline where imageQA is performed on each frame
### Results

![ex1](https://github.com/ZJULearning/videoqa/tree/master/examples/117791.gif)
- Question: What is a boy combing his hair with?
- Groundtruth: with his fingers
- Prediction: with his hands

![ex2](https://github.com/ZJULearning/videoqa/tree/master/examples/076306.gif)
- Question: What runs up a fence?
- Groundtruth: a cat
- Prediction: a cat

![ex3](https://github.com/ZJULearning/videoqa/tree/master/examples/112935.gif)
- Question: What is a young girl in a car adjusting?
- Groundtruth: her dark glasses
- Prediction: her hair

## Dependency
- [Theano](https://github.com/Theano)
- [Blocks](https://github.com/mila-udem/blocks)
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
