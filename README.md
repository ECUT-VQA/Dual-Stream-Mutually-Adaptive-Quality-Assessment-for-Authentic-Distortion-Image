<h1 align="center">Welcome to ECUT_VQA 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0-blue.svg?cacheSeconds=2592000" />
</p>

Welcome to ECUT_VQA (Visual Quality Assessment group of East China University of Technology)

<div align="center">
    <b><a href="#中文说明">中文</a> | <a href="#english-description">English</a></b>
</div>

This is the PyTorch implementation of our paper accepted to Journal of Visual Communication and Image Representation. Thank you for your citation (Dual-stream mutually adaptive quality assessment for authentic 
distortion image. J. Vis. Commun. Image R. 102 (2024) 104216. [https: //doi. org/10.1016/j.jvcir.2024.104216].

<!-- 中文内容 -->
## <a name="中文说明"></a>中文说明
> 展示课题组论文成果

基于pytorch开发的无参考图像质量评价算法。评价效果图请[点击这里](#效果图)查看。  


## 目录

1. [环境配置](#环境配置)
2. [开发指南](#开发指南)
3. [效果图](#效果图)

## 环境配置

1. 首先确保已经安装和配置好的python版本>=3.8
2. pytorch>=1.13.1、
3. cuda>=11.7
4. cudnn>=8.0 
5. 从仓库下载requements.txt,并根据这个环境下载所有需要的包
6. 根据提示，下载代码，训练自己的模型，或者加载我们提供的模型参数


## 开发指南
1. 下载对应的公开数据集和标签文件
2. 首先训练自己的双流网络或者提取自己的双流特征
3. 根据我们的实验，可以用于基于对比学习和基于vae的图像质量评价算法中


## 效果图
![](./image/论文.png)
![](./image/添加.png)
![](./image/管理.png)

<!-- 英文内容 -->
## <a name="english-description"></a>English Description
>Display our research team's paper achievements

  A no-reference  image quality assessment algorithm developed based on PyTorch. Please [click here] (#效果图) to view the evaluation rendering.
1. [Environment Configuration] (# Environment Configuration)
2. [Development Guide] (# Development Guide)
3. [Renderings] (# Renderings)

## Environment Configuration
1. First, ensure that the installed and configured Python version is>=3.8
2. pytorch>=1.13.1
3. cuda>=11.7
4. cudnn>=8.0
5. Download requests.txt from the repository and download all required packages according to this environment
6. According to the prompts, download the code, train your own model, or load the model parameters we provide

## Development Guide
1. Download the corresponding public dataset and label files
2. First, train your own dual-stream network or extract your own dual-stream features
3. According to our experiment, it can be used in image quality assessment algorithms based on contrastive learning and VAE

## Renderings

## Author

👤 **ECUT_VQA**

## Show your support
For more details, please wait for further organization of the code

Give a ⭐️ if this project helped you and quote our paper!
