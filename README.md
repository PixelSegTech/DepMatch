# DepMatch: Boosting Semi-supervised  Semantic Segmentation by Exploring  Depth Difference Knowledge

Abstract:— Existing semi-supervised semantic segmentation (SSS) methods fail to explore the potential of depth information in unlabeled data, as they suffer from i) inter-class depth similarity, and ii) intra-class depth discrepancy. To address these challenges,  this paper proposes DepMatch, a simple yet effective approach that leverages  depth difference knowledge to guide consistency learning. Specifically, a Class-wise Depth Disparity Perception (CDDP) module is designed to exploit depth difference information, driven by class prediction priors, facilitating  robust feature learning. Depth-feature discrepancy set is first constructed and then reliable pixel pairs are selected for inter-class depth disparity knowledge distillation. Simultaneously, exponential normalization is applied to intra-category depth disparity for suppressing large outlier variations, and an entropy-based adaptive weight is derived to prioritize feature learning of high entropy areas. Moreover, we propose the Uncertain Logit Disparity Regulation (ULDR) module, which leverages the depth variations at class boundaries to promote  the mutual regulation of uncertain pixel logit information, enhancing the model’s spatial understanding.  Experiments on two public benchmarks show that DepMatch  can be seamlessly  incorporated as a plug-and-play plugin into popular SSS frameworks, achieving significant performance improvements across various visual encoders. 

# Pipeline
![alt text](https://github.com/Anonymous-DepMatch/DepMatch/blob/main/Fig/Net.png)


# Installation
> pip install -r requirements.txt

# Datasets
We have demonstrated state-of-the-art experimental performance of our method on Pascal VOC2012 and Cityscapes datasets.
You can download the Pascal VOC2012 on [this](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).

You can download the Cityscapes on [this](https://www.cityscapes-dataset.com/).

# Training 
## How to train on Pascal VOC2012
If training is performed on the 1/2 setting, set the configuration file for the VOC dataset, set the path  for labeled data and the path  for unlabeled data, as well as the corresponding training model parameter storage path. Here is an example shell script to run DepMatch on Pascal VOC2012 :

     CUDA_VISIBLE_DEVICES=0,1,2,3 nohup  python -m torch.distributed.launch --nproc_per_node=4 --master_port=6719   DepMatch.py >VOC_1_2.log &


## How to train on Cityscapes
 If training is performed on the 1/2 setting, set the configuration file for the Cityscapes dataset, set the path  for labeled data and the path  for unlabeled data, as well as the corresponding training model parameter storage path. Here is an example shell script to run DepMatch on Cityscapes :

     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup  python -m torch.distributed.launch --nproc_per_node=8 --master_port=6719   DepMatch.py >Cityscapes_1_2.log &

# Qualitative Results

<img src="https://github.com/Anonymous-DepMatch/DepMatch/blob/main/Fig/Results.png" width="700">



