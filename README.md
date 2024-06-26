# knowledgeDistillation

KD implementation practice based on PyTorch

## Teacher Network pre-training

I use **MobileNetv2** as the teacher network in this project, network architecture is copied from [here](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test6_mobilenet/model_v2.py), thanks! In the pre-training stage, I trained MobileNetv2 on the **Caltech256** dataset for **300** epochs to get a well-generalized backbone. The pre-training results are shown below：

|Network|Dataset|Classes num|Top-1 Accuracy|
|:-----:|:-----:|:-----:|:-----:|
|MobileNetV2|Caltech-256|257|69.1%|

#### Training strategy

Learning methods:
- Optimizer: AdamW
- Learning Rate: 0.0001
- LR Scheduler: CosineAnnealingWarmRestarts(T_0=34560, T_mult=2, eta_min=0.000009)
- Training Epochs: 300
- Loss: Cross-Entropy


Data Augmentation:
- RandomResizedCrop
- RandomHorizontalFlip
- RandAugment
- RandomErasing

## Student network stage

I use modified MobileNetv2 with the innovative inverted residual block (called overlapping group convolution 
block) as the student network, which is designed based on my personal MSc final project at the University of 
Birmingham, you can find the code and the report from this [repository](https://github.com/Fronik-Lihaotian/Channel-overlapping-group-convolution). The entire stage will be: 

1. Fine-tuning the teacher network based on the backbone gained from the above stage. 
2. Training student model with knowledge distillation. 

### Fine-tuning teacher model

I loaded the backbone model from the path `./teacher_weights/MobileNetv2.pth` and froze the weights 
except the classifier layer. Then fine-tune the teacher network on **Caltech-101**, which is our domain 
dataset in this project, for **30 (or 60) epochs** to obtain the teacher network. The fine-tuning results are 
shown below:

|Network|Dataset|Classes num|Epochs|Top-1 Accuracy|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|MobileNetV2_fine_tuned|Caltech-101|101|30|84.4%|

#### Training strategy

Learning methods:
- Optimizer: AdamW
- Learning Rate: 0.0001
- LR Scheduler: None
- Training Epochs: 30
- Loss: Cross-Entropy

Data Augmentation:
- RandomResizedCrop
- RandomHorizontalFlip
- RandAugment
- RandomErasing

### Training student network

Here I use the medium size of modified MobileNetV2, called MobileNets-M, as the student network, with the `expansion_rate=4`. Kullback-Leibler divergence is applied to provide distillation loss, the loss function can be described as:

$$D_{KL}(P_t||P_s) = -\sum (P_t\log P_s-P_t\log P_t),$$

where $P_t$ and $P_s$ individually represent the distributions from the teacher network and student network. 
The temperature parameter equals **5.0** here. 

The total loss combines the hard loss (Cross-Entropy loss) and the soft loss (Distillation loss), which can be described as:

$$ Loss_{total} = \alpha Loss_{hard} + (1- \alpha) Loss_{soft},$$

where $\alpha$ is a hyperparameter for adjusting the weight between two losses, which is **0.3** here. 

The student network will be trained on Caltech-101 for 120 epochs, the results are shown below:

|Network|Dataset|Classes num|Epochs|Top-1 Accuracy|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|MobileNets-M|Caltech-101|101|120|84.5%|

#### Training strategy

Learning methods:
- Optimizer: AdamW
- Learning Rate: 0.0001
- LR Scheduler: None
- Training Epochs: 120
- Loss: Cross-Entropy + KL Divergence 
- Temperature: 5.0
- $\alpha$: 0.3

Data Augmentation:
- RandomResizedCrop
- RandomHorizontalFlip
- RandAugment
- RandomErasing

## Comparison Group

### With & without KnowledgeDistillation 

I compared the performance difference between with and without distillation, the training recipe is the same as the student stage. The comparison is shown below:

|Network|Dataset|Classes num|Epochs|Top-1 Accuracy|with KD|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|MobileNets-M|Caltech-101|101|120|**84.5%**|&#10003;|
|MobileNets-M|Caltech-101|101|120|82.8%|&#10008;|

Clearly, without the distillation, the top-1 accuracy is 1.7% lower than with distillation, which showed the 
effectiveness of knowledge distillation. 

### Influence of LR scheduler

Here, I mainly discuss the impact of the *learning rate scheduler*, I'm planning to make a series of experiments
to explore the performance difference between with and without LR scheduler.

I firstly added the cosine lr scheduler in the student training procedure, the scheduler can be described as: 

$$factor = (1 + \cos(iteration * \pi / (epochs * iterations))) / 2) \cdot (1 - 0.1) + 0.1$$

$$lr = lr_{initial}*factor$$

The change of factor is shown in the picture below (epochs = 120, range: 1-0.1):

<div align="center">
<img src=https://github.com/Fronik-Lihaotian/knowledgeDistillation/blob/main/imgs/lr_scheduler.png width=50%/>
</div>

The results are shown below:

|Network|Dataset|Top-1 Accuracy|with KD|with scheduler|initial lr|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|MobileNets-M|Caltech-101|**84.5%**|&#10003;|&#10008;|0.0001|
|MobileNets-M|Caltech-101|82.8%|&#10008;|&#10008;|0.0001|
|MobileNets-M|Caltech-101|79.6%|&#10003;|&#10003;|0.0001|
|MobileNets-M|Caltech-101|79.9%|&#10008;|&#10003;|0.0001|
|MobileNets-M|Caltech-101|**86.5%**|&#10003;|&#10008;|0.0002|
|MobileNets-M|Caltech-101|86.0%|&#10008;|&#10008;|0.0002|
|MobileNets-M|Caltech-101|85.7%|&#10003;|&#10003;|0.0002|
|MobileNets-M|Caltech-101|84.6%|&#10008;|&#10003;|0.0002|

From the results, applying the cosine LR scheduler cannot improve the performance of the network. 

### Influence of Data Augmentation

Here, I made the experiments to discuss the performance impact of data augmentations, I fixed the value of the hyperparameter as:

- Temperature: 3.0
- $\alpha$: 0.3
- epochs: 120
- Lr: 0.0002

The results are shown in the table below:

|Network|Random ResizedCrop|Random HorizontalFlip|Rand Augment|Random Erasing|acc|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|MobileNets-M|&#10003;|&#10003;|&#10003;|&#10003;|**87.6%**|
|MobileNets-M|&#10003;|&#10003;|&#10008;|&#10003;|85.4%|
|MobileNets-M|&#10003;|&#10003;|&#10003;|&#10008;|86.8%|
|MobileNets-M|&#10003;|&#10003;|&#10008;|&#10008;|87.4%|

From the results, it's clear that strong data augmentation can improve network performance, however, the network seems to perform better without Random HorizontalFlip and Rand Erasing. 

### $\alpha$ & temperature 

#### Temperature hyperparameter:

In this section, I mainly discuss the impact of the temperature hyperparameter. The temperature is used to affect the distribution from the network, specifically, the relations between the output distribution and the temperature can be described as:

$$p_i=\frac{exp(z_i/T)}{\sum_{j}exp(z_j/T)},$$

where $z_i$ represents the logits from the network, and $p_i$ represents the final distributions.

In the previous experiments, I predefined the temperature hyperparameter as 5. However, it may not be the best training recipe, therefore, the impact of this hyperparameter is worth discussing. 

From the results of the experiments above, it's obvious that the performance will be better when the learning rate equals 0.0002, so, I fixed the learning rate to 0.0002 **without** the LR scheduler in the following experiments, and only the temperature is the changeable hyperparameter. The results are shown below:

|Network|Dataset|Top-1 Accuracy|epochs|temperature|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|MobileNets-M|Caltech-101|86.1%|120|10.0|
|MobileNets-M|Caltech-101|86.3%|120|7.0|
|MobileNets-M|Caltech-101|86.5%|120|5.0|
|MobileNets-M|Caltech-101|**87.6%**|120|3.0|
|MobileNets-M|Caltech-101|85.5%|120|1.0|
|MobileNets-M|Caltech-101|84.3%|120|0.5|

From the results, the highest top-1 accuracy reached 87.6% when temperature equals 3. 

#### $\alpha$ hyperparameter: 

Here, I mainly discuss the impact of the parameter $\alpha$, from the formula below, the $\alpha$ is used to control the weights of KD loss compared and Cross-Entropy loss.

$$ Loss_{total} = \alpha Loss_{hard} + (1- \alpha) Loss_{soft},$$

In the previous experiments, the $\alpha$ equals 0.3, but is this the best setting? and how much performance difference can be made by changing the $\alpha$. I fixed the temperature as 3.0 since it's got the best results, and everything else remains the same. The results are shown below:

|Network|Dataset|Top-1 Accuracy|$\alpha$|
|:-----:|:-----:|:-----:|:-----:|
|MobileNets-M|Caltech-101|85.1%|1|
|MobileNets-M|Caltech-101|85.5%|0.7|
|MobileNets-M|Caltech-101|86.2%|0.5|
|MobileNets-M|Caltech-101|**87.6%**|0.3|
|MobileNets-M|Caltech-101|86.2%|0|

From the results, 0.3 is the ideal setting of $\alpha$ in this project, and the Top-1 accuracy is 87.6% which is the best performance within these results. Another interesting thing is that, when the $\alpha$ equaled 1, which means the hard loss was completely ignored, the network can still reach 85.1% Top-1 accuracy which is even **higher than the fine-tuned teacher network (84.4%)**. But why? IDK :confused: yet. :laughing:


## Acknowledgment

The code is written regarding this [repository](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test6_mobilenet). 
And I also learned a lot from this [blog](https://blog.csdn.net/weixin_44911037/article/details/123134947). Many thanks!