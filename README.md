# knowledgeDistillation

KD implementation practice by pytorch

## Teacher Network pre-training

I use **MobileNetv2** as the teacher network in this project, network architecture is copied from [here](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test6_mobilenet/model_v2.py), thanks! 
In pre-training stage, I trained MobileNetv2 on **Caltech 256** dataset for **300** epochs to get a 
well-generalizaed backbone. The pre-training results are shown below：

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


Data Agumentation:
- RandomResizedCrop
- RandomHorizontalFlip
- RandAugment
- RandomErasing

## Student network stage

I use modified MobileNetv2 with innovative inverted residual block (called overlapping group convolution 
block) as the student network, which is designed based on my personal MSc final project in the University of 
Birmingham. The entire stage will be: 

1. Fine tuning the teacher network based on the backbone gained from above stage. 
2. Training student model with knowledge distillation. 

### Fine tuning teacher model

I loaded the backbone model from the path `./teacher_weights/MobileNetv2.pth` and frozen the weights 
except the classifier layer. Then fine tuning the teacher network on **Caltech-101**, which is our domain 
dataset in this project, for **30 (or 60) epochs** to obtain the teacher network. The fine tuning results are 
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

Data Agumentation:
- RandomResizedCrop
- RandomHorizontalFlip
- RandAugment
- RandomErasing

### Training student network

Here I use the medium size of modified MobileNetV2, called MobileNets-M, as the student network, which 
the `expansion_rate = 4`. Kullback-Leibler divergence is applied to provide distillation loss, the loss function 
can be discribed as:

$$D_{KL}(P_t||P_s) = -\sum (P_t\log P_s-P_t\log P_t),$$

where $P_t$ and $P_s$ individually represent the distributions from teacher network and student network. 
The temperature parameter equals to **5.0** in here. 

The total loss combines the hard loss (Cross-Entropy loss) and the soft loss (Distillation loss), which can be 
discribed as:

$$ Loss_{total} = \alpha Loss_{hard} + (1- \alpha) Loss_{soft},$$

where $\alpha$ is a hyperparameter for adjusting the weight between two loss, which is **0.3** in here. 

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

Data Agumentation:
- RandomResizedCrop
- RandomHorizontalFlip
- RandAugment
- RandomErasing

## Comparison Group

### With & without knowledge distillation 

I compared the performance difference between with and without distillation, the training recipe is same 
with the student stage. The comparison is shown below:

|Network|Dataset|Classes num|Epochs|Top-1 Accuracy|with KD|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|MobileNets-M|Caltech-101|101|120|**84.5%**|&#10003;|
|MobileNets-M|Caltech-101|101|120|82.8%|&#10008;|

Clearly, without the distillation, the top-1 accuracy is 1.7% lower than with distillation, which showed the 
effectiveness of knowledge distillation. 

### Influence of LR scheduler

In here, I mainly discuss the impact from *learning rate scheduler*, I'm planing to make a series of experiments
to explore the performance difference between with and without LR scheduler.

I firstly add the cosine lr scheduler in the student training procedure, the scheduler can be described as: 

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

From the results, applying cosine LR scheduler cannot improve the performance of network. 

### Influence of Data Augmentation

TBD

### \alpha & temperature 

TBD

## Acknowledgement

The code is written with reference to this [repository](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test6_mobilenet). 
And I also learned a lot from this [blog](https://blog.csdn.net/weixin_44911037/article/details/123134947). Many thanks!