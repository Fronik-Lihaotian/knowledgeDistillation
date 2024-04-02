# knowledgeDistillation

KD implementation practice by pytorch

## Teacher Network pre-training

I use **MobileNetv2** as the teacher network in this project, network architecture followed pytorch official 
implementation. In pre-training stage, I trained MobileNetv2 on **Caltech 256** dataset for **300** epochs to 
get a well-generalizaed backbone. The pre-training results is shown below：

|Network|Dataset|Classes num|Top-1 Accuracy|
|:-----:|:-----:|:-----:|:-----:|
|MobileNetV2|Caltech-256|257|69.1%|

### Training strategy

Learning methods:
- Optimizer: AdamW
- Learning Rate: 0.0001
- LR Scheduler: CosineAnnealingWarmRestarts(T_0=34560, T_mult=2, eta_min=0.000009)
- Training Epochs: 300


Data Agumentation:
- RandomResizedCrop
- RandomHorizontalFlip
- RandAugment
- RandomErasing


