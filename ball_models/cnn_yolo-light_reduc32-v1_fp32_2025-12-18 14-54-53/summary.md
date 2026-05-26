# Experiment Summary - cnn_yolo-light_reduc32-v1_fp32_2025-12-18 14-54-53

## Model Architecture
```
LightweightYOLO(
  (backbone): Sequential(
    (0): ConvBlock(
      (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ConvBlock(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): ConvBlock(
      (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): ConvBlock(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): ConvBlock(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (head): YOLOHead(
    (detector): Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

## Hyperparameters
- **Num Classes**: 1
- **Num Epochs**: 50
- **Batch Size**: 16
- **Learning Rate**: (See Optimizer details)
- **Reduction Factor**: 1
- **Centers**: [[0.06765915 0.05136196]
 [0.28178137 0.32423383]
 [0.16482957 0.13552239]]
- **N_ANCHORS**: 3

## Training Components
- **Loss Function**: YoloLoss(
  (mse): MSELoss()
  (bce): BCEWithLogitsLoss()
)
- **Lambda Coord**: 10.0
- **Lambda Noobj**: 0.1
- **Lambda Obj**: 1.0
- **Smooth Factor**: 0.0
- **Optimizer**: AdamW (
Parameter Group 0
    amsgrad: False
    base_momentum: 0.85
    betas: (0.9299906789644576, 0.999)
    capturable: False
    decoupled_weight_decay: True
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    initial_lr: 0.0002
    lr: 0.0011604474097060297
    max_lr: 0.005
    max_momentum: 0.95
    maximize: False
    min_lr: 2.0000000000000002e-07
    weight_decay: 0.0005
)
- **Scheduler**: <torch.optim.lr_scheduler.OneCycleLR object at 0x0000028EE7EB27B0>
- **Last Loss**: 3.576888224657844
- **Last Coord**: 0.1395084261894226
- **Last Obj**: 1.5384879112243652
- **Last Noobj**: 6.430543899536133
- **Last Class**: 0.0002617440477479249
- **Last IoU**: 0.4309459714328541
- **Last Avg Obj**: 0.4164787425714381
