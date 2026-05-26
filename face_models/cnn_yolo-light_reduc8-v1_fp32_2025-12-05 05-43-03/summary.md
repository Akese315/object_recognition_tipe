# Experiment Summary - cnn_yolo-light_reduc8-v1_fp32_2025-12-05 05-43-03

## Model Architecture
```
LightweightYOLO(
  (backbone): Sequential(
    (0): ConvBlock(
      (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ConvBlock(
      (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): ConvBlock(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (head): YOLOHead(
    (detector): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

## Hyperparameters
- **Num Classes**: 1
- **Num Epochs**: 10
- **Batch Size**: 4
- **Learning Rate**: (See Optimizer details)
- **Reduction Factor**: 8
- **Centers**: [[1.         0.58289368]
 [1.         0.83194668]
 [0.99950009 0.41818036]]
- **N_ANCHORS**: 3

## Training Components
- **Loss Function**: YoloLoss(
  (mse): MSELoss()
  (bce): BCEWithLogitsLoss()
)
- **Lambda Coord**: 5
- **Lambda Noobj**: 0.1
- **Lambda Obj**: 20
- **Smooth Factor**: 0
- **Optimizer**: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
)
- **Scheduler**: <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x0000023E0273AD50>
- **Last Loss**: 53.99023386346839
- **Last Coord**: 5.856319307051302
- **Last Obj**: 80.28731173842802
- **Last Noobj**: 2959.4647165889114
- **Last Class**: 4.759623069417804e-06
