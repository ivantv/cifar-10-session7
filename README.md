## CIFAR-10: C1–C2–C3–C4 CNN

### 1. Data analysis and exploration
- **Dataset**: CIFAR-10 (50k train, 10k test; 10 classes).

{'airplane': 5000, 'automobile': 5000, 'bird': 5000, 'cat': 5000, 'deer': 5000, 'dog': 5000, 'frog': 5000, 'horse': 5000, 'ship': 5000, 'truck': 5000}


- **Class balance**: Uniform (5,000 images per class in training set).
- **Visualization**: Sample grids are displayed to inspect image variety and labels.

### 2. Mean and Std deviation computation and updates into transforms
- **Computed on training set** (channel-wise, RGB):
  - **Mean**: [0.4914, 0.4822, 0.4465]
  - **Std**: [0.2470, 0.2435, 0.2616]
- **Usage**:
  - Applied in torchvision and Albumentations `Normalize` for train/test.
  - Used for de-normalizing images when visualizing augmentations.

### 3. Defining the neural network (high parameter count)
- **Architecture**: C1 → C2 → C3 → C4 (with stride-2 in C4) → GAP → 1×1 Conv classifier → log_softmax.
- **Parameters**: ~7,665,376 (very high for 32×32 inputs). Much higher that th 200K asked in the assignment
- **Blocks**:
  - **C1**: Two 3×3 conv layers, 32 channels each, with BatchNorm, ReLU, Dropout(0.05).
  - **C2**: Two 3×3 conv layers, 64 channels each, with BatchNorm, ReLU, Dropout(0.05).
  - **C3**: Two 3×3 conv layers, 128 channels each, with BatchNorm, ReLU, Dropout(0.05).
  - **C4**: Sequence to expand receptive field efficiently:
    - 3×3 (s=1), 256 channels
    - 3×3 (s=2), 256 channels  ← downsampling instead of MaxPool with stride of 2
    - 7×7 (s=1), 256 channels
    - 5×5 (s=1), 256 channels
    - 5×5 (s=1), 256 channels
  - **Head**: Global Average Pooling → 1×1 Conv to 10 classes.



Re-run the notebook to reproduce or extend these results; accuracy may vary with training duration and augmentation strength.


### 4. Albumentations transforms
- **Training transforms** (applied on raw uint8 HWC images before tensor conversion):
  - HorizontalFlip(p=0.5)
  - ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5)
  - CoarseDropout(p=0.5) with up to 1 hole of size 16×16, fill set to dataset mean (RGB scaled to 0–255)
  - Normalize(mean=mean, std=std)
  - ToTensorV2()

- **Test transforms**:
  - Normalize(mean=mean, std=std)
  - ToTensorV2()

- **Integration**:
  - A lightweight wrapper `AlbumentationsCIFAR` subclasses `torchvision.datasets.CIFAR10` and applies the Albumentations pipeline inside `__getitem__`.
  - Data loaders use these wrapped datasets with batch size 128; training loader shuffles, test loader does not.

### 5. Model accuracy
- **Training setup**: SGD (lr=0.01, momentum=0.9), batch size 128, standard augmentations.
- **Observed test accuracy (example run)**:

Model acheived highest accuracy of 87.94% at epoch 15. The model looks to be learning very slowly stopped the run.


- **Raw Logs**:

EPOCH: 0
Loss=1.0834534168243408 Batch_id=390 Accuracy=47.79: 100%|██████████| 391/391 [03:38<00:00,  1.79it/s]

Test set: Average loss: 1.1779, Accuracy: 5880/10000 (58.80%)

EPOCH: 1
Loss=0.8317362666130066 Batch_id=390 Accuracy=66.25: 100%|██████████| 391/391 [03:34<00:00,  1.82it/s]

Test set: Average loss: 0.9777, Accuracy: 6604/10000 (66.04%)

EPOCH: 2
Loss=0.6390466690063477 Batch_id=390 Accuracy=73.15: 100%|██████████| 391/391 [03:35<00:00,  1.81it/s]

Test set: Average loss: 0.7505, Accuracy: 7366/10000 (73.66%)

EPOCH: 3
Loss=0.6605027318000793 Batch_id=390 Accuracy=77.55: 100%|██████████| 391/391 [03:34<00:00,  1.82it/s]

Test set: Average loss: 0.6522, Accuracy: 7730/10000 (77.30%)

EPOCH: 4
Loss=0.5114954113960266 Batch_id=390 Accuracy=80.18: 100%|██████████| 391/391 [03:35<00:00,  1.82it/s]

Test set: Average loss: 0.6335, Accuracy: 7900/10000 (79.00%)

EPOCH: 5
Loss=0.3955336809158325 Batch_id=390 Accuracy=81.98: 100%|██████████| 391/391 [03:35<00:00,  1.82it/s]

Test set: Average loss: 0.5577, Accuracy: 8125/10000 (81.25%)

EPOCH: 6
Loss=0.6212559938430786 Batch_id=390 Accuracy=83.62: 100%|██████████| 391/391 [03:33<00:00,  1.83it/s]

Test set: Average loss: 0.5432, Accuracy: 8254/10000 (82.54%)

EPOCH: 7
Loss=0.37730517983436584 Batch_id=390 Accuracy=85.00: 100%|██████████| 391/391 [03:32<00:00,  1.84it/s]

Test set: Average loss: 0.4903, Accuracy: 8319/10000 (83.19%)

EPOCH: 8
Loss=0.3767012357711792 Batch_id=390 Accuracy=86.12: 100%|██████████| 391/391 [03:32<00:00,  1.84it/s]

Test set: Average loss: 0.5554, Accuracy: 8173/10000 (81.73%)

EPOCH: 9
Loss=0.6129117608070374 Batch_id=390 Accuracy=86.99: 100%|██████████| 391/391 [03:33<00:00,  1.83it/s]

Test set: Average loss: 0.5250, Accuracy: 8248/10000 (82.48%)

EPOCH: 10
Loss=0.36804574728012085 Batch_id=390 Accuracy=87.79: 100%|██████████| 391/391 [03:33<00:00,  1.83it/s]

Test set: Average loss: 0.4105, Accuracy: 8649/10000 (86.49%)

EPOCH: 11
Loss=0.3752756118774414 Batch_id=390 Accuracy=88.57: 100%|██████████| 391/391 [03:33<00:00,  1.83it/s]

Test set: Average loss: 0.4461, Accuracy: 8536/10000 (85.36%)

EPOCH: 12
Loss=0.24930736422538757 Batch_id=390 Accuracy=89.34: 100%|██████████| 391/391 [03:33<00:00,  1.84it/s]

Test set: Average loss: 0.4058, Accuracy: 8658/10000 (86.58%)

EPOCH: 13
Loss=0.4889381527900696 Batch_id=390 Accuracy=89.92: 100%|██████████| 391/391 [03:33<00:00,  1.83it/s]

Test set: Average loss: 0.3501, Accuracy: 8756/10000 (87.56%)

EPOCH: 14
Loss=0.33045631647109985 Batch_id=390 Accuracy=90.32: 100%|██████████| 391/391 [03:33<00:00,  1.83it/s]

Test set: Average loss: 0.3812, Accuracy: 8725/10000 (87.25%)

EPOCH: 15
Loss=0.24605686962604523 Batch_id=390 Accuracy=90.79: 100%|██████████| 391/391 [03:33<00:00,  1.83it/s]

Test set: Average loss: 0.3817, Accuracy: 8794/10000 (87.94%)