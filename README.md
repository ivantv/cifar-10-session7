## CIFAR-10: C1–C2–C3–C4 CNN

### 1. Data analysis and exploration
- **Dataset**: CIFAR-10 (50k train, 10k test; 10 classes).
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

### 4. Model accuracy
- **Training setup**: SGD (lr=0.01, momentum=0.9), batch size 128, standard augmentations.
- **Observed test accuracy (example run)**:
  - Epoch 0: 58.80%
  - Epoch 1: 66.04%
  - Epoch 2: 73.66%
  - Epoch 3: 77.30%

Re-run the notebook to reproduce or extend these results; accuracy may vary with training duration and augmentation strength.


### 5. Albumentations transforms
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

