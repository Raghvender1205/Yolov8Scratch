# YOLOv8 from Scratch
Yolov8 implementation from Scratch

## Implementation
https://www.youtube.com/watch?v=6zQP0L-ph0M&t=1s

### Backbone

`YOLOv8` uses `CSPDarknet53` as the backbone. It contains
1. Conv: `Conv2d` + `BatchNorm2d` + `SiLU`
2. C2f: Cross-stage Partial Bottleneck with 2 convolutions
3. SPPF: Spatial Pyramid Pooling Fast

- `C2f` combines high-level features with contextual information to improve `detection` accuracy.
- `SPPF` processes features at various scales and pool them into a fixed sized feature map.

### Neck
The neck has 
1. Upsample
2. C2f

`Upsample` is nearest-neighbor interpolation with scale_factor = 2

### Head
It consists of 3 modules `bbox` coordinates, `classification` scores, `distribution focal loss` (DFL) 

`DFL` consists the predicted box coordinates as a probability distribution. At inference time, it samples from the distribution 
to get the refined coordinates (x, y, w, h). For example, if the predicted coordinate x in the normal range [0, 1].
1. `DFL` uses 16 bins which are equally spaced in [0, 1], bin length = 1/16.
2. The model outputs 16 numbers which corresponds to probabilities that x falls in these bins, For example [0, 0, ..., 9/10, 1/10].
3. Prediction for x = mean value = 9/10 . 15/16 + 1/10 . 1 = 0.94375
