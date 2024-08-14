# YOLOv8 from Scratch
Yolov8 implementation from Scratch

## Implementation
### Backbone

`YOLOv8` uses `CSPDarknet53` as the backbone. It contains
1. Conv: `Conv2d` + `BatchNorm2d` + `SiLU`
2. C2f: Cross-stage Partial Bottleneck with 2 convolutions
3. SPPF: Spatial Pyramid Pooling Fast

- `C2f` combines high-level features with contextual information to improve `detection` accuracy.
- `SPPF` processes features at various scales and pool them into a fixed sized feature map.