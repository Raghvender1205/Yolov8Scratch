import torch
import torch.nn as nn

from backbone import Backbone
from neck import Neck
from head import Head


class CustomYolo(nn.Module):
    def __init__(self, version):
        super().__init__()

        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = Head(version=version)

    def forward(self, x):
        x = self.backbone(x)  # return out1, out2, out3
        x = self.neck(x[0], x[1], x[2])  # return out_1, out_2, out_3

        return self.head(list(x))


if __name__ == "__main__":
    model = CustomYolo(version="n")
    print(f"{sum(p.numel() for p in model.parameters())/1e6} million parameters")
    print(model)
