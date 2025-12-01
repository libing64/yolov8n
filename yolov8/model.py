import torch
import torch.nn as nn
from .modules import Conv, C2f, SPPF, Detect, Concat

class YOLOv8n(nn.Module):
    """YOLOv8 Nano model architecture."""
    def __init__(self, nc=80):
        super().__init__()
        self.nc = nc
        
        # Backbone
        # 0: P1/2 
        self.b0 = Conv(3, 16, 3, 2)
        # 1: P2/4
        self.b1 = Conv(16, 32, 3, 2)
        # 2: C2f with 1 repeat (n=1 repeats * depth_multiple 0.33 -> 1? No, wait. 3*0.33=1)
        # Config says: [2, 3, C2f, [128, True]] (channels are base 128, so 128*0.25=32)
        self.b2 = C2f(32, 32, n=1, shortcut=True)
        # 3: P3/8
        self.b3 = Conv(32, 64, 3, 2)
        # 4: C2f n=2 (6*0.33=2)
        self.b4 = C2f(64, 64, n=2, shortcut=True)
        # 5: P4/16
        self.b5 = Conv(64, 128, 3, 2)
        # 6: C2f n=2 (6*0.33=2)
        self.b6 = C2f(128, 128, n=2, shortcut=True)
        # 7: P5/32
        self.b7 = Conv(128, 256, 3, 2)
        # 8: C2f n=1 (3*0.33=1)
        self.b8 = C2f(256, 256, n=1, shortcut=True)
        # 9: SPPF
        self.b9 = SPPF(256, 256, 5)

        # Head (PANet)
        # 10: Upsample
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        # 11: Concat (b9 + b6) -> C2f
        # Input to C2f is 256 + 128 = 384. Output 128.
        self.h1 = C2f(256 + 128, 128, n=1, shortcut=False)
        
        # 13: Upsample
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        # 14: Concat (h1 + b4) -> C2f
        # Input 128 + 64 = 192. Output 64.
        self.h2 = C2f(128 + 64, 64, n=1, shortcut=False) # P3/8 (small)

        # 16: Conv
        self.d1 = Conv(64, 64, 3, 2)
        # 17: Concat (d1 + h1) -> C2f
        # Input 64 + 128 = 192. Output 128.
        self.h3 = C2f(64 + 128, 128, n=1, shortcut=False) # P4/16 (medium)

        # 19: Conv
        self.d2 = Conv(128, 128, 3, 2)
        # 20: Concat (d2 + b9) -> C2f
        # Input 128 + 256 = 384. Output 256.
        self.h4 = C2f(128 + 256, 256, n=1, shortcut=False) # P5/32 (large)

        # 22: Detect
        self.detect = Detect(nc, ch=(64, 128, 256))

    def forward(self, x):
        # Backbone
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3) # P3
        x5 = self.b5(x4)
        x6 = self.b6(x5) # P4
        x7 = self.b7(x6)
        x8 = self.b8(x7)
        x9 = self.b9(x8) # P5

        # Head
        x10 = self.up1(x9)
        # concat x10 and x6
        x11 = torch.cat([x10, x6], dim=1)
        x11 = self.h1(x11) # -> x12

        x13 = self.up2(x11)
        # concat x13 and x4
        x14 = torch.cat([x13, x4], dim=1)
        x15 = self.h2(x14) # P3 output

        x16 = self.d1(x15)
        # concat x16 and x11
        x17 = torch.cat([x16, x11], dim=1)
        x18 = self.h3(x17) # P4 output

        x19 = self.d2(x18)
        # concat x19 and x9
        x20 = torch.cat([x19, x9], dim=1)
        x21 = self.h4(x20) # P5 output

        return self.detect([x15, x18, x21])

if __name__ == '__main__':
    model = YOLOv8n(nc=80)
    input_tensor = torch.randn(1, 3, 640, 640)
    output = model(input_tensor)
    print(f"Output shape: {[o.shape for o in output] if isinstance(output, tuple) else output.shape}")

