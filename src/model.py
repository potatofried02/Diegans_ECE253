import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, imgH, nc, n_class, nh=256):
        super(CRNN, self).__init__()
        
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        # Backbone: VGG-style 
        # input: [batch, nc, imgH, W]
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 64x16xW/2

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 128x8xW/4

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 256x4xW/4

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 512x2xW/4

            # map height from 2 to 1
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True) 
            # output: 512x1xW/4
        )

        self.rnn = nn.LSTM(512, nh, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(nh * 2, n_class)

    def forward(self, x):
        # conv features
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # [b, c, w]
        conv = conv.permute(2, 0, 1) # [w, b, c] -> [seq_len, batch, feature]

        # rnn
        output, _ = self.rnn(conv)
        
        # fc
        T, b, h = output.size()
        t_rec = output.view(T * b, h)
        output = self.fc(t_rec)
        output = output.view(T, b, -1)

        return F.log_softmax(output, dim=2)

# Quick test
imgH = 32
nc = 1
n_class = 80 # 79 chars + blank
bs = 16
W = 180

model = CRNN(imgH, nc, n_class)
print(model)

input_tensor = torch.randn(bs, nc, imgH, W)
out = model(input_tensor)

print(f"Input: {input_tensor.shape}")
print(f"Output: {out.shape}") # Expect: [W/4, bs, n_class]