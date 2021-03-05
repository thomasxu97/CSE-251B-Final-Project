import torch
import torch.nn as nn

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 16, 21, 21]           1,040
#               ReLU-2           [-1, 16, 21, 21]               0
#        BatchNorm2d-3           [-1, 16, 21, 21]              32
#             Conv2d-4           [-1, 32, 10, 10]           8,224
#               ReLU-5           [-1, 32, 10, 10]               0
#        BatchNorm2d-6           [-1, 32, 10, 10]              64
#             Linear-7                  [-1, 256]         819,456
#               ReLU-8                  [-1, 256]               0
#             Linear-9                    [-1, 6]           1,542
# ================================================================
# Total params: 830,358
# Trainable params: 830,358
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.03
# Forward/backward pass size (MB): 0.24
# Params size (MB): 3.17
# Estimated Total Size (MB): 3.43
# ----------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2)
        self.bnd1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.bnd2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(3200, 256)
        self.fc2 = nn.Linear(256, self.action_space)
        

    def forward(self, x):
        # convolution 
        x = self.bnd1(self.relu(self.conv1(x)))
        x = self.bnd2(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  