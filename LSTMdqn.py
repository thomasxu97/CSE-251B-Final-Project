import torch
import torch.nn as nn
from torch.autograd import Variable

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

LSTM_MEMORY = 128

class LSTMDQN(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4, padding=2)
        self.bnd1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.bnd2 = nn.BatchNorm2d(32)
        
        self.lstm = nn.LSTM(100, LSTM_MEMORY, 1) 
        
        self.fc1 = nn.Linear(LSTM_MEMORY, 256)
        self.fc2 = nn.Linear(256, self.action_space)
        
        
    
    def forward(self, x, hidden_state, cell_state):
        # convolution 
        x = self.bnd1(self.relu(self.conv1(x)))
        x = self.bnd2(self.relu(self.conv2(x)))
        #print(x.shape)
        
     
        #LSTM 
        x = x.view(x.size(0), x.size(1), 100) 
        x, (next_hidden_state, next_cell_state) = self.lstm(x, (hidden_state, cell_state))
        x = x.view(x.size(0), -1) 
      #  print(x.shape)
        
        #fully connected 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x, next_hidden_state, next_cell_state
    
    def init_states(self) -> [Variable, Variable]:
        hidden_state = Variable(torch.zeros(1, 32, LSTM_MEMORY).cuda())
        cell_state = Variable(torch.zeros(1, 32, LSTM_MEMORY).cuda())
        return hidden_state, cell_state

    def reset_states(self, hidden_state, cell_state):
        hidden_state[:, :, :] = 0
        cell_state[:, :, :] = 0
        return hidden_state.detach(), cell_state.detach()