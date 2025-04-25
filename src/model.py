import torch
import torch.nn as nn

class WhisperAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim * 2)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # ensure x is on the same device as the model
        if x.device != self.fc1.weight.device:
            x = x.to(self.fc1.weight.device)
            
        # pass through first layer
        x = self.fc1(x)
        
        # split into gate and value
        gate, v = x.chunk(2, dim=-1)
        
        # apply gated activation
        x = self.activation(v) * torch.sigmoid(gate)
        
        # pass through second layer
        x = self.fc2(x)
        
        # apply dropout
        return self.dropout(x)