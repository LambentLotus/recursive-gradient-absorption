import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import RGALayer

class RGA_Network(nn.Module):
    """Deep network using RGA layers."""
    
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, depth=50):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList([
            RGALayer(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_layer(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output_layer(x)


class Standard_Network(nn.Module):
    """Baseline network without RGA."""
    
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, depth=20):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_layer(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output_layer(x)
