import torch
import torch.nn as nn

class GradientTrap(nn.Module):
    """Self-referential subnet for gradient absorption."""
    
    def __init__(self, dim, depth=3, alpha=0.1):
        super().__init__()
        self.depth = depth
        self.alpha = alpha
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
            for _ in range(depth)
        ])
        
    def forward(self, grad):
        x = grad
        for layer in self.layers:
            x = layer(x)
        return grad + self.alpha * x


class RGALayer(nn.Module):
    """Drop-in replacement for nn.Linear with gradient absorption."""
    
    def __init__(self, in_features, out_features, trap_depth=3, threshold=10.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.trap = GradientTrap(out_features, trap_depth)
        self.threshold = threshold
        
    def forward(self, x):
        out = self.linear(x)
        if self.training and out.requires_grad:
            out.register_hook(self._gradient_handler)
        return out
    
    def _gradient_handler(self, grad):
        grad_norm = grad.norm().item()
        if grad_norm > self.threshold:
            trapped = self.trap(grad)
            damping = self.threshold / grad_norm
            return trapped * damping
        return grad
