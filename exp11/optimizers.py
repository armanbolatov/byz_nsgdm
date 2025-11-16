# optimizers.py
"""Optimization algorithms for Byzantine-robust federated learning"""

import numpy as np


class BaselineOptimizer:
    """Standard Momentum SGD"""
    
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None
        self.name = "Baseline"
    
    def step(self, gradient):
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        
        if not np.all(np.isfinite(gradient)):
            gradient = np.zeros_like(gradient)
        
        self.velocity = self.momentum * self.velocity + gradient
        if not np.all(np.isfinite(self.velocity)):
            self.velocity = np.zeros_like(gradient)
        
        return -self.lr * self.velocity
    
    def reset(self):
        self.velocity = None


class BaselineDecayOptimizer:
    """Momentum SGD with learning rate decay"""
    
    def __init__(self, lr0=0.01, momentum=0.9, decay_power=0.5):
        self.lr0 = lr0
        self.momentum = momentum
        self.decay_power = decay_power
        self.velocity = None
        self.step_count = 0
        self.name = "Baseline-Decay"
    
    def step(self, gradient):
        self.step_count += 1
        
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        
        current_lr = self.lr0 / (self.step_count ** self.decay_power)
        
        if not np.all(np.isfinite(gradient)):
            gradient = np.zeros_like(gradient)
        
        self.velocity = self.momentum * self.velocity + gradient
        if not np.all(np.isfinite(self.velocity)):
            self.velocity = np.zeros_like(gradient)
        
        return -current_lr * self.velocity
    
    def reset(self):
        self.velocity = None
        self.step_count = 0


class ByzNSGDMOptimizer:
    """Byzantine-robust Normalized SGD with Momentum"""
    
    def __init__(self, lr0=0.01, momentum=0.9, decay_power=0.5):
        self.lr0 = lr0
        self.momentum = momentum
        self.decay_power = decay_power
        self.velocity = None
        self.step_count = 0
        self.name = "Byz-NSGDM"
    
    def step(self, gradient):
        self.step_count += 1
        
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        
        current_lr = self.lr0 / (self.step_count ** self.decay_power)
        
        if not np.all(np.isfinite(gradient)):
            gradient = np.zeros_like(gradient)
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-10:
            normalized_grad = gradient / grad_norm
        else:
            normalized_grad = np.zeros_like(gradient)
        
        self.velocity = self.momentum * self.velocity + normalized_grad
        if not np.all(np.isfinite(self.velocity)):
            self.velocity = np.zeros_like(gradient)
        
        return -current_lr * self.velocity
    
    def reset(self):
        self.velocity = None
        self.step_count = 0


def get_optimizer(optimizer_type, lr=0.01, momentum=0.9):
    """Factory function to get the appropriate optimizer"""
    if optimizer_type.lower() == "baseline":
        return BaselineOptimizer(lr=lr, momentum=momentum)
    elif optimizer_type.lower() == "baseline-decay":
        return BaselineDecayOptimizer(lr0=lr, momentum=momentum)
    elif optimizer_type.lower() == "byz-nsgdm" or optimizer_type.lower() == "byznsgdm":
        return ByzNSGDMOptimizer(lr0=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")