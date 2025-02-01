from typing import Dict, Any, Optional, Callable, List
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from functools import wraps
import numpy as np

from mvi.config import MonitorConfig


class ModuleMonitor:
    """Monitors individual module statistics"""
    def __init__(self, name: str, writer: SummaryWriter, log_frequency: int = 1):
        self.name = name
        self.writer = writer
        self.log_frequency = log_frequency
        
    def log_tensor(self, name: str, tensor: torch.Tensor, step: int) -> None:
        """Log various statistics for a tensor"""
        if step % self.log_frequency != 0:
            return
            
        # Convert to numpy for statistics
        if tensor.numel() == 0:
            return
            
        tensor_np = tensor.detach().cpu().numpy()
        
        # Basic statistics
        self.writer.add_scalar(f"{self.name}/{name}/mean", np.mean(tensor_np), step)
        self.writer.add_scalar(f"{self.name}/{name}/std", np.std(tensor_np), step)
        self.writer.add_scalar(f"{self.name}/{name}/min", np.min(tensor_np), step)
        self.writer.add_scalar(f"{self.name}/{name}/max", np.max(tensor_np), step)
        
        # Add histogram every 100 steps
        if step % 100 == 0:
            self.writer.add_histogram(f"{self.name}/{name}/dist", tensor_np, step)


class AgentMonitor:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.writer = SummaryWriter(log_dir=config.log_dir) if config.enabled else None
        self.step = 0
        self.modules: Dict[str, ModuleMonitor] = {}
        
    def get_module_monitor(self, name: str) -> ModuleMonitor:
        """Get or create a module monitor"""
        if not self.writer:
            raise RuntimeError(f"Expected writer to be initialized. Got {self.writer}.")
        if name not in self.modules:
            self.modules[name] = ModuleMonitor(
                name, 
                self.writer, 
                self.config.log_frequency
            )
        return self.modules[name]
    
    def monitor_module(self, name: str):
        """Decorator for monitoring module inputs and outputs"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.writer:
                    return func(*args, **kwargs)
                    
                module_monitor = self.get_module_monitor(name)
                
                # Monitor inputs
                for i, arg in enumerate(args[1:]):  # Skip self
                    if isinstance(arg, torch.Tensor):
                        module_monitor.log_tensor(f"input_{i}", arg, self.step)
                
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        module_monitor.log_tensor(f"input_{k}", v, self.step)
                
                # Call function
                result = func(*args, **kwargs)
                
                # Monitor outputs
                if isinstance(result, torch.Tensor):
                    module_monitor.log_tensor("output", result, self.step)
                elif isinstance(result, tuple):
                    for i, r in enumerate(result):
                        if isinstance(r, torch.Tensor):
                            module_monitor.log_tensor(f"output_{i}", r, self.step)
                
                return result
            return wrapper
        return decorator

    def monitor_parameters(self, name: str, module: nn.Module) -> None:
        """Log parameter statistics for a module"""
        if not self.writer or self.step % self.config.log_frequency != 0:
            return
            
        module_monitor = self.get_module_monitor(name)
        
        for param_name, param in module.named_parameters():
            module_monitor.log_tensor(f"param/{param_name}", param, self.step)
            if param.grad is not None:
                module_monitor.log_tensor(f"grad/{param_name}", param.grad, self.step)
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log high-level metrics"""
        if not self.writer or self.step % self.config.log_frequency != 0:
            return
            
        for name, value in metrics.items():
            if name not in self.config.metrics:
                continue
                
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            
            self.writer.add_scalar(f"agent/{name}", value, self.step)
    
    def increment_step(self) -> None:
        self.step += 1
    
    def close(self) -> None:
        if self.writer:
            self.writer.close() 