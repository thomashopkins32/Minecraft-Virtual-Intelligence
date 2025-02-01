from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
import torch

from mvi.config import MonitorConfig


class AgentMonitor:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.writer = SummaryWriter(log_dir=config.log_dir) if config.enabled else None
        self.step = 0
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics if monitoring is enabled and it's time to log."""
        if not self.config.enabled or self.step % self.config.log_frequency != 0:
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
        if self.writer is not None:
            self.writer.close() 