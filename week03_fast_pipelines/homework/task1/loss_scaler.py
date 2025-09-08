from typing import Optional, List
import torch


class StaticLossScaler:
    def __init__(self, scale: float = 2**16):
        """Static loss scaler with fixed scaling factor.
        
        Args:
            scale: Fixed scaling factor to use
        """
        self.scale = scale
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss by multiplying with scale factor."""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients by dividing by scale factor."""
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
    
    def step(self, optimizer: torch.optim.Optimizer, *args, **kwargs) -> bool:
        """Unscale and step optimizer. Returns True if step was successful."""
        self.unscale_gradients(optimizer)
        optimizer.step()
        return True


class DynamicLossScaler:
    def __init__(
        self,
        init_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2**24,
    ):
        """Dynamic loss scaler that adjusts scaling factor based on gradient statistics.
        
        Args:
            init_scale: Initial scaling factor
            growth_factor: Factor to multiply scale by after growth_interval steps without inf/nan
            backoff_factor: Factor to multiply scale by when inf/nan gradients occur
            growth_interval: Number of consecutive steps without inf/nan before growing scale
            min_scale: Minimum allowed scale
            max_scale: Maximum allowed scale
        """
        self.cur_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.step_count = 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss by multiplying with current scale factor."""
        return loss * self.cur_scale
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients by dividing by current scale factor."""
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    param.grad.data.div_(self.cur_scale)
    
    def _check_inf_nan(self, optimizer: torch.optim.Optimizer) -> bool:
        """Check if any gradients are inf/nan."""
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    if torch.any(torch.isinf(param.grad)) or torch.any(torch.isnan(param.grad)):
                        return True
        return False
    
    def update_scale(self, has_inf_nan: bool) -> None:
        """Update scale based on presence of inf/nan gradients."""
        if has_inf_nan:
            self.cur_scale = max(self.cur_scale * self.backoff_factor, self.min_scale)
            self.step_count = 0
        else:
            self.step_count += 1
            if self.step_count >= self.growth_interval:
                self.cur_scale = min(self.cur_scale * self.growth_factor, self.max_scale)
                self.step_count = 0
    
    def step(self, optimizer: torch.optim.Optimizer, *args, **kwargs) -> bool:
        """Unscale gradients, check for inf/nan, update scale and step optimizer if valid.
        
        Returns:
            bool: Whether the optimization step was performed
        """
        self.unscale_gradients(optimizer)
        has_inf_nan = self._check_inf_nan(optimizer)
        
        if has_inf_nan:
            # Skip step if we have inf/nan gradients
            optimizer.zero_grad()
            self.update_scale(has_inf_nan=True)
            return False
        
        # Perform optimizer step and update scale
        optimizer.step()
        self.update_scale(has_inf_nan=False)
        return True
