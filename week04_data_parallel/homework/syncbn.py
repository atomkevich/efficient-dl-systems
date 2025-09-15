import contextlib
import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm
import contextlib

class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float):
        # Вычисляем статистики для текущего процесса
        batch_size = input.size(0)
        channel_size = input.size(1)
        spatial_size = input.numel() // (batch_size * channel_size)
        
        # Вычисляем среднее и variance для текущего процесса
        local_sum = input.sum(dim=[0, 2, 3]) if input.dim() == 4 else input.sum(dim=0)
        local_square_sum = (input ** 2).sum(dim=[0, 2, 3]) if input.dim() == 4 else (input ** 2).sum(dim=0)
        local_count = batch_size * spatial_size
        
        # Собираем статистики со всех процессов используя один коллективный вызов
        # Передаем local_count как тензор нужной размерности
        count_tensor = torch.full_like(local_sum, local_count)
        stats = torch.stack([local_sum, local_square_sum, count_tensor])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
        global_sum, global_square_sum, global_count = stats
        # Берем первый элемент global_count, так как все элементы одинаковые
        global_count = global_count[0]
        global_mean = global_sum / global_count
        global_var = (global_square_sum / global_count) - (global_mean ** 2)
        global_std = torch.sqrt(global_var + eps)
        
        # Обновляем running статистики
        if running_mean is not None and running_std is not None:
            running_mean.mul_(1 - momentum).add_(global_mean * momentum)
            running_std.mul_(1 - momentum).add_(global_std * momentum)
        
        # Нормализуем input
        normalized = (input - global_mean.view(-1, 1, 1) if input.dim() == 3 
                     else global_mean.view(1, -1, 1, 1)) / (global_std.view(-1, 1, 1) 
                     if input.dim() == 3 else global_std.view(1, -1, 1, 1))
        
        # Сохраняем для backward pass
        ctx.save_for_backward(input, global_mean, global_std)
        ctx.batch_size = batch_size
        ctx.spatial_size = spatial_size
        
        return normalized

    @staticmethod
    def backward(ctx, grad_output):
        input, mean, std = ctx.saved_tensors
        batch_size = ctx.batch_size
        spatial_size = ctx.spatial_size
        
        # Настраиваем размерности в зависимости от входных данных
        is_4d = input.dim() == 4
        mean_shape = (1, -1, 1, 1) if is_4d else (-1, 1, 1)
        reduce_dims = [0, 2, 3] if is_4d else [0]
        
        # Градиенты для нормализованного входа
        mean_view = mean.view(*mean_shape)
        std_view = std.view(*mean_shape)
        
        # Базовый градиент
        grad_input = grad_output / std_view
        
        # Вычисляем локальные градиенты
        local_grad_mean = grad_output.sum(dim=reduce_dims)
        local_grad_std = (grad_output * (input - mean_view)).sum(dim=reduce_dims)
        
        # Собираем градиенты со всех процессов (один вызов all_reduce)
        stats = torch.stack([local_grad_mean, local_grad_std])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        grad_mean, grad_std = stats
        
        # Нормализуем на общее количество элементов
        total_count = batch_size * spatial_size * dist.get_world_size()
        
        # Собираем финальный градиент
        grad_mean_term = grad_mean.view(*mean_shape) / total_count
        grad_std_term = (input - mean_view) * grad_std.view(*mean_shape) / (total_count * std_view)
        
        grad_input = grad_input - grad_mean_term - grad_std_term
        
        # None для остальных входных параметров forward (running_mean, running_std, eps, momentum)
        return grad_input, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        self.running_mean = torch.zeros(num_features)
        self.running_std = torch.ones(num_features)
        
        # Проверяем инициализацию distributed
        if not dist.is_available():
            raise RuntimeError("Distributed package is not available")
        with contextlib.suppress(RuntimeError):
            if not dist.is_initialized():
                raise RuntimeError("Default process group is not initialized")
       
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_contiguous():
            input = input.contiguous()
            
        if not self.training and self.track_running_stats:
            # Используем running статистики в режиме eval
            return (
                input - self.running_mean.view(-1, 1, 1)
                if input.dim() == 3
                else self.running_mean.view(1, -1, 1, 1)
            ) / (
                self.running_std.view(-1, 1, 1)
                if input.dim() == 3
                else self.running_std.view(1, -1, 1, 1)
            )

        return sync_batch_norm.apply(
            input, 
            self.running_mean if self.track_running_stats else None,
            self.running_std if self.track_running_stats else None,
            self.eps,
            self.momentum
        )

    @staticmethod
    def convert_sync_batchnorm(module):
        """
        Traverses the module and its children, replacing all BatchNorm layers with SyncBatchNorm layers.
        Similar to torch.nn.SyncBatchNorm.convert_sync_batchnorm but for our implementation.
        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
            )
            if module.track_running_stats:
                module_output.running_mean = module.running_mean
                module_output.running_std = module.running_var.sqrt()
                
        for name, child in module.named_children():
            module_output.add_module(
                name, SyncBatchNorm.convert_sync_batchnorm(child)
            )
        return module_output
