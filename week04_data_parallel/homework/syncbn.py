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
        # Определяем размерности входа
        batch_size = input.size(0)
        channel_size = input.size(1) if input.dim() > 1 else input.size(0)
        
        # Определяем размерности для суммирования в зависимости от размерности входа
        if input.dim() == 2:  # (N, C)
            sum_dims = [0]
            spatial_size = 1
            mean_shape = (1, -1)
        elif input.dim() == 3:  # (N, C, L)
            sum_dims = [0, 2]
            spatial_size = input.size(2)
            mean_shape = (-1, 1, 1)
        else:  # (N, C, H, W)
            sum_dims = [0, 2, 3]
            spatial_size = input.size(2) * input.size(3)
            mean_shape = (1, -1, 1, 1)
            
        # Вычисляем статистики для текущего процесса
        local_sum = input.sum(dim=sum_dims)
        local_square_sum = (input ** 2).sum(dim=sum_dims)
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
        # Применяем правильные размерности для нормализации
        if input.dim() == 2:  # (N, C)
            mean_view = global_mean.view(1, -1)
            std_view = global_std.view(1, -1)
        elif input.dim() == 3:  # (N, C, L)
            mean_view = global_mean.view(-1, 1, 1)
            std_view = global_std.view(-1, 1, 1)
        else:  # (N, C, H, W)
            mean_view = global_mean.view(1, -1, 1, 1)
            std_view = global_std.view(1, -1, 1, 1)
            
        normalized = (input - mean_view) / std_view
        
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
        
        # Определяем размерности в зависимости от входных данных
        if input.dim() == 2:  # (N, C)
            reduce_dims = [0]
            mean_shape = (1, -1)
        elif input.dim() == 3:  # (N, C, L)
            reduce_dims = [0, 2]
            mean_shape = (-1, 1, 1)
        else:  # (N, C, H, W)
            reduce_dims = [0, 2, 3]
            mean_shape = (1, -1, 1, 1)
        
        # Подготавливаем mean и std для broadcast
        mean_view = mean.view(*mean_shape)
        std_view = std.view(*mean_shape)
        
        # Базовый градиент через std
        grad_input = grad_output / std_view
        
        # Вычисляем локальные градиенты
        grad_output_sum = grad_output.sum(dim=reduce_dims)  # [C]
        dot_prod = ((input - mean_view) * grad_output).sum(dim=reduce_dims)  # [C]
        
        # Собираем градиенты в единый тензор для синхронизации
        combined_grad = torch.stack([grad_output_sum, dot_prod])  # [2, C]
        
        # Синхронизируем градиенты между процессами
        dist.all_reduce(combined_grad, op=dist.ReduceOp.SUM)
        
        # Нормализуем на общее количество элементов
        total_count = batch_size * spatial_size * dist.get_world_size()
        
        # Применяем градиенты с правильными размерностями
        grad_mean = combined_grad[0].view(*mean_shape) / total_count
        grad_std = combined_grad[1].view(*mean_shape) / (total_count * std_view)
        
        # Собираем финальный градиент
        grad_input = grad_input - grad_mean - (input - mean_view) * grad_std
        
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
        # Регистрируем буферы чтобы они автоматически перемещались на нужное устройство
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))
        
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
            if input.dim() == 4:          # N,C,H,W
                mean_view = self.running_mean.view(1, -1, 1, 1)
                std_view  = self.running_std.view(1, -1, 1, 1)
            elif input.dim() == 3:         # N,C,L
                mean_view = self.running_mean.view(1, -1, 1)
                std_view  = self.running_std.view(1, -1, 1)
            elif input.dim() == 2:         # N,C  (BatchNorm1d после FC)
                mean_view = self.running_mean.view(1, -1)
                std_view  = self.running_std.view(1, -1)
            else:
                raise ValueError(f"Unsupported input.dim() = {input.dim()}")

            normalized = (input - mean_view) / std_view
            # Используем running статистики в режиме eval
            return normalized

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
            # Создаем новый SyncBatchNorm
            module_output = SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
            )
            # Копируем статистики из оригинального BatchNorm
            if module.track_running_stats:
                with torch.no_grad():
                    module_output.running_mean.copy_(module.running_mean)
                    module_output.running_std.copy_(module.running_var.sqrt())
            # Копируем устройство
            module_output.to(module.weight.device if module.weight is not None else module.running_mean.device)
                
        for name, child in module.named_children():
            module_output.add_module(
                name, SyncBatchNorm.convert_sync_batchnorm(child)
            )
        return module_output
