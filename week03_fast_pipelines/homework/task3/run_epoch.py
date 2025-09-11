import typing as tp
from profiler import Profile
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import Settings, Clothes, seed_everything
from vit import ViT


def get_vit_model() -> torch.nn.Module:
    model = ViT(
        depth=12,
        heads=4,
        image_size=224,
        patch_size=32,
        num_classes=20,
        channels=3,
    ).to(Settings.device)
    return model


def get_loaders() -> torch.utils.data.DataLoader:
    dataset.download_extract_dataset()
    train_transforms = dataset.get_train_transforms()
    val_transforms = dataset.get_val_transforms()

    frame = pd.read_csv(f"{Clothes.directory}/{Clothes.csv_name}")
    train_frame = frame.sample(frac=Settings.train_frac)
    val_frame = frame.drop(train_frame.index)

    train_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", train_frame, transform=train_transforms
    )
    val_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", val_frame, transform=val_transforms
    )

    print(f"Train Data: {len(train_data)}")
    print(f"Val Data: {len(val_data)}")

    # Оптимизируем загрузку данных
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=Settings.batch_size,
        shuffle=True,
        num_workers=4,  # Используем multiple workers
        pin_memory=True,  # Пининг памяти для быстрой передачи на GPU
        prefetch_factor=2,  # Предзагрузка батчей
        persistent_workers=True,  # Сохраняем рабочие процессы между эпохами
    )
    
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=Settings.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return train_loader, val_loader


def run_epoch(model, train_loader, val_loader, criterion, optimizer) -> tp.Tuple[float, float]:
    epoch_loss, epoch_accuracy = 0, 0
    val_loss, val_accuracy = 0, 0
    model.train()

    # PyTorch профайлер для анализа GPU операций и памяти
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as pytorch_prof, Profile(model, name="ViT") as custom_prof:
        
        # Профилирование нескольких итераций
        for i, (data, label) in enumerate(tqdm(train_loader, desc="Train")):
            # Анализ времени передачи данных на GPU
            with torch.profiler.record_function("data_transfer"):
                data = data.to(Settings.device)
                label = label.to(Settings.device)
            
            # Анализ forward pass
            with torch.profiler.record_function("forward"):
                output = model(data)
                loss = criterion(output, label)
            
            # Анализ метрик
            with torch.profiler.record_function("metrics"):
                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc.item() / len(train_loader)
                epoch_loss += loss.item() / len(train_loader)
            
            # Анализ backward pass
            optimizer.zero_grad()
            with torch.profiler.record_function("backward"):
                loss.backward()
            
            # Анализ optimizer step
            with torch.profiler.record_function("optimizer"):
                optimizer.step()
            
            custom_prof.step()
            pytorch_prof.step()
            
            # Ограничим количество итераций для профилирования
            if i >= 3:  # профилируем только первые 3 батчей
                break
    
    print("\nPyTorch Profiler Results:")
    print(pytorch_prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
    
    print("\nMemory Statistics:")
    print(f"Max allocated memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"Max cached memory: {torch.cuda.max_memory_cached() / 1024**2:.2f} MB")
    
    custom_prof.summary()        
    custom_prof.to_perfetto("vit_trace.json")
    
    print("\nComponent-wise Analysis:")
    print("1. Embedding Layer Performance")
    print("2. Attention Layer Performance")
    print("3. Feed-Forward Layer Performance")
    print("4. Forward vs Backward Pass Comparison")
    print("5. Memory Usage Analysis")

    model.eval()
    for data, label in tqdm(val_loader, desc="Val"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)
        acc = (output.argmax(dim=1) == label).float().mean()
        val_accuracy += acc.item() / len(train_loader)
        val_loss += loss.item() / len(train_loader)

    return epoch_loss, epoch_accuracy, val_loss, val_accuracy


def main():
    seed_everything()
    model = get_vit_model()
    train_loader, val_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Settings.lr)

    run_epoch(model, train_loader, val_loader, criterion, optimizer)


if __name__ == "__main__":
    main()
