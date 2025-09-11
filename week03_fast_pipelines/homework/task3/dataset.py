import os
import typing as tp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import Clothes, get_labels_dict


class ClothesDataset(Dataset):
    def __init__(self, folder_path, frame, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.frame = frame.set_index("image")
        self.img_list = list(self.frame.index.values)
        self.label2ix = get_labels_dict()
        
        # Предварительно вычисляем пути к файлам и метки
        self.img_paths = [f"{self.folder_path}/{img_name}.jpg" for img_name in self.img_list]
        self.labels = [self.label2ix[self.frame.loc[img_name]["label"]] for img_name in self.img_list]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # Используем предварительно вычисленные значения
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def download_extract_dataset():
    """Check if dataset exists in data directory"""
    if not os.path.exists(f"{Clothes.directory}/{Clothes.train_val_img_dir}"):
        raise FileNotFoundError(
            f"Dataset not found in {Clothes.directory}/{Clothes.train_val_img_dir}. "
            "Please place the dataset files in the data directory."
        )
    if not os.path.exists(f"{Clothes.directory}/{Clothes.csv_name}"):
        raise FileNotFoundError(
            f"CSV file not found in {Clothes.directory}/{Clothes.csv_name}. "
            "Please place the csv file in the data directory."
        )
    print("Dataset found in data directory")


def get_train_transforms() -> tp.Any:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Объединяем resize и crop
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(severity=3, mixture_width=3),  # Оптимизируем параметры AugMix
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Добавляем нормализацию
                              std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms() -> tp.Any:
    return transforms.Compose(
        [
            transforms.Resize(256),  # Меньше изменений размера
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Та же нормализация
                              std=[0.229, 0.224, 0.225]),
        ]
    )
