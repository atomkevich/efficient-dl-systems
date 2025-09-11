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

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(f"{self.folder_path}/{img_name}.jpg").convert("RGB")
        img_transformed = self.transform(img)
        label = self.label2ix[self.frame.loc[img_name]["label"]]

        return img_transformed, label


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
            transforms.Resize((320, 320)),
            transforms.CenterCrop(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(),
            transforms.ToTensor(),
        ]
    )


def get_val_transforms() -> tp.Any:
    return transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
