from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from math import floor

def get_dataset(transform):
    # train_set = datasets.Food101(
    #     root="./Data", download=True, transform=transform
    #     , split="train"
    # )
    # val_set = datasets.Food101(
    #     root="./Data", download=True, transform=transform
    #     , split="test"
    # )
    # return train_set, val_set
    dataset = ImageFolder("./Data/food-101/images", transform)
    val_size = floor((len(dataset)) * 0.10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return train_ds, val_ds

